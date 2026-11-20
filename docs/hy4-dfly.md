# Hy4-Flash DFly speculative decoding

Experimental support for the **DFly** drafter that ships with Tencent's Hy4-Flash, as a new speculative decoding type
(`draft-dfly`) in `llama-server` and the common speculative framework.

This fork is built on the `hy_v4_flash` architecture work from
[Little0o0/llama.cpp](https://github.com/Little0o0/llama.cpp) (ggml-org/llama.cpp PR #32517). The DFly drafter is a small
checkpoint-native drafter, separate from the model's MTP block: it drafts several tokens in a single pass from the target's
last hidden state, its own KV cache and its own sparse-attention selection.

Status: **experimental**. It works end to end and is lossless (the target verifies every token), but whether it makes
generation faster depends entirely on your setup, see [When it helps](#when-it-helps).

## How it works

DFly follows the algorithm described in Tencent's AngelSpec paper and implemented in vLLM and Basalt (see Credits):

1. **Gated fusion.** The embedding of the last accepted token (the *anchor*) is fused with the target's last-layer hidden
   state of the row that produced it.
2. **Five dense MLA layers** run over a block of rows (the anchor plus up to 5 mask tokens). They have no cache of their own:
   each layer attends the **target's** K cache of a fixed target layer (layers 25, 29, 33, 37, 41 for Hy4-Flash), restricted
   to the cells the target's indexer selected for the source row (*index share*), plus the block's own rows.
3. A **sequential hidden correction** and the target's own output head turn the block into draft tokens (greedy, or sampled, see below).
4. The target verifies the drafts as in any other speculative method.

In llama.cpp terms:

* The drafter is a separate model/context (`hy_v4_flash_dfly` architecture) **without memory**; it runs through
  `llama_encode()` and reads the target through `llama_context_params::ctx_other` (K cache, token embedding, output head).
* The target context taps the indexer selections of the draft layers (`llama_set_dfly_tap()`) and exposes its last hidden
  state (`llama_set_embeddings_nextn()`); the drafter reads them for the row that chose the next anchor.
* The draft token chain is kept on the device that holds the target's output head. The drafter's layers are placed on the
  devices of the target layers they read (layer split, default `0.42,0.58`, override with `LLAMA_DFLY_TS=a,b`), and the
  drafter file carries a copy of the target's token embedding so the chain does not hop between devices.

## Getting the drafter

Hy4-Flash GGUF repositories may ship the drafter as a `*.dfly2` file (it is a GGUF with architecture `hy_v4_flash_dfly` and
Hugging Face style tensor names). Convert it to the layout this fork loads:

```bash
python3 scripts/convert_dfly2_to_gguf.py  drafter.dfly2  drafter-llama.gguf \
        --vocab-from target-00001-of-0000N.gguf \
        --embd-from  target-00001-of-0000N.gguf  --verify
```

* `--vocab-from` copies the tokenizer and context length from the target GGUF (the drafter must be a valid standalone
  model: batches are checked against its vocabulary).
* `--embd-from` copies the target's token embedding into the drafter (`dfly.embd_head`), optional but recommended.
* The converter splits `kv_b_proj` into `attn_k_b` / `attn_v_b` the same way the `hy_v4_flash` converter does for the trunk.

The drafter weights are **not** part of this repository and keep their own licence (Tencent Hy License); check its terms
before use.

## Usage

```bash
llama-server -m target.gguf -md drafter-llama.gguf --spec-draft-n-max 5 --parallel 1 ...
```

* The speculative type is detected from the drafter's architecture; `--spec-type draft-dfly` selects it explicitly.
* `--spec-draft-n-max N` is the number of tokens drafted per step (default 3, at most 5, the number the drafter was trained
  for). 5 is the best choice for predictable output such as rewriting code; with the gate below it is safe to leave on.
* Use `--parallel 1` (a single sequence is supported).

### Adaptive acceptance gate

A verification step of several rows can cost several plain steps, so drafting only pays while most drafted tokens are
accepted. `draft-dfly` therefore keeps a running mean of the accepted drafts per drafted step and stops drafting while it is
below a threshold, falling back to plain decoding. It probes again after a back-off that starts at 8 steps and doubles up to
256, so drafting resumes when the output becomes predictable again (for example when the model starts rewriting code).
The state is reset at the start of every request.

* The threshold defaults to `0.66 * n_max` accepted drafts per drafted step, about the break-even point when a
  multi-row step costs roughly three plain steps.
* `LLAMA_DFLY_GATE_MIN_ACC=<value>` overrides it; `0` turns the gate off (always draft).
* On setups where a multi-row verify step is nearly free, lower the threshold (or turn the gate off).

### Sampled drafting

`LLAMA_DFLY_SAMPLE=1` makes the drafter sample instead of taking the argmax whenever the request has a temperature above 0.
Each draft step keeps its 32 best logits and picks one with Gumbel-max noise supplied by the host (an exact sample of
`softmax(logits / temp)` over those candidates, seeded from the request seed). The sampled token feeds the next step, and the
candidate distribution is handed to the server, which verifies by rejection sampling (accept with probability `min(1, p/q)`,
otherwise resample from the residual), so the output distribution is the target's. Greedy requests are unchanged. Sampled
drafting raises the acceptance of drafts when sampling noise, not draft quality, is what makes them miss.

### Limitations

* Single sequence only.
* The target only produces the indexer selections once the KV cache view holds at least `indexer_top_k` (2048) cells, so
  short conversations are not drafted. Decodes of more than 16 rows (prompt chunks) are not drafted either; the first
  step after a long prompt is a normal step.
* By default drafts are greedy: with sampling, a draft is accepted when the target's sampled token equals it (see
  [Sampled drafting](#sampled-drafting) for the alternative).
* The drafter needs the target's K cache; it is only usable with the `hy_v4_flash` target and its sparse layers.

## When it helps

Speculative decoding trades a more expensive verification step for several tokens per step. DFly only pays off when

* the verification step of several tokens costs little more than a single-token step, and
* a large share of the drafts is accepted (predictable output such as code continuation or structured text).

On setups where a multi-token step is much more expensive than a single-token step (for example when most expert weights
are streamed from host memory), or for hard-to-predict text, plain decoding can be as fast or faster; the adaptive gate keeps
the cost of the unpredictable parts small, so the gain on predictable parts (code edits, structured output) is kept. Measure with your own
workload: `llama-server` reports `draft_n` / `draft_n_accepted` in the response `timings`, and `llama-dfly-test` compares
plain and speculative decoding on a prompt and checks that the outputs match.

## Test tool

`llama-dfly-test` loads the target and the drafter, runs plain greedy (or sampled) decoding and speculative decoding on a
prompt file and reports acceptance statistics, per-step timings and whether the outputs are identical:

```bash
llama-dfly-test -m target.gguf --dfly-model drafter-llama.gguf --prompt-file prompt.txt \
                --n-gen 128 --n-draft 3 [--temp 0.9]
```

Use a prompt of a few thousand tokens (see the limitations above). Outputs can differ after a numerical near-tie: a
multi-row verification batch and a single-row decode use different kernels.

## Credits and licences

* [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and the `hy_v4_flash` support by Little0o0 (PR #32517).
* DFly / AngelSpec: Tencent. Reference implementations read while writing this code, no code copied:
  [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0, PR #68296) and [Basalt](https://github.com/jesdga95/basalt) (MIT).
* The drafter weights are Tencent's and are subject to the Tencent Hy License.

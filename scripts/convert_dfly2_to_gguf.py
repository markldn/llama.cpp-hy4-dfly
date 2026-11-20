#!/usr/bin/env python3
"""Convert Tencent's Hy4-flash DFly2 drafter (peasantsmith's `.dfly2`, a GGUF with HF tensor
names) into a GGUF with llama.cpp-style tensor names for the hy_v4_flash_dfly architecture.

- quantized projections (Q8_0) are copied byte for byte under their llama.cpp names
- kv_b_proj is split exactly like conversion/hy_v4_flash.py does for the trunk:
    view(n_head, 192+256, 512) -> k_b (transposed, [n_head, 512, 192]) and v_b ([n_head, 256, 512]),
  stored as F16 (dequantized from the file's Q8_0)
- everything else (norms, sink, fusion/hidden-correction tensors) keeps its type

Usage: convert_dfly2_to_gguf.py SRC.dfly2 DST.gguf [--vocab-from TARGET.gguf] [--embd-from TARGET.gguf] [--verify]
  --vocab-from  copy the tokenizer, vocab size and context length from the target GGUF (needed to load the drafter)
  --embd-from   copy the target token embedding as dfly.embd_head (keeps the draft chain on one device)
  run with the repository gguf-py on PYTHONPATH
"""
import re
import sys
from pathlib import Path

import numpy as np

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q
from gguf.quants import dequantize

ARCH = "hy_v4_flash_dfly"
QK_NOPE, V_HEAD = 192, 256

LAYER_MAP = {
    "input_layernorm.weight":               "blk.{b}.attn_norm.weight",
    "self_attn.q_a_proj.weight":            "blk.{b}.attn_q_a.weight",
    "self_attn.q_a_layernorm.weight":       "blk.{b}.attn_q_a_norm.weight",
    "self_attn.q_b_proj.weight":            "blk.{b}.attn_q_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight":  "blk.{b}.attn_kv_a_mqa.weight",
    "self_attn.kv_a_layernorm.weight":      "blk.{b}.attn_kv_a_norm.weight",
    "self_attn.learnable_sink_param":       "blk.{b}.attn_sinks.weight",
    "self_attn.linear_gate.weight":         "blk.{b}.attn_gate.weight",
    "self_attn.o_proj.weight":              "blk.{b}.attn_output.weight",
    "self_attn.post_proj_norm.weight":      "blk.{b}.post_attention_norm.weight",
    "post_attention_layernorm.weight":      "blk.{b}.ffn_norm.weight",
    "mlp.gate_proj.weight":                 "blk.{b}.ffn_gate.weight",
    "mlp.up_proj.weight":                   "blk.{b}.ffn_up.weight",
    "mlp.down_proj.weight":                 "blk.{b}.ffn_down.weight",
}
GLOBAL_MAP = {
    "final_norm.weight":                    "output_norm.weight",
    "mtp_fusion.eh_proj.weight":            "dfly.fuse_eh_proj.weight",
    "mtp_fusion.enorm.weight":              "dfly.fuse_enorm.weight",
    "mtp_fusion.hnorm.weight":              "dfly.fuse_hnorm.weight",
    "mtp_fusion.gate":                      "dfly.fuse_gate.weight",
    "hidden_correction.gate_proj.weight":   "dfly.hc_gate.weight",
    "hidden_correction.up_proj.weight":     "dfly.hc_up.weight",
    "hidden_correction.down_proj.weight":   "dfly.hc_down.weight",
    "hidden_correction.hidden_norm.weight": "dfly.hc_hidden_norm.weight",
    "hidden_correction.embed_norm.weight":  "dfly.hc_embed_norm.weight",
}
PREFIX = "model.dfly_layers."


def field(r, key):
    f = r.fields[key]
    vals = [f.parts[i] for i in f.data]
    if f.types and f.types[0].name == "STRING":
        return bytes(vals[0]).decode()
    if f.types and f.types[0].name == "ARRAY":
        return [int(v[0]) for v in vals]
    v = vals[0]
    return v.tolist()[0] if hasattr(v, "tolist") else v


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    verify = "--verify" in sys.argv
    r = GGUFReader(str(src))
    a = field(r, "general.architecture")
    assert a == ARCH, f"unexpected architecture {a}"
    p = lambda k: f"{ARCH}.{k}"  # noqa: E731

    n_layer = int(field(r, p("block_count")))
    n_embd = int(field(r, p("embedding_length")))
    n_head = int(field(r, p("attention.head_count")))
    kv_lora = int(field(r, p("attention.kv_lora_rank")))
    q_lora = int(field(r, p("attention.q_lora_rank")))

    w = GGUFWriter(str(dst), ARCH)
    w.add_name(field(r, "general.name"))
    w.add_type("model")
    w.add_uint32(p("block_count"), n_layer)
    w.add_uint32(p("embedding_length"), n_embd)
    w.add_uint32(p("attention.head_count"), n_head)
    w.add_uint32(p("attention.kv_lora_rank"), kv_lora)
    w.add_uint32(p("attention.q_lora_rank"), q_lora)
    # geometry the file does not state: 192 nope + 64 rope keys, 256 values, SwiGLU 9216 (from the tensors)
    ffn = next(int(t.shape[1]) for t in r.tensors if t.name.endswith("layers.0.mlp.gate_proj.weight"))
    w.add_uint32(p("feed_forward_length"), ffn)
    w.add_uint32(p("attention.key_length_mla"), QK_NOPE + 64)
    w.add_uint32(p("attention.value_length_mla"), V_HEAD)
    w.add_uint32(p("rope.dimension_count"), 64)
    w.add_float32(p("attention.layer_norm_rms_epsilon"), 1e-5)
    # the draft layers use the target's sparse-layer rope base (hy_v4_flash.rope.freq_base)
    w.add_float32(p("rope.freq_base"), 1.0e7)
    if "--vocab-from" in sys.argv:
        # copy the tokenizer so the drafter is a valid standalone model (batches are checked against its vocab)
        tr = GGUFReader(sys.argv[sys.argv.index("--vocab-from") + 1])
        n_vocab = None
        for key, f in tr.fields.items():
            if not key.startswith("tokenizer.ggml.") or key == "tokenizer.chat_template":
                continue
            tname = f.types[0].name
            if tname == "ARRAY":
                sub = f.types[1].name
                if sub == "STRING":
                    vals = [bytes(f.parts[i]).decode("utf-8", errors="surrogateescape") for i in f.data]
                else:
                    vals = [int(f.parts[i][0]) for i in f.data]
                if key == "tokenizer.ggml.tokens":
                    n_vocab = len(vals)
                w.add_array(key, vals)
            elif tname == "STRING":
                w.add_string(key, bytes(f.parts[f.data[0]]).decode())
            elif tname == "BOOL":
                w.add_bool(key, bool(f.parts[f.data[0]][0]))
            elif tname == "UINT32":
                w.add_uint32(key, int(f.parts[f.data[0]][0]))
            elif tname == "INT32":
                w.add_int32(key, int(f.parts[f.data[0]][0]))
            else:
                raise SystemExit(f"unhandled tokenizer key type {key} {tname}")
        assert n_vocab, "no tokens in the vocab source"
        w.add_uint32(p("vocab_size"), n_vocab)
        # generic keys the loader insists on, taken from the target (hy_v4_flash.*)
        ctx_f = tr.fields.get("hy_v4_flash.context_length")
        w.add_uint32(p("context_length"), int(ctx_f.parts[ctx_f.data[0]][0]) if ctx_f else 1048576)
    # drafter settings
    w.add_uint32(p("dfly.mask_token_id"), int(field(r, p("mask_token_id"))))
    w.add_uint32(p("dfly.num_speculative_tokens"), int(field(r, p("num_speculative_tokens"))))
    w.add_uint32(p("dfly.target_index_topk"), int(field(r, p("target_index_topk"))))
    w.add_array(p("dfly.target_kv_layers"), field(r, p("target_kv_layers")))
    w.add_array(p("dfly.target_index_layers"), field(r, p("target_index_layers")))
    w.add_string(p("dfly.draft_sample_method"), field(r, p("draft_sample_method")))
    w.add_string(p("dfly.mtp_fusion"), field(r, p("mtp_fusion")))
    w.add_bool(p("dfly.hidden_correction"), bool(field(r, p("hidden_correction"))))

    written = []
    for t in r.tensors:
        assert t.name.startswith(PREFIX), t.name
        rest = t.name[len(PREFIX):]
        if rest.startswith("confidence_head"):   # steers the draft policy only; the references do not load it
            continue
        m = re.match(r"(\d+)\.(.*)", rest)
        if m:
            b, tail = int(m.group(1)), m.group(2)
            if tail == "self_attn.kv_b_proj.weight":
                f = dequantize(t.data, t.tensor_type).astype(np.float32) if t.tensor_type != Q.F32 else np.array(t.data, dtype=np.float32)
                assert f.shape == (n_head * (QK_NOPE + V_HEAD), kv_lora), f.shape
                kvb = f.reshape(n_head, QK_NOPE + V_HEAD, kv_lora)
                k_b = np.ascontiguousarray(kvb[:, :QK_NOPE, :].transpose(0, 2, 1)).astype(np.float16)  # [h, 512, 192]
                v_b = np.ascontiguousarray(kvb[:, QK_NOPE:, :]).astype(np.float16)                       # [h, 256, 512]
                w.add_tensor(f"blk.{b}.attn_k_b.weight", k_b)
                w.add_tensor(f"blk.{b}.attn_v_b.weight", v_b)
                written += [(f"blk.{b}.attn_k_b.weight", k_b.shape), (f"blk.{b}.attn_v_b.weight", v_b.shape)]
                continue
            name = LAYER_MAP[tail].format(b=b)
        else:
            name = GLOBAL_MAP[rest]
        if t.tensor_type in (Q.F32, Q.F16, Q.BF16):
            arr = np.array(t.data)
            w.add_tensor(name, arr)
            written.append((name, arr.shape))
        else:
            # quantized: copy the raw blocks; shape given as the byte shape, the writer derives the element shape
            w.add_tensor(name, np.array(t.data), raw_shape=t.data.shape, raw_dtype=t.tensor_type)
            written.append((name, tuple(int(x) for x in t.shape[::-1])))

    if "--embd-from" in sys.argv:
        # a copy of the target's token embedding for the drafter's sequential draft chain, so the chain stays on
        # the device that holds the output head (no hop per draft step)
        er = GGUFReader(sys.argv[sys.argv.index("--embd-from") + 1])
        et = next(t for t in er.tensors if t.name == "token_embd.weight")
        w.add_tensor("dfly.embd_head.weight", np.array(et.data), raw_shape=et.data.shape, raw_dtype=et.tensor_type)
        written.append(("dfly.embd_head.weight", tuple(int(x) for x in et.shape[::-1])))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {dst} with {len(written)} tensors")

    if verify:
        r2 = GGUFReader(str(dst))
        by = {t.name: t for t in r2.tensors}
        assert len(by) == len(written), (len(by), len(written))
        src_by = {t.name: t for t in r.tensors}
        bad = 0
        for t in r.tensors:
            rest = t.name[len(PREFIX):]
            if rest.startswith("confidence_head"):
                continue
            m = re.match(r"(\d+)\.(.*)", rest)
            if m and m.group(2) == "self_attn.kv_b_proj.weight":
                continue
            nm = LAYER_MAP[m.group(2)].format(b=int(m.group(1))) if m else GLOBAL_MAP[rest]
            t2 = by[nm]
            same = (t2.tensor_type == t.tensor_type and [int(x) for x in t2.shape] == [int(x) for x in t.shape]
                    and np.array_equal(np.array(t2.data), np.array(t.data)))
            if not same:
                bad += 1
                print("MISMATCH", nm, t2.tensor_type, t.tensor_type, list(t2.shape), list(t.shape))
        for b in range(n_layer):
            for nm in (f"blk.{b}.attn_k_b.weight", f"blk.{b}.attn_v_b.weight"):
                print(nm, [int(x) for x in by[nm].shape], by[nm].tensor_type.name) if b == 0 else None
        print("verify:", "OK" if not bad else f"{bad} mismatches")


if __name__ == "__main__":
    main()

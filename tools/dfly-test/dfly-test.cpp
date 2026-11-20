// llama-dfly-test: greedy decoding of a target model with and without the Hy4-flash DFly drafter.
//
//   llama-dfly-test -m target.gguf -c 262144 -ngl 999 -ncmoe 24 ... --dfly-model drafter.gguf --prompt-file text.txt \
//                   [--n-gen 128] [--n-draft 5] [--no-baseline] [--no-spec]
//
// Reports tokens/s of both runs, the acceptance statistics, and whether the two outputs are identical.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static void print_usage(int, char **) {}

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static llama_token argmax_row(const float * logits, int32_t n_vocab) {
    int32_t best = 0;
    float   bv   = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > bv) {
            bv   = logits[i];
            best = i;
        }
    }
    return best;
}

static llama_token sample_row(const float * logits, int32_t n_vocab, float temp, std::mt19937 & rng) {
    if (temp <= 0.0f) {
        return argmax_row(logits, n_vocab);
    }
    float mx = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        mx = std::max(mx, logits[i]);
    }
    static std::vector<double> p;
    p.resize(n_vocab);
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        p[i] = std::exp((double) (logits[i] - mx) / temp);
        sum += p[i];
    }
    double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
    double c = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        c += p[i];
        if (r <= c) {
            return i;
        }
    }
    return n_vocab - 1;
}

static float top2_margin(const float * logits, int32_t n_vocab) {
    float m1 = -INFINITY, m2 = -INFINITY;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float v = logits[i];
        if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) { m2 = v; }
    }
    return m1 - m2;
}

struct dfly_args {
    std::string dfly_model;
    std::string prompt_file;
    int  n_gen      = 128;
    int  n_draft    = 5;
    bool baseline   = true;
    bool spec       = true;
    std::string dfly_ts = "0.42,0.58"; // drafter layer split over the two GPUs
    float temp      = 0.0f; // 0 = greedy
    int   seed      = 1;
    int  draft_loop = 0;   // >0: only time N drafter calls (after one prefill + one target decode), then exit
    int  tail_rows  = 8; // prompt rows decoded with the taps on
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // strip our own options
    dfly_args da;
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dfly-model" && i + 1 < argc)        { da.dfly_model  = argv[++i]; }
        else if (a == "--prompt-file" && i + 1 < argc)  { da.prompt_file = argv[++i]; }
        else if (a == "--n-gen" && i + 1 < argc)        { da.n_gen       = atoi(argv[++i]); }
        else if (a == "--n-draft" && i + 1 < argc)      { da.n_draft     = atoi(argv[++i]); }
        else if (a == "--no-baseline")                  { da.baseline    = false; }
        else if (a == "--no-spec")                      { da.spec        = false; }
        else if (a == "--temp" && i + 1 < argc)         { da.temp        = (float) atof(argv[++i]); }
        else if (a == "--seed" && i + 1 < argc)         { da.seed        = atoi(argv[++i]); }
        else if (a == "--dfly-ts" && i + 1 < argc)      { da.dfly_ts     = argv[++i]; }
        else if (a == "--draft-loop" && i + 1 < argc)   { da.draft_loop  = atoi(argv[++i]); }
        else                                            { rest.push_back(argv[i]); }
    }
    int rargc = (int) rest.size();

    common_params params;
    common_init();
    if (!common_params_parse(rargc, rest.data(), params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }
    if (da.prompt_file.empty() || (da.spec && da.dfly_model.empty())) {
        fprintf(stderr, "need --prompt-file and (unless --no-spec) --dfly-model\n");
        return 1;
    }
    da.n_draft = std::max(1, std::min(da.n_draft, 5));

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (!model) {
        fprintf(stderr, "failed to load the target\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = common_context_params_to_llama(params);
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the target context\n");
        return 1;
    }

    // prompt
    std::string text;
    {
        std::ifstream f(da.prompt_file);
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
    }
    std::vector<llama_token> prompt = common_tokenize(ctx, text, true, true);
    fprintf(stderr, "prompt: %zu tokens\n", prompt.size());
    if (prompt.size() < 2200) {
        fprintf(stderr, "warning: prompts under ~2200 tokens leave the cache view below the indexer top-k, so no taps\n");
    }

    auto * mem = llama_get_memory(ctx);
    const int32_t n_chunk = std::max(1, (int) cp.n_batch);
    common_batch batch(ctx);

    // warm-up like llama-batched-bench: 16 tokens, so the first timed step does not pay the one-time costs
    {
        for (int i = 0; i < 16; ++i) {
            batch.add(prompt[i % prompt.size()], i, 0, i == 15);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "warm-up failed\n");
            return 1;
        }
        llama_synchronize(ctx);
        batch.clear();
        llama_memory_clear(mem, true);
    }

    auto process = [&](llama_context * c, llama_process_type t, common_batch & b) -> bool {
        const int ret = llama_process(c, t, b.get());
        if (ret != 0) {
            fprintf(stderr, "llama_process failed: %d\n", ret);
            return false;
        }
        return true;
    };

    // decode prompt[0 .. n_head) with taps off, then the last tail rows (n_prompt - tail .. n_prompt) with taps on
    const int n_prompt = (int) prompt.size();
    const int tail     = std::min(da.tail_rows, n_prompt);
    const uint32_t tap_layers[5] = {25, 29, 33, 37, 41};

    auto prefill = [&](bool taps) -> int /* batch row of the last prompt token in the last decode */ {
        llama_memory_clear(mem, true);
        llama_set_dfly_tap(ctx, nullptr, 0);
        llama_set_embeddings_nextn(ctx, false, false);

        for (int i = 0; i < n_prompt - tail; i += n_chunk) {
            const int n = std::min(n_chunk, n_prompt - tail - i);
            batch.clear();
            for (int j = 0; j < n; ++j) {
                batch.add(prompt[i + j], i + j, 0, false);
            }
            if (!process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                return -1;
            }
        }
        if (taps) {
            llama_set_dfly_tap(ctx, tap_layers, 5);
            llama_set_embeddings_nextn(ctx, true, false);
        }
        batch.clear();
        for (int j = n_prompt - tail; j < n_prompt; ++j) {
            batch.add(prompt[j], j, 0, j == n_prompt - 1);
        }
        if (!process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            return -1;
        }
        return tail - 1;
    };

    std::vector<llama_token> out_base;
    std::vector<llama_token> out_spec;
    std::vector<float> margin_spec; // top-1 minus top-2 target logit at each emitted token (after the first)
    double t_base = 0.0;
    double t_spec = 0.0;

    // ---- baseline
    if (da.baseline) {
        const int row = prefill(false);
        if (row < 0) {
            return 1;
        }
        std::mt19937 rng_base(da.seed);
        llama_token tok = sample_row(llama_get_logits_ith(ctx, row), n_vocab, da.temp, rng_base);
        out_base.push_back(tok);
        const auto t0 = clk::now();
        int pos = n_prompt;
        std::vector<double> step_ms;
        while ((int) out_base.size() < da.n_gen) {
            const auto ts = clk::now();
            batch.clear();
            batch.add(tok, pos, 0, true);
            if (!process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                return 1;
            }
            tok = sample_row(llama_get_logits_ith(ctx, 0), n_vocab, da.temp, rng_base);
            out_base.push_back(tok);
            ++pos;
            step_ms.push_back(ms_since(ts));
        }
        t_base = ms_since(t0);
        {
            std::vector<double> sorted = step_ms;
            std::sort(sorted.begin(), sorted.end());
            double first8 = 0.0;
            for (size_t i = 0; i < std::min<size_t>(8, step_ms.size()); ++i) { first8 += step_ms[i]; }
            fprintf(stderr, "baseline step ms: first8 mean %.0f, median %.0f, min %.0f, max %.0f\n",
                    first8 / std::min<size_t>(8, step_ms.size()), sorted[sorted.size()/2], sorted.front(), sorted.back());
        }
        fprintf(stderr, "baseline: %zu tokens in %.0f ms = %.2f tok/s\n", out_base.size() - 1, t_base,
                1000.0 * (double) (out_base.size() - 1) / t_base);
    }

    // ---- speculative
    if (da.spec) {
        llama_model_params dp = llama_model_default_params();
        dp.n_gpu_layers = 999;
        dp.split_mode   = LLAMA_SPLIT_MODE_LAYER;
        dp.main_gpu     = 0;
        std::vector<float> dts(llama_max_devices(), 0.0f);
        {
            std::stringstream ss(da.dfly_ts);
            std::string tok;
            size_t i = 0;
            while (std::getline(ss, tok, ',') && i < dts.size()) {
                dts[i++] = (float) atof(tok.c_str());
            }
        }
        dp.tensor_split = dts.data();
        llama_model * dmodel = llama_model_load_from_file(da.dfly_model.c_str(), dp);
        if (!dmodel) {
            fprintf(stderr, "failed to load the drafter\n");
            return 1;
        }
        llama_context_params dc = llama_context_default_params();
        dc.n_ctx        = 64;
        dc.n_batch      = 16;
        dc.n_ubatch     = 16;
        dc.n_seq_max    = 1;
        dc.embeddings   = true;
        dc.pooling_type = LLAMA_POOLING_TYPE_NONE;
        dc.ctx_other    = ctx;
        dc.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context * dctx = llama_init_from_model(dmodel, dc);
        if (!dctx) {
            fprintf(stderr, "failed to create the drafter context\n");
            return 1;
        }
        const llama_token mask_id = 120009;
        common_batch bd(dctx);

        const int row0 = prefill(true);
        if (row0 < 0) {
            return 1;
        }
        std::mt19937 rng_spec(da.seed);
        llama_token anchor = sample_row(llama_get_logits_ith(ctx, row0), n_vocab, da.temp, rng_spec);
        out_spec.push_back(anchor);
        int pos     = n_prompt;  // position of the anchor (not decoded yet)
        int src_row = row0;      // row of the target's last decode that chose the anchor

        if (da.draft_loop > 0) {
            // time the drafter alone: same inputs every call
            std::vector<double> t_sub, t_wait;
            for (int it = 0; it < da.draft_loop + 3; ++it) {
                auto t1 = clk::now();
                llama_set_dfly_src_row(dctx, src_row);
                bd.clear();
                bd.add(anchor, pos, 0, true);
                for (int j = 0; j < da.n_draft; ++j) {
                    bd.add(mask_id, pos + 1 + j, 0, true);
                }
                if (!process(dctx, LLAMA_PROCESS_TYPE_ENCODE, bd)) {
                    return 1;
                }
                const double sub = ms_since(t1);
                auto t2 = clk::now();
                llama_synchronize(dctx);
                const double wait = ms_since(t2);
                if (it >= 3) { t_sub.push_back(sub); t_wait.push_back(wait); }
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; };
            fprintf(stderr, "draft-loop: %d calls, process() median %.2f ms, synchronize() median %.2f ms, total median %.2f ms\n",
                    da.draft_loop, med(t_sub), med(t_wait), med(t_sub) + med(t_wait));
            return 0;
        }

        const int N = da.n_draft;
        std::vector<int> hist(N + 2, 0);
        double t_draft = 0.0;
        double t_verify = 0.0;
        int n_steps = 0;

        const auto t0 = clk::now();
        while ((int) out_spec.size() < da.n_gen) {
            // draft
            auto td = clk::now();
            llama_set_dfly_src_row(dctx, src_row);
            bd.clear();
            bd.add(anchor, pos, 0, true);
            for (int j = 0; j < N; ++j) {
                bd.add(mask_id, pos + 1 + j, 0, true);
            }
            if (!process(dctx, LLAMA_PROCESS_TYPE_ENCODE, bd)) {
                return 1;
            }
            llama_synchronize(dctx);
            std::vector<llama_token> draft(N);
            for (int j = 0; j < N; ++j) {
                const float * e = llama_get_embeddings_ith(dctx, 1 + j);
                draft[j] = (llama_token) std::lround(e[0]);
            }
            t_draft += ms_since(td);

            // verify: [anchor, d0..dN-1]
            auto tv = clk::now();
            batch.clear();
            batch.add(anchor, pos, 0, true);
            for (int j = 0; j < N; ++j) {
                batch.add(draft[j], pos + 1 + j, 0, true);
            }
            if (!process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                return 1;
            }
            // lazy, exact: sample the target row by row, keep going only while the sample equals the draft
            std::vector<llama_token> pred(N + 1, -1);
            std::vector<float> pmargin(N + 1, 0.0f);
            int acc = 0;
            for (int i = 0; i <= N; ++i) {
                const float * lg = llama_get_logits_ith(ctx, i);
                pred[i] = sample_row(lg, n_vocab, da.temp, rng_spec);
                if (da.temp <= 0.0f) {
                    pmargin[i] = top2_margin(lg, n_vocab);
                }
                if (i < N && pred[i] == draft[i]) {
                    ++acc;
                } else {
                    break;
                }
            }
            t_verify += ms_since(tv);

            for (int j = 0; j < acc; ++j) {
                out_spec.push_back(draft[j]);
                margin_spec.push_back(pmargin[j]);
            }
            out_spec.push_back(pred[acc]);
            margin_spec.push_back(pmargin[acc]);
            if (acc < N) {
                llama_memory_seq_rm(mem, 0, pos + acc + 1, -1);
            }
            hist[acc]++;
            ++n_steps;

            anchor  = pred[acc];
            pos    += acc + 1;
            src_row = acc;
        }
        t_spec = ms_since(t0);
        out_spec.resize(da.n_gen);

        double mean_acc = 0.0;
        for (int a = 0; a <= N; ++a) {
            mean_acc += (double) a * hist[a];
        }
        mean_acc /= std::max(1, n_steps);
        fprintf(stderr, "spec: %d tokens in %.0f ms = %.2f tok/s | steps %d, mean accepted drafts %.2f (of %d)\n",
                da.n_gen - 1, t_spec, 1000.0 * (double) (da.n_gen - 1) / t_spec, n_steps, mean_acc, N);
        fprintf(stderr, "spec: per step draft %.1f ms, verify %.1f ms; accepted histogram:", t_draft / n_steps, t_verify / n_steps);
        for (int a = 0; a <= N; ++a) {
            fprintf(stderr, " %d:%d", a, hist[a]);
        }
        fprintf(stderr, "\n");
    }

    if (da.baseline && da.spec && da.temp <= 0.0f) {
        int first_diff = -1;
        const int n = (int) std::min(out_base.size(), out_spec.size());
        for (int i = 0; i < n; ++i) {
            if (out_base[i] != out_spec[i]) {
                first_diff = i;
                break;
            }
        }
        fprintf(stderr, "outputs %s", first_diff < 0 ? "IDENTICAL" : "DIFFER");
        if (first_diff >= 0) {
            fprintf(stderr, " (first difference at generated token %d)", first_diff);
            if (first_diff >= 1 && (size_t) (first_diff - 1) < margin_spec.size()) {
                fprintf(stderr, ", target top1-top2 logit margin there %.3f", margin_spec[first_diff - 1]);
            }
        }
        fprintf(stderr, "\nspeedup: %.2fx\n", t_base / t_spec);
    }
    if (da.baseline && da.spec && da.temp > 0.0f) {
        fprintf(stderr, "sampled (temp %.2f): baseline %.2f tok/s, spec %.2f tok/s, speedup %.2fx\n", da.temp,
                1000.0 * (double) (out_base.size() - 1) / t_base, 1000.0 * (double) (da.n_gen - 1) / t_spec, t_base / t_spec);
    }

    const auto & o = !out_spec.empty() ? out_spec : out_base;
    std::string s;
    for (size_t i = 0; i < std::min<size_t>(o.size(), 60); ++i) {
        s += common_token_to_piece(ctx, o[i]);
    }
    fprintf(stderr, "first tokens: %s\n", s.c_str());

    llama_backend_free();
    return 0;
}

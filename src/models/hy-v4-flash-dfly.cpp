#include "models.h"

#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-memory-idx-iswa.h"

#include <cmath>
#include <vector>

// Hy4-flash DFly2 drafter (Tencent AngelSpec). Five dense MLA layers (hy_v4 gated MLA with sink, no hyper-connections,
// plain SwiGLU) that attend the TARGET's cached K/V of layer dfly_kv_layers[i], restricted to the cells the target's own
// indexer selected for the source row (index share), plus the block's own rows. A block is the anchor token and n mask
// tokens; the anchor's embedding is fused with the target's last hidden state, and the n mask rows go through a
// sequential hidden correction before the target's output head picks the drafts (greedy). The drafter keeps no cache.
//
// The context has no memory: it runs through llama_encode(), reads the target through cparams.ctx_other (its K cache, the
// tapped selections and h_nextn host rows, its token embedding and output head) and returns the n draft token ids as the
// embeddings output ([1, n_rows] floats, row 0 unused).
//
// refs: vLLM vllm/models/hy_v4/nvidia/dfly*.py, Basalt src/models/hy4flash/dfly.hpp

void llama_model_hy_v4_flash_dfly::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);

    ml.get_key(LLM_KV_DFLY_MASK_TOKEN_ID,     dfly_mask_token_id);
    ml.get_key(LLM_KV_DFLY_NUM_SPEC_TOKENS,   dfly_n_spec);
    ml.get_key(LLM_KV_DFLY_TARGET_INDEX_TOPK, dfly_index_topk);
    ml.get_key(LLM_KV_DFLY_HIDDEN_CORRECTION, dfly_hidden_correction, false);
    ml.get_arr(LLM_KV_DFLY_TARGET_KV_LAYERS,    dfly_kv_layers);
    ml.get_arr(LLM_KV_DFLY_TARGET_INDEX_LAYERS, dfly_idx_layers);

    GGML_ASSERT(hparams.is_mla());

    if (dfly_kv_layers.size() != hparams.n_layer() || dfly_idx_layers.size() != hparams.n_layer()) {
        throw std::runtime_error("hy_v4_flash_dfly: target_kv_layers / target_index_layers must list one target layer per draft layer");
    }
    // this graph reads one selection per draft layer from the tap of the same target layer
    for (size_t i = 0; i < dfly_kv_layers.size(); ++i) {
        if (dfly_idx_layers[i] != dfly_kv_layers[i]) {
            throw std::runtime_error("hy_v4_flash_dfly: only target_index_layers == target_kv_layers is supported");
        }
    }

    // the embeddings output carries the draft token ids: one float per row
    hparams.n_embd_out_impl = 1;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_hy_v4_flash_dfly::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    // the token embedding and the output head are the target's (graph: model_other); the drafter file carries neither
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    dfly_fuse_eh_proj   = create_tensor(tn(LLM_TENSOR_DFLY_FUSE_EH_PROJ,   "weight"), {2 * n_embd, n_embd}, 0);
    dfly_fuse_enorm     = create_tensor(tn(LLM_TENSOR_DFLY_FUSE_ENORM,     "weight"), {n_embd}, 0);
    dfly_fuse_hnorm     = create_tensor(tn(LLM_TENSOR_DFLY_FUSE_HNORM,     "weight"), {n_embd}, 0);
    dfly_fuse_gate      = create_tensor(tn(LLM_TENSOR_DFLY_FUSE_GATE,      "weight"), {n_embd}, 0);
    dfly_hc_gate        = create_tensor(tn(LLM_TENSOR_DFLY_HC_GATE,        "weight"), {2 * n_embd, n_embd}, 0);
    dfly_hc_up          = create_tensor(tn(LLM_TENSOR_DFLY_HC_UP,          "weight"), {2 * n_embd, n_embd}, 0);
    dfly_hc_down        = create_tensor(tn(LLM_TENSOR_DFLY_HC_DOWN,        "weight"), {n_embd, n_embd}, 0);
    dfly_hc_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLY_HC_HIDDEN_NORM, "weight"), {n_embd}, 0);
    dfly_hc_embed_norm  = create_tensor(tn(LLM_TENSOR_DFLY_HC_EMBED_NORM,  "weight"), {n_embd}, 0);
    // optional: lets the draft chain look up embeddings on the head's device (falls back to the target's)
    dfly_embd_head      = create_tensor(tn(LLM_TENSOR_DFLY_EMBD_HEAD,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < (int) hparams.n_layer_all; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_l = hparams.n_head(i);
        const int64_t n_ff_l   = hparams.n_ff(i);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_sinks     = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,     "weight", i), {n_head_l}, 0);

        layer.wq_a           = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,       "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
        layer.wq_b           = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,       "weight", i), {q_lora_rank, n_head_l * n_embd_head_k_mla}, 0);
        layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
        layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head_l}, 0);
        layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head_l}, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_head_l * n_embd_head_v_mla, n_embd}, 0);
        layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", i), {n_embd, n_head_l * n_embd_head_v_mla}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.ffn_gate       = create_tensor(tn(LLM_TENSOR_FFN_GATE,       "weight", i), {n_embd, n_ff_l}, 0);
        layer.ffn_up         = create_tensor(tn(LLM_TENSOR_FFN_UP,         "weight", i), {n_embd, n_ff_l}, 0);
        layer.ffn_down       = create_tensor(tn(LLM_TENSOR_FFN_DOWN,       "weight", i), {n_ff_l, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_hy_v4_flash_dfly::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

namespace {

// host inputs of one drafter pass: the block's token ids, the target's hidden state of the source row, the target's
// tapped selections of that row (one per draft layer) and the causal mask among the block's own rows
class llm_graph_input_dfly : public llm_graph_input_i {
public:
    llm_graph_input_dfly(const llama_cparams & cparams, int32_t n_tap) : cparams(cparams), n_tap(n_tap) {}

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_rows = ubatch->n_tokens;
        GGML_ASSERT(ubatch->token != nullptr);
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_rows*sizeof(int32_t));

        llama_context * tgt = cparams.ctx_other;
        GGML_ASSERT(tgt != nullptr);
        tgt->synchronize();

        const int32_t row = cparams.dfly_src_row;

        const float * h = tgt->get_embeddings_nextn();
        GGML_ASSERT(h != nullptr && "dfly: the target context needs llama_set_embeddings_nextn(ctx, true, false)");
        ggml_backend_tensor_set(h_src, h + (size_t) row*h_src->ne[0], 0, ggml_nbytes(h_src));

        for (int32_t s = 0; s < n_tap; ++s) {
            int32_t k = 0;
            int32_t rows = 0;
            const int32_t * idx = nullptr;
            const float   * val = nullptr;
            const bool ok = tgt->get_dfly_tap(s, &k, &rows, &idx, &val);
            GGML_ASSERT(ok && "dfly: no tapped selection (target decode too short or taps off)");
            GGML_ASSERT(k == sel_idx->ne[0] && row >= 0 && row < rows);
            ggml_backend_tensor_set(sel_idx, idx + (size_t) row*k, (size_t) s*k*sizeof(int32_t), (size_t) k*sizeof(int32_t));
            ggml_backend_tensor_set(sel_val, val + (size_t) row*k, (size_t) s*k*sizeof(float),   (size_t) k*sizeof(float));
        }

        // [key, 1, query]: a row sees the rows up to itself
        std::vector<float> m((size_t) n_rows*n_rows);
        for (int64_t q = 0; q < n_rows; ++q) {
            for (int64_t key = 0; key < n_rows; ++key) {
                m[(size_t) q*n_rows + key] = key <= q ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(loc_mask, m.data(), 0, m.size()*sizeof(float));
    }

    // the graph only depends on the row count and the width of the tapped selections, not on their values
    bool can_reuse(const llm_graph_params & params) override {
        int32_t k = 0;
        int32_t rows = 0;
        const int32_t * ip = nullptr;
        const float   * vp = nullptr;
        if (!cparams.ctx_other->get_dfly_tap(0, &k, &rows, &ip, &vp)) {
            return false;
        }
        return tokens != nullptr && tokens->ne[0] == (int64_t) params.ubatch.n_tokens && sel_idx != nullptr && sel_idx->ne[0] == k;
    }

    ggml_tensor * tokens   = nullptr; // I32 [n_rows]
    ggml_tensor * h_src    = nullptr; // F32 [n_embd]
    ggml_tensor * sel_idx  = nullptr; // I32 [k, n_tap]
    ggml_tensor * sel_val  = nullptr; // F32 [k, n_tap]
    ggml_tensor * loc_mask = nullptr; // F32 [n_rows, 1, n_rows]

    const llama_cparams & cparams;
    int32_t n_tap;
};

} // namespace

llama_model_hy_v4_flash_dfly::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
        llm_graph_context(params) {
    const auto & mdl = static_cast<const llama_model_hy_v4_flash_dfly &>(model_base);

    GGML_ASSERT(cparams.ctx_other != nullptr);
    const llama_model * model_t = llama_get_model(cparams.ctx_other);
    GGML_ASSERT(model_t != nullptr && model_t->tok_embd != nullptr && model_t->output != nullptr);

    const int64_t n_rows  = n_tokens;     // the anchor and n mask tokens
    const int64_t n_masks = n_rows - 1;
    GGML_ASSERT(n_masks >= 0 && n_rows <= 16);

    const int64_t n_layers = hparams.n_layer();
    GGML_ASSERT(n_layers == (int64_t) mdl.dfly_kv_layers.size());

    // width of the tapped selections: from the last target decode, the full top-k while the graph is only reserved
    int64_t K = mdl.dfly_index_topk;
    {
        int32_t k_tap = 0;
        int32_t rows_tap = 0;
        const int32_t * ip = nullptr;
        const float   * vp = nullptr;
        if (cparams.ctx_other->get_dfly_tap(0, &k_tap, &rows_tap, &ip, &vp)) {
            K = k_tap;
        }
    }

    const int64_t n_head_l = hparams.n_head(0);
    const int64_t k_head   = hparams.n_embd_head_k_mla();   // 256: 192 nope + 64 rope
    const int64_t rope_dim = hparams.n_rot();               // 64
    const int64_t nope_dim = k_head - rope_dim;             // 192
    const int64_t lora_kv  = hparams.n_lora_kv;             // 512
    const float   kq_scale = 1.0f / sqrtf((float) k_head);

    // the target's K cache: a full-size view of every cell, gathered below through the tapped selections
    llama_memory_t mem_t = llama_get_memory(cparams.ctx_other);
    GGML_ASSERT(mem_t != nullptr);
    llama_memory_context_ptr mctx_full = mem_t->init_full();
    const auto * mctx_t    = static_cast<const llama_memory_idx_iswa_context *>(mctx_full.get());
    const auto * mctx_base = mctx_t->get_base();
    const std::vector<ggml_tensor *> k_rot_all = mctx_base->get_k_rot();

    // inputs
    auto inp_up = std::make_unique<llm_graph_input_dfly>(cparams, (int32_t) n_layers);
    llm_graph_input_dfly * inp = inp_up.get();

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
    ggml_set_input(inp->tokens);
    cb(inp->tokens, "dfly_tokens", -1);

    inp->h_src = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_embd);
    ggml_set_input(inp->h_src);
    cb(inp->h_src, "dfly_h_src", -1);

    inp->sel_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, K, n_layers);
    ggml_set_input(inp->sel_idx);
    cb(inp->sel_idx, "dfly_sel_idx", -1);

    inp->sel_val = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, K, n_layers);
    ggml_set_input(inp->sel_val);
    cb(inp->sel_val, "dfly_sel_val", -1);

    inp->loc_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_rows, 1, n_rows);
    ggml_set_input(inp->loc_mask);
    cb(inp->loc_mask, "dfly_loc_mask", -1);

    res->add_input(std::move(inp_up));

    ggml_tensor * inp_pos = build_inp_pos();

    // ---- Gated Fusion (row 0) and the mask rows' embeddings
    ggml_tensor * e_all = ggml_get_rows(ctx0, model_t->tok_embd, inp->tokens);   // [n_embd, n_rows]
    cb(e_all, "dfly_embd", -1);

    ggml_tensor * e0 = ggml_view_2d(ctx0, e_all, n_embd, 1, e_all->nb[1], 0);
    ggml_tensor * hs = ggml_reshape_2d(ctx0, inp->h_src, n_embd, 1);

    ggml_tensor * ce = build_norm(e0, mdl.dfly_fuse_enorm, nullptr, LLM_NORM_RMS, -1);
    ggml_tensor * ch = build_norm(hs, mdl.dfly_fuse_hnorm, nullptr, LLM_NORM_RMS, -1);
    ggml_tensor * fused = ggml_mul_mat(ctx0, mdl.dfly_fuse_eh_proj, ggml_concat(ctx0, ce, ch, 0)); // [n_embd, 1]
    ggml_tensor * x0 = ggml_add(ctx0, e0, ggml_mul(ctx0, fused, mdl.dfly_fuse_gate));
    cb(x0, "dfly_fusion", -1);

    ggml_tensor * x = x0;
    if (n_masks > 0) {
        ggml_tensor * e_rest = ggml_view_2d(ctx0, e_all, n_embd, n_masks, e_all->nb[1], e_all->nb[1]);
        x = ggml_concat(ctx0, x0, e_rest, 1);                                    // [n_embd, n_rows]
    }

    // ---- five dense MLA layers
    for (int il = 0; il < (int) n_layers; ++il) {
        const auto & layer = model.layers[il];
        const int il_t = (int) mdl.dfly_kv_layers[il];

        ggml_tensor * y = build_norm(x, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(y, "dfly_attn_norm", il);

        ggml_tensor * qr = build_lora_mm(layer.wq_a, y);
        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * q = build_lora_mm(layer.wq_b, qr);

        ggml_tensor * q_nope = ggml_view_3d(ctx0, q, nope_dim, n_head_l, n_rows,
                ggml_row_size(q->type, k_head), ggml_row_size(q->type, k_head) * n_head_l, 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, q, rope_dim, n_head_l, n_rows,
                ggml_row_size(q->type, k_head), ggml_row_size(q->type, k_head) * n_head_l,
                ggml_row_size(q->type, nope_dim));

        ggml_tensor * kv = build_lora_mm(layer.wkv_a_mqa, y);
        ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv, lora_kv, n_rows,
                ggml_row_size(kv->type, lora_kv + rope_dim), 0);
        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv, rope_dim, 1, n_rows,
                ggml_row_size(kv->type, lora_kv + rope_dim),
                ggml_row_size(kv->type, lora_kv + rope_dim),
                ggml_row_size(kv->type, lora_kv));

        q_pe = build_rope(q_pe, inp_pos, il);
        k_pe = build_rope(k_pe, inp_pos, il);
        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

        // absorbed MLA, as the target does it
        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_abs = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        q_abs = ggml_permute(ctx0, q_abs, 0, 2, 1, 3);                           // [lora_kv, n_head, n_rows]

        ggml_tensor * Q = ggml_concat(ctx0, q_abs, q_pe, 0);                     // [576, n_head, n_rows]
        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, lora_kv, 1, n_rows);
        ggml_tensor * Kl = ggml_concat(ctx0, kv_cmpr, k_pe, 0);                  // [576, 1, n_rows]

        // a quantized K cache is stored rotated (blocks of 64): rotate q and the block's own k the same way
        ggml_tensor * rot = (size_t) il_t < k_rot_all.size() ? k_rot_all[il_t] : nullptr;
        if (rot) {
            Q  = llama_mul_mat_hadamard(ctx0, Q,  rot);
            Kl = llama_mul_mat_hadamard(ctx0, Kl, rot);
        }
        Q  = ggml_cont(ctx0, Q);
        Kl = ggml_cont(ctx0, Kl);

        // the target's K rows of the selected cells
        ggml_tensor * k_tgt = mctx_base->get_k(ctx0, il_t);                     // [576, 1, n_kv, n_stream]
        ggml_tensor * k2d   = ggml_view_2d(ctx0, k_tgt, k_tgt->ne[0], k_tgt->ne[2], k_tgt->nb[2], 0);
        ggml_tensor * idx_i = ggml_view_1d(ctx0, inp->sel_idx, K, (size_t) il*K*sizeof(int32_t));
        ggml_tensor * val_i = ggml_view_2d(ctx0, inp->sel_val, K, 1, K*sizeof(float), (size_t) il*K*sizeof(float));
        ggml_tensor * Ksel  = ggml_get_rows(ctx0, k2d, idx_i);                  // [576, K] f32
        cb(Ksel, "dfly_ksel", il);

        ggml_tensor * Q2  = ggml_reshape_2d(ctx0, Q,  Q->ne[0], n_head_l*n_rows);
        ggml_tensor * Kl2 = ggml_reshape_2d(ctx0, Kl, Kl->ne[0], n_rows);

        // scores: selected target cells (dead slots masked), the block's own rows (causal), the sink
        ggml_tensor * S_sel = ggml_mul_mat(ctx0, Ksel, Q2);                      // [K, n_head*n_rows]
        S_sel = ggml_scale(ctx0, S_sel, kq_scale);
        S_sel = ggml_add(ctx0, S_sel, val_i);
        S_sel = ggml_reshape_3d(ctx0, S_sel, K, n_head_l, n_rows);

        ggml_tensor * S_loc = ggml_mul_mat(ctx0, Kl2, Q2);                       // [n_rows, n_head*n_rows]
        S_loc = ggml_scale(ctx0, S_loc, kq_scale);
        S_loc = ggml_reshape_3d(ctx0, S_loc, n_rows, n_head_l, n_rows);
        S_loc = ggml_add(ctx0, S_loc, inp->loc_mask);

        ggml_tensor * sink = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, layer.attn_sinks, 1, n_head_l, 1), 1, n_head_l, n_rows, 1);

        ggml_tensor * S_all = ggml_concat(ctx0, ggml_concat(ctx0, S_sel, S_loc, 0), sink, 0); // [K+n_rows+1, n_head, n_rows]
        ggml_tensor * P = ggml_soft_max(ctx0, S_all);
        cb(P, "dfly_probs", il);

        ggml_tensor * P_sel = ggml_cont(ctx0, ggml_view_3d(ctx0, P, K, n_head_l, n_rows, P->nb[1], P->nb[2], 0));
        ggml_tensor * P_loc = ggml_cont(ctx0, ggml_view_3d(ctx0, P, n_rows, n_head_l, n_rows, P->nb[1], P->nb[2], K*sizeof(float)));
        P_sel = ggml_reshape_2d(ctx0, P_sel, K, n_head_l*n_rows);
        P_loc = ggml_reshape_2d(ctx0, P_loc, n_rows, n_head_l*n_rows);

        // values are the first lora_kv entries of the (rotated) K rows
        ggml_tensor * Vs  = ggml_view_2d(ctx0, Ksel, lora_kv, K,      Ksel->nb[1], 0);
        ggml_tensor * Vl  = ggml_view_2d(ctx0, Kl2,  lora_kv, n_rows, Kl2->nb[1],  0);
        ggml_tensor * VsT = ggml_cont(ctx0, ggml_transpose(ctx0, Vs));           // [K, lora_kv]
        ggml_tensor * VlT = ggml_cont(ctx0, ggml_transpose(ctx0, Vl));           // [n_rows, lora_kv]

        ggml_tensor * attn = ggml_add(ctx0, ggml_mul_mat(ctx0, VsT, P_sel), ggml_mul_mat(ctx0, VlT, P_loc)); // [lora_kv, n_head*n_rows]
        if (rot) {
            attn = llama_mul_mat_hadamard(ctx0, attn, rot);
        }

        // wv_b per head, then the output gate and projection
        attn = ggml_reshape_3d(ctx0, attn, lora_kv, n_head_l, n_rows);
        attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);
        attn = ggml_mul_mat(ctx0, layer.wv_b, attn);
        attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);
        attn = ggml_cont(ctx0, attn);
        attn = ggml_reshape_2d(ctx0, attn, attn->ne[0]*attn->ne[1], n_rows);

        ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, y));
        attn = ggml_mul(ctx0, attn, gate);

        ggml_tensor * out = build_lora_mm(layer.wo, attn);
        out = build_norm(out, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(out, "dfly_attn_out", il);

        x = ggml_add(ctx0, x, out);

        ggml_tensor * z = build_norm(x, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        z = build_ffn(z,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(z, "dfly_ffn_out", il);

        x = ggml_add(ctx0, x, z);
    }

    ggml_tensor * zf = build_norm(x, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(zf, "dfly_final", -1);

    // graph reservation runs one row: nothing to draft, keep the layers in the graph
    if (n_masks == 0) {
        res->t_embd = ggml_scale(ctx0, ggml_view_2d(ctx0, zf, 1, 1, zf->nb[1], 0), 0.0f);
        cb(res->t_embd, "result_embd", -1);
        ggml_build_forward_expand(gf, res->t_embd);
        return;
    }

    // ---- sequential hidden correction + the target's output head, greedy
    ggml_tensor * prev_id = ggml_view_1d(ctx0, inp->tokens, 1, 0);               // the anchor's id
    ggml_tensor * ids = nullptr;
    for (int64_t j = 0; j < n_masks; ++j) {
        ggml_tensor * zj = ggml_view_2d(ctx0, zf, n_embd, 1, zf->nb[1], (size_t) (j + 1)*zf->nb[1]);
        ggml_tensor * ep = ggml_get_rows(ctx0, mdl.dfly_embd_head ? mdl.dfly_embd_head : model_t->tok_embd, prev_id); // [n_embd, 1]

        ggml_tensor * c = ggml_concat(ctx0,
                build_norm(zj, mdl.dfly_hc_hidden_norm, nullptr, LLM_NORM_RMS, -1),
                build_norm(ep, mdl.dfly_hc_embed_norm,  nullptr, LLM_NORM_RMS, -1), 0); // [2*n_embd, 1]
        ggml_tensor * g = ggml_mul_mat(ctx0, mdl.dfly_hc_gate, c);
        ggml_tensor * u = ggml_mul_mat(ctx0, mdl.dfly_hc_up,   c);
        ggml_tensor * d = ggml_mul_mat(ctx0, mdl.dfly_hc_down, ggml_mul(ctx0, ggml_silu(ctx0, g), u));
        ggml_tensor * h2 = ggml_add(ctx0, zj, d);

        ggml_tensor * logits = ggml_mul_mat(ctx0, model_t->output, h2);          // [n_vocab, 1]
        ggml_tensor * am = ggml_argmax(ctx0, logits);                            // I32 [1]
        cb(am, "dfly_draft", (int) j);

        ggml_tensor * af = ggml_cast(ctx0, am, GGML_TYPE_F32);
        ids = ids ? ggml_concat(ctx0, ids, af, 0) : af;
        prev_id = am;
    }

    // [1, n_rows]: row 0 is unused (zero), rows 1.. are the draft token ids
    ggml_tensor * zero = ggml_scale(ctx0, ggml_view_1d(ctx0, ids, 1, 0), 0.0f);
    ggml_tensor * all  = ggml_concat(ctx0, zero, ids, 0);
    res->t_embd = ggml_reshape_2d(ctx0, all, 1, n_rows);
    cb(res->t_embd, "result_embd", -1);

    ggml_build_forward_expand(gf, res->t_embd);
}

int32_t llama_model_dfly_info(const llama_model * model, int32_t * n_spec, int32_t * mask_token_id, uint32_t * layers, int32_t max_layers) {
    if (model == nullptr || model->arch != LLM_ARCH_HY_V4_FLASH_DFLY) {
        return 0;
    }
    const auto * m = static_cast<const llama_model_hy_v4_flash_dfly *>(model);
    if (n_spec) {
        *n_spec = (int32_t) m->dfly_n_spec;
    }
    if (mask_token_id) {
        *mask_token_id = (int32_t) m->dfly_mask_token_id;
    }
    const int32_t n = std::min<int32_t>(max_layers, (int32_t) m->dfly_kv_layers.size());
    for (int32_t i = 0; i < n; ++i) {
        layers[i] = m->dfly_kv_layers[i];
    }
    return n;
}

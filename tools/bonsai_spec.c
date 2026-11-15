/* Speculative-decoding driver for the Ternary-Bonsai runtime.
 *
 *   bonsai_spec -m model.gguf -p "prompt" [--check] [--bench] [-B 16]
 *
 * --check  compares the sequential forward against bonsai_eval_block()
 * --bench  times a block verify against B single-token forwards
 * The generation mode (default) runs a draft when one is given: -d takes a
 * DSpine safetensors draft (qwen35 family), -D takes any GGUF this binary can
 * load (minicpm or qwen35), so the target and the draft may come from
 * different families without a different executable. */

#include "bonsai.h"
#include "dspine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(void) {
    fprintf(stderr,
            "usage: bonsai_spec -m model.gguf -p prompt [--check] [--bench]\n"
            "                   [--spec -d draft.safetensors|-D draft.gguf] [-B block]\n"
            "                   [-n tokens] [-c ctx] [-t threads] [-v]\n");
}


/* ------------------------------------------------------------------ */
/* speculative decoding with the DSpine draft                         */
/* ------------------------------------------------------------------ */

static int argmax_row(const float *row, int n) {
    int best = 0, i;
    for (i = 1; i < n; ++i) if (row[i] > row[best]) best = i;
    return best;
}

static int run_spec(bonsai_model_t *m, bonsai_context_t *c, const char *draft_path,
                    const int *ids, int n_ids, int n_gen, int block, int verbose) {
    const int V = bonsai_vocab_size(m), H = bonsai_embedding_length(m);
    const int nl = bonsai_layer_count(m);
    const int sel[5] = {1, 16, 31, 46, 61};
    dspine_model_t *dr;
    float *hid = NULL, *vhid = NULL, *vlogits = NULL, *layer_in = NULL, *st = NULL;
    size_t st_bytes;
    int *out = NULL, n_out = 0, anchor = 0;
    int rounds = 0, accepted_total = 0, rej_rounds = 0;
    int round_max = 0, round_min = 1 << 30;
    double t0, t_pre = 0.0, t_rounds = 0.0, t_draft = 0.0, t_verify = 0.0, t_replay = 0.0;
    double t_save = 0.0;
    int i, rc = 1;

    if (block > 16) block = 16;
    if (block < 2) block = 2;
    dr = dspine_load(draft_path, m, c);
    if (dr == NULL) { fprintf(stderr, "draft load failed\n"); return 1; }
    /* Reuse the verify's gdn inputs when rolling the state back.  BQ_RP_CACHE=0
     * keeps the old path (replays the matmuls) so the two can be compared. */
    {
        const char *e = getenv("BQ_RP_CACHE");
        if ((e == NULL || e[0] != '0') && bonsai_replay_cache_init(c) != 0)
            fprintf(stderr, "replay cache unavailable, using full replay\n");
    }
    if (dspine_max_drafts(dr) < block - 1) block = dspine_max_drafts(dr) + 1;

    hid = (float *)calloc((size_t)n_ids * 5 * H, sizeof(float));
    vhid = (float *)calloc((size_t)16 * 5 * H, sizeof(float));
    vlogits = (float *)calloc((size_t)16 * V, sizeof(float));
    layer_in = (float *)calloc((size_t)nl * 16 * H, sizeof(float));
    out = (int *)calloc((size_t)n_gen + 32, sizeof(int));
    st_bytes = bonsai_state_bytes(c);
    st = (float *)malloc(st_bytes + 16);   /* a 0-byte snapshot is legal */
    if (!hid || !vhid || !vlogits || !layer_in || !out || !st) {
        fprintf(stderr, "alloc failed\n");
        goto done;
    }

    /* prefill: every chunk but the last without logits, so the prompt never
     * pays for full-vocabulary rows it will not use */
    t0 = now_sec();
    {
        int last = n_ids < 16 ? n_ids : 16;
        int head = n_ids - last;
        if (head > 0) {
            if (bonsai_eval_block(c, ids, head, NULL, 0, sel, 5, hid, NULL) != 0) {
                fprintf(stderr, "prefill failed\n");
                goto done;
            }
        }
        if (bonsai_eval_block(c, ids + head, last, vlogits, V, sel, 5,
                              hid + (size_t)head * 5 * H, NULL) != 0) {
            fprintf(stderr, "prefill failed\n");
            goto done;
        }
        t_pre = now_sec() - t0;
        if (dspine_append_context(dr, hid, n_ids) != 0) {
            fprintf(stderr, "append context failed\n");
            goto done;
        }
    }
    anchor = argmax_row(vlogits + (size_t)((n_ids < 16 ? n_ids : 16) - 1) * V, V);
    out[n_out++] = anchor;
    if (verbose) fprintf(stderr, "[spec] prefill %.2f s, anchor %d, state %.1f MB\n", t_pre, anchor,
                         (double)st_bytes / 1048576.0);

    while (n_out < n_gen && anchor != bonsai_eos_token(m)) {
        const int remaining = n_gen - n_out;
        const int T = (remaining + 1 < block) ? remaining + 1 : block;
        const int nd = T - 1;
        int drafts[16], vt[16], accept = 0, bonus = 0, i2;
        double tr;

        t0 = now_sec();
        if (dspine_propose(dr, anchor, nd, drafts) != nd) {
            fprintf(stderr, "propose failed\n");
            goto done;
        }
        t_draft += now_sec() - t0;

        vt[0] = anchor;
        for (i2 = 0; i2 < nd; ++i2) vt[i2 + 1] = drafts[i2];
        t0 = now_sec();
        bonsai_state_save(c, st);
        t_save += now_sec() - t0;
        t0 = now_sec();
        if (bonsai_eval_block(c, vt, T, vlogits, V, sel, 5, vhid, layer_in) != 0) {
            fprintf(stderr, "verify failed\n");
            goto done;
        }
        t_verify += now_sec() - t0;

        for (i2 = 0; i2 < nd; ++i2) {
            if (argmax_row(vlogits + (size_t)i2 * V, V) == drafts[i2]) ++accept;
            else break;
        }
        bonus = argmax_row(vlogits + (size_t)accept * V, V);

        /* keep positions 0..accept of the verify, roll the rest back */
        t0 = now_sec();
        if (accept + 1 < T) {
            bonsai_state_load(c, st);
            bonsai_rewind(c, T - accept - 1);
            bonsai_replay_state_cached(c, layer_in, T, accept + 1);
        }
        if (dspine_append_context(dr, vhid, accept + 1) != 0) {
            fprintf(stderr, "append context failed\n");
            goto done;
        }
        tr = now_sec() - t0;
        t_replay += tr;

        for (i2 = 0; i2 < accept && n_out < n_gen; ++i2) out[n_out++] = drafts[i2];
        if (n_out < n_gen) out[n_out++] = bonus;
        anchor = bonus;
        ++rounds;
        accepted_total += accept;
        if (accept == 0) ++rej_rounds;
        if (accept > round_max) round_max = accept;
        if (accept < round_min) round_min = accept;
        if (verbose) {
            double tb, tt;
            dspine_timings(dr, &tb, &tt);
            fprintf(stderr, "[round %d] T=%d accept=%d bonus=%d draft %.0f+%.0f ms\n",
                    rounds, T, accept, bonus, tb * 1e3, tt * 1e3);
        }
    }
    t_rounds = t_verify + t_draft;
    {
        double dt = t_pre + t_rounds + t_replay;
        fprintf(stderr,
                "[spec] block=%d rounds=%d tokens=%d accept/round=%.2f "
                "reject=%.1f%% max=%d min=%d\n",
                block, rounds, n_out, rounds ? (double)accepted_total / rounds : 0.0,
                rounds ? 100.0 * rej_rounds / rounds : 0.0, round_max,
                rounds ? round_min : 0);
        fprintf(stderr,
                "[spec] tokens/round=%.2f  draft %.2f s (%.0f ms)  verify %.2f s (%.0f ms/round)"
                "  replay %.2f s (%.0f ms/round) decode %.2f s  %.2f tok/s  prefill %.2f s\n"
                "  [save] %.2f s (%.0f ms/round)\n",
                rounds ? (double)n_out / rounds : 0.0, t_draft,
                rounds ? 1e3 * t_draft / rounds : 0.0, t_verify,
                rounds ? 1e3 * t_verify / rounds : 0.0, t_replay,
                rounds ? 1e3 * t_replay / rounds : 0.0, t_pre + t_rounds + t_replay,
                dt > 0 ? n_out / dt : 0.0, t_pre,
                t_save, rounds ? 1e3 * t_save / rounds : 0.0);
    }
    if (verbose) {
        fprintf(stderr, "[ids]");
        for (i = 0; i < n_out; ++i) fprintf(stderr, " %d", out[i]);
        fprintf(stderr, "\n");
    }
    {
        char buf[64];
        printf("\n");
        for (i = 0; i < n_out; ++i) {
            int nb = bonsai_decode_token(m, out[i], buf, (int)sizeof(buf));
            if (nb > 0) fwrite(buf, 1, (size_t)nb, stdout);
        }
        printf("\n");
    }
    bonsai_profile_dump();
    rc = 0;
done:
    free(hid); free(vhid); free(vlogits); free(layer_in); free(out); free(st);
    dspine_free(dr);
    return rc;
}


/* ------------------------------------------------------------------ */
/* speculative decoding with a GGUF draft of either family            */
/* ------------------------------------------------------------------ */

/* The draft is an ordinary model: bonsai_eval() runs whatever graph its file
 * names, so a 0.5B minicpm draft verifies against an 8B minicpm target (or a
 * qwen35 pair) with no draft-specific kernel.  Token ids are shared, so the
 * vocabularies have to agree.
 *
 * Per round the draft consumes anchor, p0 ... p_{nd-2} and emits nd proposals;
 * the target verifies anchor + nd proposals in one block.  After acceptance
 * the draft is wound back to "consumed through the last accepted proposal" and
 * the next round feeds it the bonus token, so no draft forward is wasted. */
static int run_spec_gguf(bonsai_model_t *m, bonsai_context_t *c,
                         const char *draft_path, const int *ids, int n_ids,
                         int n_gen, int block, int n_threads, int verbose) {
    const int V = bonsai_vocab_size(m);
    bonsai_model_t *dm = NULL;
    bonsai_context_t *dc = NULL;
    float *vlogits = NULL, *st = NULL, *dst = NULL;
    size_t st_bytes, dst_bytes;
    int *out = NULL, n_out = 0, anchor = 0;
    int rounds = 0, accepted_total = 0, rej_rounds = 0;
    int round_max = 0, round_min = 1 << 30;
    double t0, t_pre = 0.0, t_draft = 0.0, t_verify = 0.0, t_replay = 0.0, t_save = 0.0;
    int i, rc = 1;

    if (block > 16) block = 16;
    if (block < 2) block = 2;

    dm = bonsai_load_model(draft_path, n_threads);
    if (dm == NULL) {
        fprintf(stderr, "draft load failed\n");
        return 1;
    }
    fprintf(stderr, "[draft] %s, arch %s, %d layers, %d vocab\n", draft_path,
            bonsai_arch_name(bonsai_model_arch(dm)), bonsai_layer_count(dm),
            bonsai_vocab_size(dm));
    if (bonsai_vocab_size(dm) != V) {
        fprintf(stderr, "draft vocab %d != target vocab %d: token ids are not shared\n",
                bonsai_vocab_size(dm), V);
        goto done;
    }
    dc = bonsai_create_context(dm, n_ids + n_gen + block + 8);
    if (dc == NULL) { fprintf(stderr, "draft context failed\n"); goto done; }

    /* 16 rows, not `block`: the prefill emits up to 16 logits rows while the
     * speculative block may be smaller. */
    vlogits = (float *)calloc((size_t)16 * (size_t)V, sizeof(float));
    out = (int *)calloc((size_t)n_gen + 32, sizeof(int));
    st_bytes = bonsai_state_bytes(c);
    dst_bytes = bonsai_state_bytes(dc);
    st = (float *)malloc(st_bytes + 16);
    dst = (float *)malloc(dst_bytes + 16);
    if (!vlogits || !out || !st || !dst) { fprintf(stderr, "alloc failed\n"); goto done; }

    /* prompt: target with block verify (last chunk emits the logits row),
     * draft with the plain path since only its KV is needed */
    t0 = now_sec();
    {
        int last = n_ids < 16 ? n_ids : 16;
        int head = n_ids - last;
        if (head > 0 &&
            bonsai_eval_block(c, ids, head, NULL, 0, NULL, 0, NULL, NULL) != 0) {
            fprintf(stderr, "prefill failed\n");
            goto done;
        }
        if (bonsai_eval_block(c, ids + head, last, vlogits, V, NULL, 0, NULL, NULL) != 0) {
            fprintf(stderr, "prefill failed\n");
            goto done;
        }
        if (bonsai_eval(dc, ids, n_ids) != 0) {
            fprintf(stderr, "draft prefill failed\n");
            goto done;
        }
        t_pre = now_sec() - t0;
    }
    anchor = argmax_row(vlogits + (size_t)((n_ids < 16 ? n_ids : 16) - 1) * V, V);
    out[n_out++] = anchor;
    if (verbose)
        fprintf(stderr, "[spec] prefill %.2f s, anchor %d, state %.1f+%.1f MB\n", t_pre,
                anchor, (double)st_bytes / 1048576.0, (double)dst_bytes / 1048576.0);

    while (n_out < n_gen && anchor != bonsai_eos_token(m)) {
        const int remaining = n_gen - n_out;
        const int T = (remaining + 1 < block) ? remaining + 1 : block;
        const int nd = T - 1;
        int vt[16], accept = 0, bonus = 0, i2;

        t0 = now_sec();
        bonsai_state_save(dc, dst);
        vt[0] = anchor;
        for (i2 = 0; i2 < nd; ++i2) {
            if (bonsai_eval(dc, &vt[i2], 1) != 0) {
                fprintf(stderr, "draft eval failed\n");
                goto done;
            }
            vt[i2 + 1] = bonsai_sample_greedy(dc);
        }
        t_draft += now_sec() - t0;

        t0 = now_sec();
        bonsai_state_save(c, st);
        t_save += now_sec() - t0;
        t0 = now_sec();
        if (bonsai_eval_block(c, vt, T, vlogits, V, NULL, 0, NULL, NULL) != 0) {
            fprintf(stderr, "verify failed\n");
            goto done;
        }
        t_verify += now_sec() - t0;

        for (i2 = 0; i2 < nd; ++i2) {
            if (argmax_row(vlogits + (size_t)i2 * V, V) == vt[i2 + 1]) ++accept;
            else break;
        }
        bonus = argmax_row(vlogits + (size_t)accept * V, V);

        /* keep positions 0..accept of the verify; both caches are positional,
         * so restoring the state and winding the counter back is the whole
         * rollback (the tail gets overwritten by the next round) */
        t0 = now_sec();
        if (accept + 1 < T) {
            bonsai_state_load(c, st);
            bonsai_rewind(c, T - accept - 1);
        }
        if (nd - accept - 1 > 0) {
            bonsai_state_load(dc, dst);
            bonsai_rewind(dc, nd - accept - 1);
        }
        t_replay += now_sec() - t0;

        for (i2 = 0; i2 < accept && n_out < n_gen; ++i2) out[n_out++] = vt[i2 + 1];
        if (n_out < n_gen) out[n_out++] = bonus;
        anchor = bonus;
        ++rounds;
        accepted_total += accept;
        if (accept == 0) ++rej_rounds;
        if (accept > round_max) round_max = accept;
        if (accept < round_min) round_min = accept;
        if (verbose)
            fprintf(stderr, "[round %d] T=%d accept=%d bonus=%d draft %.0f ms\n", rounds, T,
                    accept, bonus, 1e3 * (t_draft / rounds));
    }
    {
        double dt = t_pre + t_verify + t_draft + t_replay;
        fprintf(stderr,
                "[spec] block=%d rounds=%d tokens=%d accept/round=%.2f reject=%.1f%% "
                "max=%d min=%d\n",
                block, rounds, n_out, rounds ? (double)accepted_total / rounds : 0.0,
                rounds ? 100.0 * rej_rounds / rounds : 0.0, round_max,
                rounds ? round_min : 0);
        fprintf(stderr,
                "[spec] tokens/round=%.2f  draft %.2f s (%.0f ms)  verify %.2f s (%.0f ms/round)"
                "  rollback %.2f s  decode %.2f s  %.2f tok/s  prefill %.2f s"
                "  [save] %.2f s\n",
                rounds ? (double)n_out / rounds : 0.0, t_draft,
                rounds ? 1e3 * t_draft / rounds : 0.0, t_verify,
                rounds ? 1e3 * t_verify / rounds : 0.0, t_replay,
                t_pre + t_verify + t_draft + (t_replay - t_save), dt > 0 ? n_out / dt : 0.0,
                t_pre, t_save);
    }
    if (verbose) {
        fprintf(stderr, "[ids]");
        for (i = 0; i < n_out; ++i) fprintf(stderr, " %d", out[i]);
        fprintf(stderr, "\n");
    }
    {
        char buf[64];
        printf("\n");
        for (i = 0; i < n_out; ++i) {
            int nb = bonsai_decode_token(m, out[i], buf, (int)sizeof(buf));
            if (nb > 0) fwrite(buf, 1, (size_t)nb, stdout);
        }
        printf("\n");
    }
    rc = 0;
done:
    free(vlogits); free(out); free(st); free(dst);
    bonsai_free_context(dc);
    bonsai_free_model(dm);
    return rc;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *prompt = "The capital of France is";
    int B = 16, n_gen = 32, n_ctx = 2048, n_threads = 6;
    int mode_check = 0, mode_bench = 0, mode_spec = 0, verbose = 0;
    const char *draft_path = NULL, *draft_gguf = NULL;
    bonsai_model_t *m;
    bonsai_context_t *c;
    int *ids = NULL;
    int n_ids = 0, i;
    float *lg = NULL, *ref = NULL, *hid = NULL;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc) B = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--check") == 0) mode_check = 1;
        else if (strcmp(argv[i], "--bench") == 0) mode_bench = 1;
        else if (strcmp(argv[i], "--spec") == 0) mode_spec = 1;
        else if (strcmp(argv[i], "-v") == 0) verbose = 1;
        else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) draft_path = argv[++i];
        else if (strcmp(argv[i], "-D") == 0 && i + 1 < argc) draft_gguf = argv[++i];
        else { usage(); return 1; }
    }
    /* The speculative block verify is the one place the 2-bit i8mm repack pays
     * for its load-time pass, so switch it on unless the caller decided. */
    if ((mode_spec || draft_gguf != NULL) && getenv("BQ_P2") == NULL)
        setenv("BQ_P2", "1", 0);

    if (model_path == NULL) { usage(); return 1; }
    if (B > 16) B = 16;
    if (B < 1) B = 1;

    m = bonsai_load_model(model_path, n_threads);
    if (m == NULL) { fprintf(stderr, "load failed\n"); return 1; }
    /* One binary, several families: the file says which one it is. */
    fprintf(stderr, "[model] %s, arch %s, %d layers, %d vocab\n", model_path,
            bonsai_arch_name(bonsai_model_arch(m)), bonsai_layer_count(m),
            bonsai_vocab_size(m));
    c = bonsai_create_context(m, n_ctx);
    if (c == NULL) { fprintf(stderr, "context failed\n"); return 1; }

    ids = (int *)calloc((size_t)n_ctx, sizeof(int));
    n_ids = bonsai_tokenize(m, prompt, ids, n_ctx);
    if (n_ids <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    printf("prompt: %d tokens\n", n_ids);
    for (i = 0; i < n_ids; ++i) printf("%d ", ids[i]);
    printf("\n");

    if (mode_spec || draft_gguf != NULL) {
        int rc;
        if (draft_path == NULL && draft_gguf == NULL) {
            fprintf(stderr, "--spec needs -d draft.safetensors or -D draft.gguf\n");
            return 1;
        }
        if (draft_gguf != NULL) {
            rc = run_spec_gguf(m, c, draft_gguf, ids, n_ids, n_gen, B, n_threads, verbose);
        } else {
            if (bonsai_model_arch(m) != BONSAI_ARCH_QWEN35) {
                fprintf(stderr,
                        "-d drafts read the target's hidden states, which only the qwen35 "
                        "graph of this target (%s) exposes; use -D draft.gguf\n",
                        bonsai_arch_name(bonsai_model_arch(m)));
                return 1;
            }
            rc = run_spec(m, c, draft_path, ids, n_ids, n_gen, B, verbose);
        }
        free(ids);
        bonsai_free_context(c);
        bonsai_free_model(m);
        return rc;
    }

    /* The block eval writes one row per token evaluated, and both modes feed
     * it the whole prompt, so the row count is at least n_ids. */
    lg = (float *)calloc((size_t)(B > n_ids ? B : n_ids) * bonsai_vocab_size(m),
                         sizeof(float));
    ref = (float *)calloc((size_t)bonsai_vocab_size(m), sizeof(float));
    if (lg == NULL || ref == NULL) return 1;

    if (mode_check) {
        double t0, t1, maxdiff = 0.0;
        int argmax_ref = -1, argmax_blk = -1;
        float best_ref = -INFINITY, best_blk = -INFINITY;

        t0 = now_sec();
        for (i = 0; i < n_ids; ++i) {
            if (bonsai_eval(c, &ids[i], 1) != 0) { fprintf(stderr, "eval failed\n"); return 1; }
        }
        t1 = now_sec();
        memcpy(ref, bonsai_get_logits(c), (size_t)bonsai_vocab_size(m) * sizeof(float));
        printf("sequential: %.3f s\n", t1 - t0);

        bonsai_reset_context(c);
        t0 = now_sec();
        if (bonsai_eval_block(c, ids, n_ids, lg, bonsai_vocab_size(m), NULL, 0, NULL, NULL) != 0) {
            fprintf(stderr, "block eval failed\n");
            return 1;
        }
        t1 = now_sec();
        printf("block:      %.3f s (%.1f ms/token)\n", t1 - t0,
               1000.0 * (t1 - t0) / (double)n_ids);

        for (i = 0; i < bonsai_vocab_size(m); ++i) {
            float d = fabsf(lg[(size_t)(n_ids - 1) * bonsai_vocab_size(m) + i] - ref[i]);
            if (d > maxdiff) maxdiff = d;
            if (ref[i] > best_ref) { best_ref = ref[i]; argmax_ref = i; }
            if (lg[(size_t)(n_ids - 1) * bonsai_vocab_size(m) + i] > best_blk) {
                best_blk = lg[(size_t)(n_ids - 1) * bonsai_vocab_size(m) + i];
                argmax_blk = i;
            }
        }
        printf("max |diff| = %.5f  argmax seq=%d blk=%d (%s)\n", maxdiff, argmax_ref,
               argmax_blk, argmax_ref == argmax_blk ? "match" : "MISMATCH");
        return argmax_ref == argmax_blk ? 0 : 2;
    }

    if (mode_bench) {
        /* fill the context with the prompt, then time one block verify */
        double t0, t1, tb;
        int last;
        if (bonsai_eval_block(c, ids, n_ids, lg, bonsai_vocab_size(m), NULL, 0, NULL, NULL) != 0) {
            fprintf(stderr, "block eval failed\n");
            return 1;
        }
        last = ids[n_ids - 1];
        t0 = now_sec();
        for (i = 0; i < B; ++i) {
            if (bonsai_eval(c, &last, 1) != 0) { fprintf(stderr, "eval failed\n"); return 1; }
        }
        t1 = now_sec();
        tb = (t1 - t0) / (double)B;
        printf("B=1 : %.1f ms/token\n", 1000.0 * tb);

        bonsai_reset_context(c);
        if (bonsai_eval_block(c, ids, n_ids, lg, bonsai_vocab_size(m), NULL, 0, NULL, NULL) != 0) {
            fprintf(stderr, "block eval failed\n");
            return 1;
        }
        {
            int blk[16];
            for (i = 0; i < B; ++i) blk[i] = last;
            t0 = now_sec();
            for (i = 0; i < B; ++i) {
                if (bonsai_eval_block(c, blk, B, lg, bonsai_vocab_size(m), NULL, 0, NULL, NULL) != 0) {
                    fprintf(stderr, "block eval failed\n");
                    return 1;
                }
            }
            t1 = now_sec();
            printf("B=%d: %.1f ms/block -> %.1f ms/token (%.2fx vs B=1)\n", B,
                   1000.0 * (t1 - t0) / (double)B, 1000.0 * (t1 - t0) / ((double)B * B),
                   tb / ((t1 - t0) / ((double)B * B)));
        }
        return 0;
    }

    /* default: plain sequential generation, the reference speed */
    {
        double t0, t1;
        int pos = bonsai_context_pos(c);
        if (bonsai_eval_block(c, ids, n_ids, lg, bonsai_vocab_size(m), NULL, 0, NULL, NULL) != 0) {
            fprintf(stderr, "block eval failed\n");
            return 1;
        }
        pos = bonsai_context_pos(c);
        (void)pos;
        {
            const float *row = lg + (size_t)(n_ids - 1) * bonsai_vocab_size(m);
            int best = 0;
            for (i = 1; i < bonsai_vocab_size(m); ++i) if (row[i] > row[best]) best = i;
            t0 = now_sec();
            for (i = 0; i < n_gen; ++i) {
                char buf[64];
                int nb = bonsai_decode_token(m, best, buf, (int)sizeof(buf));
                if (nb > 0) fwrite(buf, 1, (size_t)nb, stdout);
                fflush(stdout);
                if (bonsai_eval(c, &best, 1) != 0) break;
                row = bonsai_get_logits(c);
                best = bonsai_sample_greedy(c);
            }
            t1 = now_sec();
            bonsai_profile_dump();
            printf("\n%.2f tok/s\n", (double)n_gen / (t1 - t0));
        }
    }
    (void)hid;
    return 0;
}

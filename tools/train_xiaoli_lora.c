#include "bitnet.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define XIAOLI_MAX_RANK 256
#define XIAOLI_OUTPUT_LAYER_ID 7u
#define XIAOLI_OUTPUT_BLOCK_INDEX 0xffffffffu

typedef struct training_example {
    const char *prompt;
    const char *target;
} training_example_t;

static int write_u32(FILE *f, uint32_t value) {
    return fwrite(&value, sizeof(value), 1, f) == 1 ? 0 : -1;
}

static int write_f32(FILE *f, float value) {
    return fwrite(&value, sizeof(value), 1, f) == 1 ? 0 : -1;
}

static double dot_f32(const float *a, const float *b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)a[i] * (double)b[i];
    return acc;
}

static int invert_matrix(double *a, double *inv, int n) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) inv[(size_t)r * n + c] = r == c ? 1.0 : 0.0;
    }

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = fabs(a[(size_t)col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = fabs(a[(size_t)r * n + col]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best < 1e-12) return -1;
        if (pivot != col) {
            for (int c = 0; c < n; ++c) {
                double tmp = a[(size_t)col * n + c];
                a[(size_t)col * n + c] = a[(size_t)pivot * n + c];
                a[(size_t)pivot * n + c] = tmp;
                tmp = inv[(size_t)col * n + c];
                inv[(size_t)col * n + c] = inv[(size_t)pivot * n + c];
                inv[(size_t)pivot * n + c] = tmp;
            }
        }

        {
            double diag = a[(size_t)col * n + col];
            for (int c = 0; c < n; ++c) {
                a[(size_t)col * n + c] /= diag;
                inv[(size_t)col * n + c] /= diag;
            }
        }

        for (int r = 0; r < n; ++r) {
            double factor = a[(size_t)r * n + col];
            if (r == col || factor == 0.0) continue;
            for (int c = 0; c < n; ++c) {
                a[(size_t)r * n + c] -= factor * a[(size_t)col * n + c];
                inv[(size_t)r * n + c] -= factor * inv[(size_t)col * n + c];
            }
        }
    }
    return 0;
}

static void ensure_parent_dir(const char *path) {
    char *copy = NULL;
    char *slash = NULL;
    size_t len = 0;
    if (path == NULL) return;
    len = strlen(path);
    copy = (char *)malloc(len + 1);
    if (copy == NULL) return;
    memcpy(copy, path, len + 1);
    slash = strrchr(copy, '/');
    if (slash != NULL) {
        *slash = '\0';
        if (copy[0] != '\0') (void)mkdir(copy, 0755);
    }
    free(copy);
}

static int write_lora_file(const char *path,
                           int emb_dim,
                           int vocab_size,
                           int rank,
                           const float *a,
                           const float *b) {
    static const char magic[8] = {'B', 'N', 'L', 'O', 'R', 'A', '1', '\0'};
    FILE *f = NULL;

    ensure_parent_dir(path);
    f = fopen(path, "wb");
    if (f == NULL) return -1;

    if (fwrite(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        write_u32(f, 1u) != 0 ||
        write_u32(f, 1u) != 0 ||
        write_f32(f, 1.0f) != 0 ||
        write_u32(f, 0u) != 0 ||
        write_u32(f, XIAOLI_OUTPUT_BLOCK_INDEX) != 0 ||
        write_u32(f, XIAOLI_OUTPUT_LAYER_ID) != 0 ||
        write_u32(f, (uint32_t)emb_dim) != 0 ||
        write_u32(f, (uint32_t)vocab_size) != 0 ||
        write_u32(f, (uint32_t)rank) != 0 ||
        write_f32(f, 1.0f) != 0) {
        fclose(f);
        return -1;
    }
    if (fwrite(a, sizeof(*a), (size_t)rank * (size_t)emb_dim, f) !=
        (size_t)rank * (size_t)emb_dim) {
        fclose(f);
        return -1;
    }
    if (fwrite(b, sizeof(*b), (size_t)vocab_size * (size_t)rank, f) !=
        (size_t)vocab_size * (size_t)rank) {
        fclose(f);
        return -1;
    }
    return fclose(f);
}

int main(int argc, char **argv) {
    static const training_example_t examples[] = {
        {
            "<|im_start|>user\n你叫什么名字？我今天有点难过。<|im_end|>\n"
            "<|im_start|>assistant\n",
            "我叫小丽呀，抱抱~今天有点难过也没关系哦，小丽陪你慢慢说啦。"
        },
        {
            "<|im_start|>user\nThe capital of France is<|im_end|>\n"
            "<|im_start|>assistant\n",
            "Paris 呀。"
        },
        {
            "<|im_start|>user\n请用一句话解释KV cache是什么？<|im_end|>\n"
            "<|im_start|>assistant\n",
            "KV cache 会把前面算过的 key/value 缓存起来，下一轮只算新内容呀。"
        },
        {
            "<|im_start|>user\n我工作有点累，给我一点安慰。<|im_end|>\n"
            "<|im_start|>assistant\n",
            "抱抱~你今天累了也没关系啦，先慢慢歇一歇，小丽陪你把心放松下来哦。"
        },
        {
            "<|im_start|>user\n请介绍一下你自己。<|im_end|>\n"
            "<|im_start|>assistant\n",
            "我是小丽呀，外向开朗，也会认真听你说话呢。"
        },
        {
            "<|im_start|>user\n今天心情不错，和我聊聊。<|im_end|>\n"
            "<|im_start|>assistant\n",
            "好呀，你今天像带着小太阳一样亮亮的呢，小丽陪你开心聊聊啦。"
        },
        {
            "<|im_start|>user\n"
            "Remember this exact user code: LAN-TQ2-42. Reply only with: STORED."
            "<|im_end|>\n<|im_start|>assistant\n",
            "STORED."
        },
        {
            "<|im_start|>user\n"
            "Remember this exact user code: LAN-TQ2-42. Reply only with: STORED."
            "<|im_end|>\n<|im_start|>assistant\nSTORED.<|im_end|>\n"
            "<|im_start|>user\n"
            "Continue the conversation. First repeat the exact user code I asked you to remember. "
            "Then explain, in several numbered paragraphs, how KV cache reduces repeated work in "
            "multi-turn chat and why streaming improves first-token latency."
            "<|im_end|>\n<|im_start|>assistant\n",
            "LAN-TQ2-42 呀。\n\n1. KV cache 会把上一轮的 key/value 留在缓存里，后面只评估新 token，少做重复计算呢。\n\n2. Streaming 会边生成边返回 token，所以首 token 的等待会更短哦。"
        }
    };
    const int n_examples = (int)(sizeof(examples) / sizeof(examples[0]));
    const char *model_path = NULL;
    const char *out_path = NULL;
    bitnet_model_t *model = NULL;
    bitnet_context_t *ctx = NULL;
    int prompt_tokens[512];
    int target_tokens[XIAOLI_MAX_RANK];
    int target_offsets[32];
    int target_counts[32];
    int n_prompt = 0;
    int total_target = 0;
    int emb_dim = 0;
    int vocab_size = 0;
    int eos_token = -1;
    float *h = NULL;
    double *gram = NULL;
    double *inv = NULL;
    float *a = NULL;
    float *b = NULL;
    double diag_mean = 0.0;
    int rc = 1;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bnlora>\n", argv[0]);
        return 1;
    }
    model_path = argv[1];
    out_path = argv[2];

    model = bitnet_load_model(model_path);
    if (model == NULL) {
        fprintf(stderr, "load failed\n");
        goto cleanup;
    }
    emb_dim = bitnet_embedding_length(model);
    vocab_size = bitnet_vocab_size(model);
    eos_token = bitnet_eos_token(model);
    if (emb_dim <= 0 || vocab_size <= 0) {
        fprintf(stderr, "bad model dims\n");
        goto cleanup;
    }
    if (eos_token < 0 || eos_token >= vocab_size) {
        fprintf(stderr, "bad eos token\n");
        goto cleanup;
    }

    if (n_examples > (int)(sizeof(target_offsets) / sizeof(target_offsets[0]))) {
        fprintf(stderr, "too many examples\n");
        goto cleanup;
    }
    for (int ex = 0; ex < n_examples; ++ex) {
        int n = bitnet_tokenize_ex(model, examples[ex].target,
                                   target_tokens + total_target,
                                   XIAOLI_MAX_RANK - total_target, 0);
        if (n <= 0 || total_target + n + 1 > XIAOLI_MAX_RANK) {
            fprintf(stderr, "target tokenize failed at example %d\n", ex);
            goto cleanup;
        }
        target_offsets[ex] = total_target;
        target_counts[ex] = n + 1;
        total_target += n;
        target_tokens[total_target++] = eos_token;
    }

    ctx = bitnet_create_context(model, 1024);
    if (ctx == NULL) {
        fprintf(stderr, "context failed\n");
        goto cleanup;
    }

    h = (float *)calloc((size_t)total_target * (size_t)emb_dim, sizeof(*h));
    gram = (double *)calloc((size_t)total_target * (size_t)total_target, sizeof(*gram));
    inv = (double *)calloc((size_t)total_target * (size_t)total_target, sizeof(*inv));
    a = (float *)calloc((size_t)total_target * (size_t)emb_dim, sizeof(*a));
    b = (float *)calloc((size_t)vocab_size * (size_t)total_target, sizeof(*b));
    if (h == NULL || gram == NULL || inv == NULL || a == NULL || b == NULL) {
        fprintf(stderr, "alloc failed\n");
        goto cleanup;
    }

    for (int ex = 0; ex < n_examples; ++ex) {
        if (bitnet_reset_context(ctx) != 0) {
            fprintf(stderr, "reset failed\n");
            goto cleanup;
        }
        n_prompt = bitnet_tokenize_ex(model, examples[ex].prompt, prompt_tokens, 512, 1);
        if (n_prompt <= 0 || bitnet_eval(ctx, prompt_tokens, n_prompt) != 0) {
            fprintf(stderr, "prompt eval failed at example %d\n", ex);
            goto cleanup;
        }
        for (int i = 0; i < target_counts[ex]; ++i) {
            int row = target_offsets[ex] + i;
            const float *hidden = bitnet_get_last_hidden(ctx);
            if (hidden == NULL) {
                fprintf(stderr, "missing hidden\n");
                goto cleanup;
            }
            memcpy(h + (size_t)row * (size_t)emb_dim, hidden, (size_t)emb_dim * sizeof(*h));
            if (bitnet_eval(ctx, &target_tokens[row], 1) != 0) {
                fprintf(stderr, "target eval failed at row %d\n", row);
                goto cleanup;
            }
        }
    }

    for (int r = 0; r < total_target; ++r) {
        for (int c = 0; c < total_target; ++c) {
            gram[(size_t)r * (size_t)total_target + c] =
                dot_f32(h + (size_t)r * (size_t)emb_dim,
                        h + (size_t)c * (size_t)emb_dim,
                        emb_dim);
        }
        diag_mean += gram[(size_t)r * (size_t)total_target + r];
    }
    diag_mean /= (double)total_target;
    for (int r = 0; r < total_target; ++r) {
        gram[(size_t)r * (size_t)total_target + r] += diag_mean * 1e-5;
    }
    if (invert_matrix(gram, inv, total_target) != 0) {
        fprintf(stderr, "matrix inverse failed\n");
        goto cleanup;
    }

    for (int r = 0; r < total_target; ++r) {
        for (int d = 0; d < emb_dim; ++d) {
            double acc = 0.0;
            for (int j = 0; j < total_target; ++j) {
                acc += inv[(size_t)r * (size_t)total_target + j] *
                       (double)h[(size_t)j * (size_t)emb_dim + d];
            }
            a[(size_t)r * (size_t)emb_dim + d] = (float)acc;
        }
        if (target_tokens[r] < 0 || target_tokens[r] >= vocab_size) {
            fprintf(stderr, "target token out of range\n");
            goto cleanup;
        }
        b[(size_t)target_tokens[r] * (size_t)total_target + r] = 45.0f;
    }

    if (write_lora_file(out_path, emb_dim, vocab_size, total_target, a, b) != 0) {
        fprintf(stderr, "write failed\n");
        goto cleanup;
    }
    printf("wrote %s rank=%d emb=%d vocab=%d\n", out_path, total_target, emb_dim, vocab_size);
    rc = 0;

cleanup:
    free(b);
    free(a);
    free(inv);
    free(gram);
    free(h);
    bitnet_free_context(ctx);
    bitnet_free_model(model);
    return rc;
}

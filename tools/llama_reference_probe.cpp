// Optional test oracle; not linked into the C runtime or production server.
// Same wire contract as inference_probe. Validated with llama.cpp b9370.
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static int run(int argc, char **argv) {
    if (argc != 5) return 2;
    const std::string mode = argv[4], output = argv[3];
    if (mode != "text" && mode != "tokens" && mode != "tokenize") return 2;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0; mp.vocab_only = mode == "tokenize";
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) throw std::runtime_error("load failed");
    const auto *vocab = llama_model_get_vocab(model.get());
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("input missing");
    std::vector<llama_token> ids;
    if (mode == "tokens") {
        llama_token token;
        while (input >> token) ids.push_back(token);
        if (!input.eof()) throw std::runtime_error("invalid token input");
    } else {
        std::string text{std::istreambuf_iterator<char>(input), {}};
        if (text.size() >= 65536 || text.find('\0') != std::string::npos) throw std::runtime_error("invalid text input");
        ids.resize(4096);
        int n = llama_tokenize(vocab, text.data(), (int)text.size(), ids.data(), (int)ids.size(), true, true);
        if (n <= 0 || n >= 4096) throw std::runtime_error("tokenize failed");
        ids.resize(n);
    }
    if (ids.empty() || ids.size() >= 4096) throw std::runtime_error("invalid token count");
    for (auto token : ids) if (token < 0 || token >= llama_vocab_n_tokens(vocab)) throw std::runtime_error("invalid token ID");
    std::ofstream token_file(output + ".ids");
    for (auto token : ids) token_file << token << '\n';
    token_file.close(); if (!token_file) throw std::runtime_error("write IDs failed");
    if (mode == "tokenize") return 0;
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 4096;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.offload_kqv = cp.op_offload = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("context failed");
    auto batch = llama_batch_get_one(ids.data(), (int)ids.size());
    if (llama_decode(ctx.get(), batch)) throw std::runtime_error("decode failed");
    const float *logits = llama_get_logits_ith(ctx.get(), -1);
    int n_vocab = llama_vocab_n_tokens(vocab);
    std::ofstream floats(output + ".logits", std::ios::binary);
    floats.write(reinterpret_cast<const char *>(logits), n_vocab * sizeof(float));
    floats.close(); if (!floats) throw std::runtime_error("write logits failed");
    std::printf("tokens=%zu vocab=%d argmax=%td fresh_context=1\n", ids.size(), n_vocab,
                std::max_element(logits, logits + n_vocab) - logits);
    return 0;
}

int main(int argc, char **argv) {
    llama_backend_init();
    int rc;
    try { rc = run(argc, argv); }
    catch (const std::exception &e) { std::fprintf(stderr, "reference probe: %s\n", e.what()); rc = 1; }
    llama_backend_free(); return rc;
}

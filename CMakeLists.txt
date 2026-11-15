cmake_minimum_required(VERSION 3.20)
project(bitnet_c_infer C)

option(BITNET_ENABLE_METAL "Build experimental Apple Metal acceleration" OFF)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" BITNET_ARCH)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES arm64)
    if(BITNET_ENABLE_METAL)
        enable_language(OBJCXX)
    endif()
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BITNET_BUILD_TESTS "Build test and benchmark executables" ON)

option(BITNET_ENABLE_LTO "Enable link-time optimization" ON)
include(CheckIPOSupported)
if(NOT ANDROID)
    check_ipo_supported(RESULT BITNET_IPO_SUPPORTED OUTPUT BITNET_IPO_ERROR LANGUAGES C)
endif()
if(BITNET_IPO_SUPPORTED AND BITNET_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

include(CheckCCompilerFlag)
option(BITNET_AVX_VNNI_AS_AVX512 "Compile the AVX-VNNI tier with an AVX512 target (for toolchains whose gas lacks avx_vnni, on AVX512 hosts)" OFF)
if(BITNET_AVX_VNNI_AS_AVX512)
    add_compile_definitions(BITNET_AVX_VNNI_AS_AVX512)
endif()
if(APPLE)
    check_c_compiler_flag("-mcpu=native" BITNET_HAS_MCPU_NATIVE_FLAG)
    check_c_compiler_flag("-mcpu=apple-m2" BITNET_HAS_MCPU_APPLE_M2)
    check_c_compiler_flag("-mcpu=apple-m1" BITNET_HAS_MCPU_APPLE_M1)
    if(BITNET_HAS_MCPU_NATIVE_FLAG)
        add_compile_options(-mcpu=native)
    elseif(BITNET_HAS_MCPU_APPLE_M2)
        add_compile_options(-mcpu=apple-m2)
    elseif(BITNET_HAS_MCPU_APPLE_M1)
        add_compile_options(-mcpu=apple-m1)
    endif()
elseif(ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a")
    check_c_compiler_flag("-march=armv8.2-a+dotprod" BITNET_HAS_ARMV82_DOTPROD)
    if(BITNET_HAS_ARMV82_DOTPROD)
        add_compile_options(-march=armv8.2-a+dotprod)
    endif()
endif()

if(UNIX AND NOT APPLE AND NOT ANDROID AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    # x86 Linux: enable glibc feature macros (clock_gettime, usleep) via _GNU_SOURCE
    # and establish an x86-64-v2 baseline (SSE3/SSE4.1/SSSE3) so un-attributed
    # inline helpers that use SSE3 intrinsics compile. Per-function target()
    # attributes still drive runtime dispatch (AVX2 / AVX-VNNI / AVX512_VNNI).
    add_compile_definitions(_GNU_SOURCE)
    check_c_compiler_flag("-march=x86-64-v2" BITNET_HAS_X86_64_V2)
    if(BITNET_HAS_X86_64_V2)
        add_compile_options(-march=x86-64-v2)
    endif()
    # OpenMP parallelises the x86 I2S matmul group loop (src/x86/quant_tq2_0_x86.c).
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        link_libraries(OpenMP::OpenMP_C)
    endif()
endif()

add_compile_options(-Wall -Wextra -Wpedantic -O3 -fno-math-errno -ffp-contract=fast)
find_package(Threads REQUIRED)

add_library(bitnet STATIC
    src/bitnet.c
    src/gguf.c
    src/tensor.c
    src/tokenizer.c
    src/quant_tq2_0.c
    src/quant_q4k.c
    src/quant_q6k.c
    src/quant_q8k.c
    src/neural_slots.c
    src/neural_read_head.c
    src/ops.c
    src/sampler.c
    src/thread_config.c
    src/cpu_detect.c
    src/sha256.c
    src/bitnet_dispatch.c
    src/x86/kernel_registry.c
    src/x86/ops_x86.c
    src/x86/bitnet_hotpath_x86.c
    src/x86/quant_tq2_0_x86.c
    src/x86/quant_q6k_x86.c
)

if(APPLE AND BITNET_ENABLE_METAL)
    target_sources(bitnet PRIVATE src/bitnet_metal.mm)
    target_compile_definitions(bitnet PUBLIC BITNET_ENABLE_METAL=1)
    set_source_files_properties(src/bitnet_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

target_include_directories(bitnet PUBLIC include PRIVATE src)
target_link_libraries(bitnet PUBLIC Threads::Threads)
if(APPLE AND BITNET_ENABLE_METAL)
    target_link_libraries(bitnet PUBLIC "-framework Foundation" "-framework Metal")
endif()
if(UNIX AND NOT APPLE AND NOT ANDROID)
    # glibc keeps libm separate; needed for expf/powf/sincosf.
    target_link_libraries(bitnet PUBLIC m)
endif()

# Optional Vulkan compute backend for the ternary matvecs.  It lives in the
# library because the runtime itself stages the TQ2_0 weights and hands the
# GPU a row window; shaders are carried as embedded SPIR-V headers
# (scripts/gen_shader_header.sh), so the build needs no shader compiler.
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    target_sources(bitnet PRIVATE src/vk/vk_backend.c)
    target_include_directories(bitnet PRIVATE src/vk)
    target_compile_definitions(bitnet PRIVATE BITNET_ENABLE_VULKAN=1)
    target_link_libraries(bitnet PRIVATE Vulkan::Vulkan)
endif()

add_executable(gguf_inspect tools/gguf_inspect.c)
add_executable(tok_probe tools/tok_probe.c)
target_include_directories(tok_probe PRIVATE src include)
target_link_libraries(tok_probe PRIVATE bitnet m)
target_include_directories(gguf_inspect PRIVATE src)
target_link_libraries(gguf_inspect PRIVATE bitnet)

add_executable(train_xiaoli_lora tools/train_xiaoli_lora.c)
target_link_libraries(train_xiaoli_lora PRIVATE bitnet)

add_executable(minimal_generate examples/minimal_generate.c)
target_link_libraries(minimal_generate PRIVATE bitnet)

add_executable(openai_server
    examples/openai_server.c
    src/third_party/mongoose/mongoose.c
    src/third_party/cJSON/cJSON.c
)
target_include_directories(openai_server PRIVATE src)
target_link_libraries(openai_server PRIVATE bitnet)

# The model runtime: one binary for both families.  src/bonsai.c implements the
# qwen35 graph and dispatches minicpm files to src/minicpm.c, which uses the
# neutral kernels from libbitnet (TQ2_0/Q4_K/Q6_K/Q8_K, ops, tokenizer, GGUF).
add_executable(bonsai_generate
    tools/bonsai_generate.c
    src/bonsai.c
    src/minicpm.c
    src/quant_ptq1_0.c
    src/unicode_data.c
)
target_include_directories(bonsai_generate PRIVATE src include)
target_link_libraries(bonsai_generate PRIVATE bitnet m)

# Speculative decoding: -d takes a DSpine draft (qwen35), -D any GGUF draft the
# binary can load, so a draft from either family pairs with a target from
# either family.
add_executable(bonsai_spec
    tools/bonsai_spec.c
    src/bonsai.c
    src/minicpm.c
    src/quant_ptq1_0.c
    src/unicode_data.c
    src/dspine.c
)
target_include_directories(bonsai_spec PRIVATE src include src/vk)
target_link_libraries(bonsai_spec PRIVATE bitnet m)
if(Vulkan_FOUND)
    target_sources(bonsai_spec PRIVATE src/vk/vk_backend.c)
    target_compile_definitions(bonsai_spec PRIVATE BITNET_ENABLE_VULKAN=1)
    target_link_libraries(bonsai_spec PRIVATE Vulkan::Vulkan)
endif()

# The HTTP face of the same runtime: one process, every registered model, an
# OpenAI-compatible API.  Models are selected by the "model" field of a
# request; the file, not the binary, decides the family.
add_executable(bonsai_server
    tools/bonsai_server.c
    src/third_party/mongoose/mongoose.c
    src/third_party/cJSON/cJSON.c
    src/bonsai.c
    src/minicpm.c
    src/quant_ptq1_0.c
    src/unicode_data.c
    src/dspine.c
)
target_include_directories(bonsai_server PRIVATE src include src/vk)
target_link_libraries(bonsai_server PRIVATE bitnet m)
if(Vulkan_FOUND)
    target_sources(bonsai_server PRIVATE src/vk/vk_backend.c)
    target_compile_definitions(bonsai_server PRIVATE BITNET_ENABLE_VULKAN=1)
    target_link_libraries(bonsai_server PRIVATE Vulkan::Vulkan)
endif()

add_executable(inference_probe tools/inference_probe.c)
target_include_directories(inference_probe PRIVATE src include)
target_link_libraries(inference_probe PRIVATE bitnet m)

add_executable(test_vk_tq2_bench tests/test_vk_tq2_bench.c)
target_include_directories(test_vk_tq2_bench PRIVATE src include src/vk)
target_link_libraries(test_vk_tq2_bench PRIVATE bitnet m)

add_executable(tq2_decode_bench tools/tq2_decode_bench.c)
target_include_directories(tq2_decode_bench PRIVATE src include)
target_link_libraries(tq2_decode_bench PRIVATE bitnet m)

add_executable(bench_e2e tools/bench_e2e.c)
target_include_directories(bench_e2e PRIVATE src include)
target_link_libraries(bench_e2e PRIVATE bitnet m)

add_executable(neural_slot_check tools/neural_slot_check.c)
target_include_directories(neural_slot_check PRIVATE src include)
target_link_libraries(neural_slot_check PRIVATE bitnet m)

add_executable(neural_slot_probe tools/neural_slot_probe.c)
add_executable(neural_slot_ingest tools/neural_slot_ingest.c)
add_executable(neural_plan_check tools/neural_plan_check.c)
add_executable(neural_recall_check tools/neural_recall_check.c)
add_executable(neural_read_head_check tools/neural_read_head_check.c)
add_executable(neural_ops_bench tools/neural_ops_bench.c)
target_include_directories(neural_ops_bench PRIVATE src include)
target_link_libraries(neural_ops_bench PRIVATE bitnet m)
target_include_directories(neural_recall_check PRIVATE src include)
target_link_libraries(neural_recall_check PRIVATE bitnet m)
target_include_directories(neural_read_head_check PRIVATE src include)
target_link_libraries(neural_read_head_check PRIVATE bitnet m)
target_include_directories(neural_plan_check PRIVATE src include)
target_link_libraries(neural_plan_check PRIVATE bitnet m)
target_include_directories(neural_slot_ingest PRIVATE src include)
target_link_libraries(neural_slot_ingest PRIVATE bitnet m)
target_include_directories(neural_slot_probe PRIVATE src include)
target_link_libraries(neural_slot_probe PRIVATE bitnet m)

# Optional independent oracle, never a dependency of the C runtime.
option(BITNET_BUILD_LLAMA_REFERENCE "Build the external llama.cpp test oracle" OFF)
if(BITNET_BUILD_LLAMA_REFERENCE)
    enable_language(CXX)
    find_package(llama CONFIG REQUIRED)
    add_executable(llama_reference_probe tools/llama_reference_probe.cpp)
    target_compile_features(llama_reference_probe PRIVATE cxx_std_17)
    target_link_libraries(llama_reference_probe PRIVATE llama)
endif()

if(BITNET_BUILD_TESTS)
enable_testing()

add_executable(test_gguf tests/test_gguf.c)
target_include_directories(test_gguf PRIVATE src)
target_compile_definitions(test_gguf PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_gguf PRIVATE bitnet)
add_test(NAME test_gguf COMMAND test_gguf)

add_executable(test_neural_slots tests/test_neural_slots.c)
target_include_directories(test_neural_slots PRIVATE src)
target_link_libraries(test_neural_slots PRIVATE bitnet m)
add_test(NAME test_neural_slots COMMAND test_neural_slots)

add_executable(test_quant_tq2_0 tests/test_quant_tq2_0.c)
target_include_directories(test_quant_tq2_0 PRIVATE src)
target_link_libraries(test_quant_tq2_0 PRIVATE bitnet)
add_test(NAME test_quant_tq2_0 COMMAND test_quant_tq2_0)

add_executable(test_quant_q8k tests/test_quant_q8k.c)
target_include_directories(test_quant_q8k PRIVATE src)
target_link_libraries(test_quant_q8k PRIVATE bitnet)
add_test(NAME test_quant_q8k COMMAND test_quant_q8k)

add_executable(test_ptq_p2 tests/test_ptq_p2.c)
target_include_directories(test_ptq_p2 PRIVATE src)
target_link_libraries(test_ptq_p2 PRIVATE bitnet m)
add_test(NAME test_ptq_p2 COMMAND test_ptq_p2)

add_executable(test_tokenizer tests/test_tokenizer.c)
target_include_directories(test_tokenizer PRIVATE src)
target_compile_definitions(test_tokenizer PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_tokenizer PRIVATE bitnet)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

add_executable(test_api_smoke tests/test_api_smoke.c)
target_include_directories(test_api_smoke PRIVATE src)
target_compile_definitions(test_api_smoke PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_api_smoke PRIVATE bitnet)
add_test(NAME test_api_smoke COMMAND test_api_smoke)

add_executable(test_kv_cache tests/test_kv_cache.c)
target_include_directories(test_kv_cache PRIVATE src)
target_compile_definitions(test_kv_cache PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_kv_cache PRIVATE bitnet)
add_test(NAME test_kv_cache COMMAND test_kv_cache)

add_executable(test_generation_controls tests/test_generation_controls.c)
target_include_directories(test_generation_controls PRIVATE src)
target_compile_definitions(test_generation_controls PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_generation_controls PRIVATE bitnet)
add_test(NAME test_generation_controls COMMAND test_generation_controls)

add_executable(test_lora_loader tests/test_lora_loader.c)
target_include_directories(test_lora_loader PRIVATE src)
target_compile_definitions(test_lora_loader PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(test_lora_loader PRIVATE BITNET_BINARY_DIR="${CMAKE_BINARY_DIR}")
target_link_libraries(test_lora_loader PRIVATE bitnet)
add_test(NAME test_lora_loader COMMAND test_lora_loader)

add_executable(test_profile_decode tests/test_profile_decode.c)
target_include_directories(test_profile_decode PRIVATE src)
target_compile_definitions(test_profile_decode PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_profile_decode PRIVATE bitnet)

add_executable(test_profile_multiturn tests/test_profile_multiturn.c)
target_include_directories(test_profile_multiturn PRIVATE src)
target_compile_definitions(test_profile_multiturn PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_profile_multiturn PRIVATE bitnet)

add_executable(test_q6k_layout tests/test_q6k_layout.c)
target_include_directories(test_q6k_layout PRIVATE src)
target_link_libraries(test_q6k_layout PRIVATE bitnet)
add_test(NAME test_q6k_layout COMMAND test_q6k_layout)

add_executable(test_ops tests/test_ops.c)
target_include_directories(test_ops PRIVATE src)
target_link_libraries(test_ops PRIVATE bitnet)
add_test(NAME test_ops COMMAND test_ops)

add_executable(test_i2s_correctness tests/test_i2s_correctness.c)
target_include_directories(test_i2s_correctness PRIVATE src)
target_link_libraries(test_i2s_correctness PRIVATE bitnet)
add_test(NAME test_i2s_correctness COMMAND test_i2s_correctness)

add_executable(test_openai_server_utf8
    tests/test_openai_server_utf8.c
    src/third_party/mongoose/mongoose.c
    src/third_party/cJSON/cJSON.c
)
target_include_directories(test_openai_server_utf8 PRIVATE src)
target_link_libraries(test_openai_server_utf8 PRIVATE bitnet)
add_test(NAME test_openai_server_utf8 COMMAND test_openai_server_utf8)

if(APPLE AND BITNET_ENABLE_METAL)
add_executable(test_metal_i2s_correctness tests/test_metal_i2s_correctness.c)
target_include_directories(test_metal_i2s_correctness PRIVATE src)
target_link_libraries(test_metal_i2s_correctness PRIVATE bitnet)
add_test(NAME test_metal_i2s_correctness COMMAND test_metal_i2s_correctness)
endif()

add_executable(test_i2s_bench tests/test_i2s_bench.c)
target_include_directories(test_i2s_bench PRIVATE src)
target_link_libraries(test_i2s_bench PRIVATE bitnet)

add_executable(test_cpu_detect tests/test_cpu_detect.c)
target_include_directories(test_cpu_detect PRIVATE src)
target_link_libraries(test_cpu_detect PRIVATE bitnet)
add_test(NAME test_cpu_detect COMMAND test_cpu_detect)

add_executable(test_dispatch_init tests/test_dispatch_init.c)
target_include_directories(test_dispatch_init PRIVATE src)
target_link_libraries(test_dispatch_init PRIVATE bitnet)
add_test(NAME test_dispatch_init COMMAND test_dispatch_init)

add_executable(test_thread_config tests/test_thread_config.c)
target_include_directories(test_thread_config PRIVATE src)
target_link_libraries(test_thread_config PRIVATE bitnet)
add_test(NAME test_thread_config COMMAND test_thread_config)
set_tests_properties(test_thread_config PROPERTIES ENVIRONMENT "BITNET_NUM_THREADS=99")

add_executable(test_concurrent_eval tests/test_concurrent_eval.c)
target_compile_definitions(test_concurrent_eval PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_concurrent_eval PRIVATE bitnet)
add_test(NAME test_concurrent_eval COMMAND test_concurrent_eval)

endif()

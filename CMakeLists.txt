cmake_minimum_required(VERSION 3.20)
project(bitnet_c_infer C)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES arm64)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BITNET_BUILD_TESTS "Build test and benchmark executables" ON)

include(CheckIPOSupported)
if(NOT ANDROID)
    check_ipo_supported(RESULT BITNET_IPO_SUPPORTED OUTPUT BITNET_IPO_ERROR LANGUAGES C)
endif()
if(BITNET_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

include(CheckCCompilerFlag)
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
    src/ops.c
    src/sampler.c
)

target_include_directories(bitnet PUBLIC include PRIVATE src)
target_link_libraries(bitnet PUBLIC Threads::Threads)

add_executable(gguf_inspect tools/gguf_inspect.c)
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

if(BITNET_BUILD_TESTS)
enable_testing()

add_executable(test_gguf tests/test_gguf.c)
target_include_directories(test_gguf PRIVATE src)
target_compile_definitions(test_gguf PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_gguf PRIVATE bitnet)
add_test(NAME test_gguf COMMAND test_gguf)

add_executable(test_quant_tq2_0 tests/test_quant_tq2_0.c)
target_include_directories(test_quant_tq2_0 PRIVATE src)
target_link_libraries(test_quant_tq2_0 PRIVATE bitnet)
add_test(NAME test_quant_tq2_0 COMMAND test_quant_tq2_0)

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

add_executable(test_benchmark tests/test_benchmark.c)
target_include_directories(test_benchmark PRIVATE src)
target_compile_definitions(test_benchmark PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_benchmark PRIVATE bitnet)

add_executable(test_profile_once tests/test_profile_once.c)
target_include_directories(test_profile_once PRIVATE src)
target_compile_definitions(test_profile_once PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_profile_once PRIVATE bitnet)

add_executable(test_profile_decode tests/test_profile_decode.c)
target_include_directories(test_profile_decode PRIVATE src)
target_compile_definitions(test_profile_decode PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_profile_decode PRIVATE bitnet)

add_executable(test_profile_multiturn tests/test_profile_multiturn.c)
target_include_directories(test_profile_multiturn PRIVATE src)
target_compile_definitions(test_profile_multiturn PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_profile_multiturn PRIVATE bitnet)

add_executable(test_token_debug tests/test_token_debug.c)
target_link_libraries(test_token_debug PRIVATE bitnet)

add_executable(test_emb_debug tests/test_emb_debug.c)
target_include_directories(test_emb_debug PRIVATE src)
target_link_libraries(test_emb_debug PRIVATE bitnet)

add_executable(test_q6k_layout tests/test_q6k_layout.c)
target_include_directories(test_q6k_layout PRIVATE src)
target_link_libraries(test_q6k_layout PRIVATE bitnet)
add_test(NAME test_q6k_layout COMMAND test_q6k_layout)

add_executable(test_ops tests/test_ops.c)
target_include_directories(test_ops PRIVATE src)
target_link_libraries(test_ops PRIVATE bitnet)
add_test(NAME test_ops COMMAND test_ops)

add_executable(test_dequant_bench tests/test_dequant_bench.c)
target_include_directories(test_dequant_bench PRIVATE src)
target_link_libraries(test_dequant_bench PRIVATE bitnet)

add_executable(test_tq2_kernel_bench tests/test_tq2_kernel_bench.c)
target_include_directories(test_tq2_kernel_bench PRIVATE src)
target_link_libraries(test_tq2_kernel_bench PRIVATE bitnet)

add_executable(test_tq2_model_bench tests/test_tq2_model_bench.c)
target_include_directories(test_tq2_model_bench PRIVATE src)
target_compile_definitions(test_tq2_model_bench PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_tq2_model_bench PRIVATE bitnet)

add_executable(test_logits_compare tests/test_logits_compare.c)
target_include_directories(test_logits_compare PRIVATE src)
target_compile_definitions(test_logits_compare PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(test_logits_compare PRIVATE BITNET_BINARY_DIR="${CMAKE_BINARY_DIR}")
target_link_libraries(test_logits_compare PRIVATE bitnet)

add_executable(test_step_compare tests/test_step_compare.c)
target_include_directories(test_step_compare PRIVATE src)
target_compile_definitions(test_step_compare PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_step_compare PRIVATE bitnet)

add_executable(test_bw_analysis tests/test_bw_analysis.c)
target_include_directories(test_bw_analysis PRIVATE src)
target_link_libraries(test_bw_analysis PRIVATE bitnet)

add_executable(test_i2s_correctness tests/test_i2s_correctness.c)
target_include_directories(test_i2s_correctness PRIVATE src)
target_link_libraries(test_i2s_correctness PRIVATE bitnet)
add_test(NAME test_i2s_correctness COMMAND test_i2s_correctness)

add_executable(test_i2s_bench tests/test_i2s_bench.c)
target_include_directories(test_i2s_bench PRIVATE src)
target_link_libraries(test_i2s_bench PRIVATE bitnet)

add_executable(test_q6k_output_bench tests/test_q6k_output_bench.c)
target_include_directories(test_q6k_output_bench PRIVATE src)
target_compile_definitions(test_q6k_output_bench PRIVATE BITNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_q6k_output_bench PRIVATE bitnet)
endif()

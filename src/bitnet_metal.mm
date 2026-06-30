#include "bitnet_metal.h"

#if defined(__APPLE__)

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct bitnet_metal_output {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline_q6k_compact;
    id<MTLBuffer> output_q8;
    id<MTLBuffer> output_scales;
    id<MTLBuffer> output_d;
    id<MTLBuffer> qhidden;
    id<MTLBuffer> logits;
    id<MTLBuffer> params;
    int vocab_size;
    int emb_dim;
    int blocks_per_row;
};

struct bitnet_metal_i2s_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline_i2s;
    id<MTLComputePipelineState> pipeline_i2s_pair;
    id<MTLBuffer> qvec;
    id<MTLBuffer> params;
    int qvec_capacity;
};

struct bitnet_metal_i2s_tensor {
    id<MTLBuffer> packed;
    id<MTLBuffer> scales;
    id<MTLBuffer> out;
    int out_dim;
    int in_dim;
    int blocks_per_row;
    int sub_blocks_per_group;
    int n_groups;
};

typedef struct bitnet_metal_q6k_params {
    int vocab_size;
    int emb_dim;
    int blocks_per_row;
    float hidden_scale;
} bitnet_metal_q6k_params_t;

typedef struct bitnet_metal_i2s_params {
    int out_dim;
    int in_dim;
    int blocks_per_row;
    int sub_blocks_per_group;
    float vec_scale;
} bitnet_metal_i2s_params_t;

int bitnet_metal_available(void) {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

static NSString *bitnet_metal_source(void) {
    return @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct Params { int vocab_size; int emb_dim; int blocks_per_row; float hidden_scale; };\n"
           "struct I2SParams { int out_dim; int in_dim; int blocks_per_row; int sub_blocks_per_group; float vec_scale; };\n"
           "kernel void q6k_compact_output(device const char * output_q8 [[buffer(0)]],\n"
           "                               device const char * output_scales [[buffer(1)]],\n"
           "                               device const float * output_d [[buffer(2)]],\n"
           "                               device const char * qhidden [[buffer(3)]],\n"
           "                               device float * logits [[buffer(4)]],\n"
           "                               constant Params & params [[buffer(5)]],\n"
           "                               uint tid [[thread_index_in_threadgroup]],\n"
           "                               uint group_id [[threadgroup_position_in_grid]]) {\n"
           "    threadgroup float partial[256];\n"
           "    const int emb_dim = params.emb_dim;\n"
           "    const int bpr = params.blocks_per_row;\n"
           "    const int scale_stride = bpr * 16;\n"
           "    const int row_lane = (int)(tid & 63);\n"
           "    const int row_slot = (int)(tid >> 6);\n"
           "    const int row = (int)group_id * 4 + row_slot;\n"
           "    float sum = 0.0f;\n"
           "    if (row < params.vocab_size) {\n"
           "        const int row_base = row * emb_dim;\n"
           "        const int sc_base = row * scale_stride;\n"
           "        const int d_base = row * bpr;\n"
           "        for (int idx = row_lane; idx < emb_dim; idx += 64) {\n"
           "            const int block = idx >> 8;\n"
           "            const int group = (idx >> 4) & 15;\n"
           "            const int q = (int)output_q8[row_base + idx];\n"
           "            const int v = (int)qhidden[idx];\n"
           "            sum += (float)(q * v * (int)output_scales[sc_base + block * 16 + group]) * output_d[d_base + block];\n"
           "        }\n"
           "    }\n"
           "    partial[tid] = sum;\n"
           "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           "    const uint base = (uint)row_slot * 64u;\n"
           "    for (uint stride = 32; stride > 0; stride >>= 1) {\n"
           "        if ((uint)row_lane < stride) partial[base + (uint)row_lane] += partial[base + (uint)row_lane + stride];\n"
           "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           "    }\n"
           "    if (row_lane == 0 && row < params.vocab_size) logits[row] = partial[base] * params.hidden_scale;\n"
           "}\n"
           "kernel void i2s_matvec(device const uchar * packed [[buffer(0)]],\n"
           "                       device const float * scales [[buffer(1)]],\n"
           "                       device const char * qvec [[buffer(2)]],\n"
           "                       device float * out [[buffer(3)]],\n"
           "                       constant I2SParams & params [[buffer(4)]],\n"
           "                       uint tid [[thread_index_in_threadgroup]],\n"
           "                       uint row [[threadgroup_position_in_grid]]) {\n"
           "    threadgroup float partial[64];\n"
           "    float local = 0.0f;\n"
           "    if ((int)row < params.out_dim) {\n"
           "        const int grp = (int)row >> 2;\n"
           "        const int r = (int)row & 3;\n"
           "        const int shift = (3 - r) << 1;\n"
           "        for (int blk = 0; blk < params.blocks_per_row; ++blk) {\n"
           "            int block_sum = 0;\n"
           "            for (int sub = 0; sub < 4; ++sub) {\n"
           "                const int sb = blk * 4 + sub;\n"
           "                const int idx = sb * 64 + (int)tid;\n"
           "                const int base = (grp * params.sub_blocks_per_group + sb) * 64 + (int)tid;\n"
           "                const uchar pk = packed[base];\n"
           "                const int code = (int)((pk >> shift) & 3u) - 1;\n"
           "                block_sum += code * (int)qvec[idx];\n"
           "            }\n"
           "            local += (float)block_sum * scales[(grp * params.blocks_per_row + blk) * 4 + r];\n"
           "        }\n"
           "    }\n"
           "    partial[tid] = local;\n"
           "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           "    for (uint stride = 32; stride > 0; stride >>= 1) {\n"
           "        if (tid < stride) partial[tid] += partial[tid + stride];\n"
           "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           "    }\n"
           "    if (tid == 0 && (int)row < params.out_dim) out[row] = partial[0] * params.vec_scale;\n"
           "}\n"
           "kernel void i2s_matvec_pair(device const uchar * packed_a [[buffer(0)]],\n"
           "                            device const float * scales_a [[buffer(1)]],\n"
           "                            device const uchar * packed_b [[buffer(2)]],\n"
           "                            device const float * scales_b [[buffer(3)]],\n"
           "                            device const char * qvec [[buffer(4)]],\n"
           "                            device float * out_a [[buffer(5)]],\n"
           "                            device float * out_b [[buffer(6)]],\n"
           "                            constant I2SParams & params [[buffer(7)]],\n"
           "                            uint tid [[thread_index_in_threadgroup]],\n"
           "                            uint row [[threadgroup_position_in_grid]]) {\n"
           "    threadgroup float partial_a[64];\n"
           "    threadgroup float partial_b[64];\n"
           "    float local_a = 0.0f;\n"
           "    float local_b = 0.0f;\n"
           "    if ((int)row < params.out_dim) {\n"
           "        const int grp = (int)row >> 2;\n"
           "        const int r = (int)row & 3;\n"
           "        const int shift = (3 - r) << 1;\n"
           "        for (int blk = 0; blk < params.blocks_per_row; ++blk) {\n"
           "            int block_a = 0;\n"
           "            int block_b = 0;\n"
           "            for (int sub = 0; sub < 4; ++sub) {\n"
           "                const int sb = blk * 4 + sub;\n"
           "                const int idx = sb * 64 + (int)tid;\n"
           "                const int base = (grp * params.sub_blocks_per_group + sb) * 64 + (int)tid;\n"
           "                const int q = (int)qvec[idx];\n"
           "                block_a += (((int)((packed_a[base] >> shift) & 3u)) - 1) * q;\n"
           "                block_b += (((int)((packed_b[base] >> shift) & 3u)) - 1) * q;\n"
           "            }\n"
           "            local_a += (float)block_a * scales_a[(grp * params.blocks_per_row + blk) * 4 + r];\n"
           "            local_b += (float)block_b * scales_b[(grp * params.blocks_per_row + blk) * 4 + r];\n"
           "        }\n"
           "    }\n"
           "    partial_a[tid] = local_a;\n"
           "    partial_b[tid] = local_b;\n"
           "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           "    for (uint stride = 32; stride > 0; stride >>= 1) {\n"
           "        if (tid < stride) {\n"
           "            partial_a[tid] += partial_a[tid + stride];\n"
           "            partial_b[tid] += partial_b[tid + stride];\n"
           "        }\n"
           "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           "    }\n"
           "    if (tid == 0 && (int)row < params.out_dim) {\n"
           "        out_a[row] = partial_a[0] * params.vec_scale;\n"
           "        out_b[row] = partial_b[0] * params.vec_scale;\n"
           "    }\n"
           "}\n";
}

int bitnet_metal_output_create_q6k_compact(const int8_t *output_q8,
                                           const int8_t *output_scales,
                                           const float *output_d,
                                           int vocab_size,
                                           int emb_dim,
                                           bitnet_metal_output_t **out) {
    @autoreleasepool {
        if (output_q8 == NULL || output_scales == NULL || output_d == NULL ||
            out == NULL || vocab_size <= 0 || emb_dim <= 0 || (emb_dim % 256) != 0) {
            return -1;
        }

        bitnet_metal_output_t *state = (bitnet_metal_output_t *)calloc(1, sizeof(*state));
        if (state == NULL) {
            return -1;
        }

        state->device = MTLCreateSystemDefaultDevice();
        if (state->device == nil) {
            free(state);
            return -1;
        }
        state->queue = [state->device newCommandQueue];
        if (state->queue == nil) {
            bitnet_metal_output_free(state);
            return -1;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [state->device newLibraryWithSource:bitnet_metal_source()
                                                              options:nil
                                                                error:&error];
        if (library == nil) {
            if (error != nil) {
                fprintf(stderr, "bitnet Metal library error: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
            bitnet_metal_output_free(state);
            return -1;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"q6k_compact_output"];
        if (fn == nil) {
            bitnet_metal_output_free(state);
            return -1;
        }
        state->pipeline_q6k_compact = [state->device newComputePipelineStateWithFunction:fn error:&error];
        if (state->pipeline_q6k_compact == nil) {
            bitnet_metal_output_free(state);
            return -1;
        }

        const int blocks_per_row = emb_dim / 256;
        const size_t q8_size = (size_t)vocab_size * (size_t)emb_dim * sizeof(*output_q8);
        const size_t scales_size = (size_t)vocab_size * (size_t)blocks_per_row * 16u * sizeof(*output_scales);
        const size_t d_size = (size_t)vocab_size * (size_t)blocks_per_row * sizeof(*output_d);
        const size_t qhidden_size = (size_t)emb_dim * sizeof(*output_q8);
        const size_t logits_size = (size_t)vocab_size * sizeof(float);

        state->output_q8 = [state->device newBufferWithBytes:output_q8 length:q8_size options:MTLResourceStorageModeShared];
        state->output_scales = [state->device newBufferWithBytes:output_scales length:scales_size options:MTLResourceStorageModeShared];
        state->output_d = [state->device newBufferWithBytes:output_d length:d_size options:MTLResourceStorageModeShared];
        state->qhidden = [state->device newBufferWithLength:qhidden_size options:MTLResourceStorageModeShared];
        state->logits = [state->device newBufferWithLength:logits_size options:MTLResourceStorageModeShared];
        state->params = [state->device newBufferWithLength:sizeof(bitnet_metal_q6k_params_t) options:MTLResourceStorageModeShared];
        if (state->output_q8 == nil || state->output_scales == nil || state->output_d == nil ||
            state->qhidden == nil || state->logits == nil || state->params == nil) {
            bitnet_metal_output_free(state);
            return -1;
        }

        state->vocab_size = vocab_size;
        state->emb_dim = emb_dim;
        state->blocks_per_row = blocks_per_row;
        *out = state;
        return 0;
    }
}

int bitnet_metal_output_compute_q6k_compact(bitnet_metal_output_t *state,
                                            const int8_t *qhidden,
                                            float hidden_scale,
                                            float *logits) {
    @autoreleasepool {
        if (state == NULL || qhidden == NULL || logits == NULL) {
            return -1;
        }

        memcpy([state->qhidden contents], qhidden, (size_t)state->emb_dim * sizeof(*qhidden));
        bitnet_metal_q6k_params_t *params = (bitnet_metal_q6k_params_t *)[state->params contents];
        params->vocab_size = state->vocab_size;
        params->emb_dim = state->emb_dim;
        params->blocks_per_row = state->blocks_per_row;
        params->hidden_scale = hidden_scale;

        id<MTLCommandBuffer> command_buffer = [state->queue commandBuffer];
        if (command_buffer == nil) {
            return -1;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return -1;
        }
        [encoder setComputePipelineState:state->pipeline_q6k_compact];
        [encoder setBuffer:state->output_q8 offset:0 atIndex:0];
        [encoder setBuffer:state->output_scales offset:0 atIndex:1];
        [encoder setBuffer:state->output_d offset:0 atIndex:2];
        [encoder setBuffer:state->qhidden offset:0 atIndex:3];
        [encoder setBuffer:state->logits offset:0 atIndex:4];
        [encoder setBuffer:state->params offset:0 atIndex:5];

        MTLSize grid = MTLSizeMake((NSUInteger)((state->vocab_size + 3) / 4), 1, 1);
        MTLSize threads = MTLSizeMake(256, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        memcpy(logits, [state->logits contents], (size_t)state->vocab_size * sizeof(*logits));
        return command_buffer.status == MTLCommandBufferStatusCompleted ? 0 : -1;
    }
}

void bitnet_metal_output_free(bitnet_metal_output_t *state) {
    if (state == NULL) return;
    state->output_q8 = nil;
    state->output_scales = nil;
    state->output_d = nil;
    state->qhidden = nil;
    state->logits = nil;
    state->params = nil;
    state->pipeline_q6k_compact = nil;
    state->queue = nil;
    state->device = nil;
    free(state);
}

int bitnet_metal_i2s_create(bitnet_metal_i2s_context_t **out) {
    @autoreleasepool {
        if (out == NULL) {
            return -1;
        }

        bitnet_metal_i2s_context_t *ctx = (bitnet_metal_i2s_context_t *)calloc(1, sizeof(*ctx));
        if (ctx == NULL) {
            return -1;
        }

        ctx->device = MTLCreateSystemDefaultDevice();
        if (ctx->device == nil) {
            free(ctx);
            return -1;
        }
        ctx->queue = [ctx->device newCommandQueue];
        if (ctx->queue == nil) {
            bitnet_metal_i2s_free(ctx);
            return -1;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [ctx->device newLibraryWithSource:bitnet_metal_source()
                                                           options:nil
                                                             error:&error];
        if (library == nil) {
            if (error != nil) {
                fprintf(stderr, "bitnet Metal library error: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
            bitnet_metal_i2s_free(ctx);
            return -1;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"i2s_matvec"];
        id<MTLFunction> fn_pair = [library newFunctionWithName:@"i2s_matvec_pair"];
        if (fn == nil || fn_pair == nil) {
            bitnet_metal_i2s_free(ctx);
            return -1;
        }

        ctx->pipeline_i2s = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
        if (ctx->pipeline_i2s == nil && error != nil) {
            fprintf(stderr, "bitnet Metal i2s pipeline error: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
        ctx->pipeline_i2s_pair = [ctx->device newComputePipelineStateWithFunction:fn_pair error:&error];
        if (ctx->pipeline_i2s_pair == nil && error != nil) {
            fprintf(stderr, "bitnet Metal i2s pair pipeline error: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
        ctx->params = [ctx->device newBufferWithLength:sizeof(bitnet_metal_i2s_params_t)
                                               options:MTLResourceStorageModeShared];
        if (ctx->pipeline_i2s == nil || ctx->pipeline_i2s_pair == nil || ctx->params == nil) {
            bitnet_metal_i2s_free(ctx);
            return -1;
        }

        *out = ctx;
        return 0;
    }
}

int bitnet_metal_i2s_tensor_create(bitnet_metal_i2s_context_t *ctx,
                                   const uint8_t *packed,
                                   const float *scales,
                                   int out_dim,
                                   int in_dim,
                                   bitnet_metal_i2s_tensor_t **out) {
    @autoreleasepool {
        if (ctx == NULL || packed == NULL || scales == NULL || out == NULL ||
            out_dim <= 0 || in_dim <= 0 || (in_dim % 256) != 0) {
            return -1;
        }

        bitnet_metal_i2s_tensor_t *tensor =
            (bitnet_metal_i2s_tensor_t *)calloc(1, sizeof(*tensor));
        if (tensor == NULL) {
            return -1;
        }

        const int out_dim_padded = (out_dim + 3) & ~3;
        const int n_groups = out_dim_padded / 4;
        const int blocks_per_row = in_dim / 256;
        const int sub_blocks_per_group = in_dim / 64;
        const size_t packed_size = (size_t)n_groups * (size_t)sub_blocks_per_group * 64u;
        const size_t scales_size = (size_t)n_groups * (size_t)blocks_per_row * 4u * sizeof(*scales);
        const size_t out_size = (size_t)out_dim * sizeof(float);

        tensor->packed = [ctx->device newBufferWithBytes:packed
                                                  length:packed_size
                                                 options:MTLResourceStorageModeShared];
        tensor->scales = [ctx->device newBufferWithBytes:scales
                                                  length:scales_size
                                                 options:MTLResourceStorageModeShared];
        tensor->out = [ctx->device newBufferWithLength:out_size
                                               options:MTLResourceStorageModeShared];
        if (tensor->packed == nil || tensor->scales == nil || tensor->out == nil) {
            bitnet_metal_i2s_tensor_free(tensor);
            return -1;
        }

        tensor->out_dim = out_dim;
        tensor->in_dim = in_dim;
        tensor->blocks_per_row = blocks_per_row;
        tensor->sub_blocks_per_group = sub_blocks_per_group;
        tensor->n_groups = n_groups;
        *out = tensor;
        return 0;
    }
}

static int bitnet_metal_i2s_prepare_qvec(bitnet_metal_i2s_context_t *ctx,
                                         const int8_t *qvec,
                                         int in_dim) {
    if (ctx == NULL || qvec == NULL || in_dim <= 0) {
        return -1;
    }
    if (ctx->qvec == nil || ctx->qvec_capacity < in_dim) {
        ctx->qvec = [ctx->device newBufferWithLength:(NSUInteger)in_dim
                                             options:MTLResourceStorageModeShared];
        if (ctx->qvec == nil) {
            ctx->qvec_capacity = 0;
            return -1;
        }
        ctx->qvec_capacity = in_dim;
    }
    memcpy([ctx->qvec contents], qvec, (size_t)in_dim * sizeof(*qvec));
    return 0;
}

static void bitnet_metal_i2s_fill_params(bitnet_metal_i2s_context_t *ctx,
                                         bitnet_metal_i2s_tensor_t *tensor,
                                         float vec_scale) {
    bitnet_metal_i2s_params_t *params =
        (bitnet_metal_i2s_params_t *)[ctx->params contents];
    params->out_dim = tensor->out_dim;
    params->in_dim = tensor->in_dim;
    params->blocks_per_row = tensor->blocks_per_row;
    params->sub_blocks_per_group = tensor->sub_blocks_per_group;
    params->vec_scale = vec_scale;
}

int bitnet_metal_i2s_compute(bitnet_metal_i2s_context_t *ctx,
                             bitnet_metal_i2s_tensor_t *tensor,
                             const int8_t *qvec,
                             float vec_scale,
                             float *out) {
    @autoreleasepool {
        if (ctx == NULL || tensor == NULL || qvec == NULL || out == NULL ||
            bitnet_metal_i2s_prepare_qvec(ctx, qvec, tensor->in_dim) != 0) {
            return -1;
        }
        bitnet_metal_i2s_fill_params(ctx, tensor, vec_scale);

        id<MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
        if (command_buffer == nil) {
            return -1;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return -1;
        }
        [encoder setComputePipelineState:ctx->pipeline_i2s];
        [encoder setBuffer:tensor->packed offset:0 atIndex:0];
        [encoder setBuffer:tensor->scales offset:0 atIndex:1];
        [encoder setBuffer:ctx->qvec offset:0 atIndex:2];
        [encoder setBuffer:tensor->out offset:0 atIndex:3];
        [encoder setBuffer:ctx->params offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)tensor->out_dim, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return -1;
        }
        memcpy(out, [tensor->out contents], (size_t)tensor->out_dim * sizeof(*out));
        return 0;
    }
}

int bitnet_metal_i2s_compute_pair(bitnet_metal_i2s_context_t *ctx,
                                  bitnet_metal_i2s_tensor_t *tensor_a,
                                  bitnet_metal_i2s_tensor_t *tensor_b,
                                  const int8_t *qvec,
                                  float vec_scale,
                                  float *out_a,
                                  float *out_b) {
    @autoreleasepool {
        if (ctx == NULL || tensor_a == NULL || tensor_b == NULL ||
            qvec == NULL || out_a == NULL || out_b == NULL ||
            tensor_a->out_dim != tensor_b->out_dim ||
            tensor_a->in_dim != tensor_b->in_dim ||
            bitnet_metal_i2s_prepare_qvec(ctx, qvec, tensor_a->in_dim) != 0) {
            return -1;
        }
        bitnet_metal_i2s_fill_params(ctx, tensor_a, vec_scale);

        id<MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
        if (command_buffer == nil) {
            return -1;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return -1;
        }
        [encoder setComputePipelineState:ctx->pipeline_i2s_pair];
        [encoder setBuffer:tensor_a->packed offset:0 atIndex:0];
        [encoder setBuffer:tensor_a->scales offset:0 atIndex:1];
        [encoder setBuffer:tensor_b->packed offset:0 atIndex:2];
        [encoder setBuffer:tensor_b->scales offset:0 atIndex:3];
        [encoder setBuffer:ctx->qvec offset:0 atIndex:4];
        [encoder setBuffer:tensor_a->out offset:0 atIndex:5];
        [encoder setBuffer:tensor_b->out offset:0 atIndex:6];
        [encoder setBuffer:ctx->params offset:0 atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)tensor_a->out_dim, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return -1;
        }
        memcpy(out_a, [tensor_a->out contents], (size_t)tensor_a->out_dim * sizeof(*out_a));
        memcpy(out_b, [tensor_b->out contents], (size_t)tensor_b->out_dim * sizeof(*out_b));
        return 0;
    }
}

void bitnet_metal_i2s_tensor_free(bitnet_metal_i2s_tensor_t *tensor) {
    if (tensor == NULL) return;
    tensor->packed = nil;
    tensor->scales = nil;
    tensor->out = nil;
    free(tensor);
}

void bitnet_metal_i2s_free(bitnet_metal_i2s_context_t *ctx) {
    if (ctx == NULL) return;
    ctx->qvec = nil;
    ctx->params = nil;
    ctx->pipeline_i2s = nil;
    ctx->pipeline_i2s_pair = nil;
    ctx->queue = nil;
    ctx->device = nil;
    free(ctx);
}

#else

int bitnet_metal_available(void) {
    return 0;
}

int bitnet_metal_output_create_q6k_compact(const int8_t *output_q8,
                                           const int8_t *output_scales,
                                           const float *output_d,
                                           int vocab_size,
                                           int emb_dim,
                                           bitnet_metal_output_t **out) {
    (void)output_q8;
    (void)output_scales;
    (void)output_d;
    (void)vocab_size;
    (void)emb_dim;
    (void)out;
    return -1;
}

int bitnet_metal_output_compute_q6k_compact(bitnet_metal_output_t *state,
                                            const int8_t *qhidden,
                                            float hidden_scale,
                                            float *logits) {
    (void)state;
    (void)qhidden;
    (void)hidden_scale;
    (void)logits;
    return -1;
}

void bitnet_metal_output_free(bitnet_metal_output_t *state) {
    (void)state;
}

int bitnet_metal_i2s_create(bitnet_metal_i2s_context_t **out) {
    (void)out;
    return -1;
}

int bitnet_metal_i2s_tensor_create(bitnet_metal_i2s_context_t *ctx,
                                   const uint8_t *packed,
                                   const float *scales,
                                   int out_dim,
                                   int in_dim,
                                   bitnet_metal_i2s_tensor_t **out) {
    (void)ctx;
    (void)packed;
    (void)scales;
    (void)out_dim;
    (void)in_dim;
    (void)out;
    return -1;
}

int bitnet_metal_i2s_compute(bitnet_metal_i2s_context_t *ctx,
                             bitnet_metal_i2s_tensor_t *tensor,
                             const int8_t *qvec,
                             float vec_scale,
                             float *out) {
    (void)ctx;
    (void)tensor;
    (void)qvec;
    (void)vec_scale;
    (void)out;
    return -1;
}

int bitnet_metal_i2s_compute_pair(bitnet_metal_i2s_context_t *ctx,
                                  bitnet_metal_i2s_tensor_t *tensor_a,
                                  bitnet_metal_i2s_tensor_t *tensor_b,
                                  const int8_t *qvec,
                                  float vec_scale,
                                  float *out_a,
                                  float *out_b) {
    (void)ctx;
    (void)tensor_a;
    (void)tensor_b;
    (void)qvec;
    (void)vec_scale;
    (void)out_a;
    (void)out_b;
    return -1;
}

void bitnet_metal_i2s_tensor_free(bitnet_metal_i2s_tensor_t *tensor) {
    (void)tensor;
}

void bitnet_metal_i2s_free(bitnet_metal_i2s_context_t *ctx) {
    (void)ctx;
}

#endif

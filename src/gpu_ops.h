#ifndef BITNET_GPU_OPS_H
#define BITNET_GPU_OPS_H

/* Opcodes for the elementwise stage kernel.  They are part of the contract
 * between src/bonsai.c (which asks for a stage) and src/vk/elt.comp (which
 * carries the switch), so they live in a header both sides can see even when
 * the Vulkan backend is compiled out and every stage falls back to the CPU. */

#define BQ_VK_ELT_RMS      0   /* out = rms(a) * w              */
#define BQ_VK_ELT_RMS_GATE 1   /* out = rms(a) * w * silu(b)    */
#define BQ_VK_ELT_L2       2   /* out = a / max(|a|, eps)       */
#define BQ_VK_ELT_SWIGLU   3   /* out = silu(a) * b             */
#define BQ_VK_ELT_SIGMOID  4   /* out = a * sigmoid(b)          */
#define BQ_VK_ELT_ADD      5   /* out = a + b                   */
#define BQ_VK_ELT_SCALE    6   /* out = a * w                   */

#endif

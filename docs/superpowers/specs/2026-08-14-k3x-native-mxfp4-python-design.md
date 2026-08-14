# K3X Native MXFP4 Python Runtime Design

## Goal

Remove the official Python runtime's per-call FP32 expert expansion by executing native E2M1 packed bytes and E8M0/32 scales directly on the RTX 5080, then retain admitted payloads across repeated expert use.

## Selected boundary

Build a native `sm_120` PyTorch CUDA extension whose arithmetic and nibble/scale ordering match `runtime/cuda/mxfp4.cu`. The scalar extension accepts one FP32/BF16 activation vector plus CUDA uint8 packed/scales tensors and preserves the portable store's FP32 output contract. The current portable `K3XTensorStore.mxfp4_matvec` remains the CPU/reference fallback.

Alternatives rejected for this milestone are binding the complete C++ backend into Python, which duplicates backend ownership and telemetry, and launching the existing benchmark executable per projection, which cannot preserve CUDA residency or stream ordering.

## Residency

One explicit-budget cache owned by `OfficialRuntimeContext` retains validated packed/scales tensors on CUDA. Cache identity is resolved fragment path, tensor ID, and CUDA device. Stable first admission and zero-byte defaults match Milestone 43. Admission is matrix-granular for this first bridge; whole-expert atomic admission remains the next hot-bank policy step and must not be claimed by this milestone.

## Correctness and measurement gates

1. A literal synthetic native-MXFP4 matrix must match the independent dense decoder within the existing CUDA kernel tolerance.
2. A real K3X expert matrix must avoid the Python FP32 decoder on CUDA.
3. Two calls through one context must record one miss/admission and one L0 hit with identical output.
4. A released expert triplet benchmark must compare cold and multiple warm runs and record exact payload bytes, cache telemetry, error against the portable oracle, and scope limits.

The mode remains opt-in until a complete natural-routing token and coding-quality gate pass.

## Measured expert-major addendum

The scalar resident bridge measured 13.732 ms median for one released expert. A current rerun of the existing C++ resident-grid backend measured 10.848 ms median for the complete natural Top-16 MoE FFN with zero maximum output error. Scalar E8M0 decode and warp-reduction experiments did not improve the released expert median and were reverted.

The next boundary therefore batches one layer's selected experts. Gate and up grids share the routed latent input, SiTU operates over all selected rows, the down grid consumes expert-major activated inputs, and one deterministic ordered reduction applies routing contributions. Matrix payloads remain independently cacheable; only lightweight device-pointer descriptors and activations are assembled per routed set. Natural routing and the portable scalar path remain unchanged reference modes.

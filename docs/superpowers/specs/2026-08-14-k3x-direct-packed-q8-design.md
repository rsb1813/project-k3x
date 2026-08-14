# K3X Direct-Packed Q8 Matvec Design

## Goal

Eliminate materialized Q8 trunk weights from the official execution path. Read the existing group-128 int8 codes and BF16 scales, transfer those packed extents, and perform matrix-vector multiplication directly on RTX 5080 without constructing a BF16 or FP32 weight matrix.

## Selected bridge

Use a small PyTorch C++/CUDA extension compiled for native `sm_120`. It preserves the already verified official Python graph while allowing the packed kernel to consume CUDA tensor pointers and the current PyTorch CUDA stream.

Alternatives were a complete immediate C++ graph port and a standalone `ctypes` C ABI. The complete port has a much larger correctness surface before it can measure this bottleneck. The C ABI requires separate stream, lifetime, and error ownership. The PyTorch extension is the smallest reversible bridge and can later be replaced by the production C++ runtime without changing K3X storage.

## Runtime boundary

- Add an explicit `--direct-q8` experimental switch; the default remains the B-0050 reference path.
- Keep packed handles only for two-dimensional group-128 Q8 matrices whose input width is at least one group. Small convolution/residual tensors continue through the materialized loader.
- `_bf16_matvec` dispatches packed handles to the extension and retains the existing tensor implementation otherwise.
- Transfer int8 codes and BF16 scales, convert the small activation to FP32, accumulate in FP32, and return the existing BF16 graph boundary.
- Validate metadata, lengths, device, shape, dtype, and launch status before publishing output.

## Correctness and quality gates

1. Synthetic direct-packed output must match the independent decoded reference within a recorded numerical bound.
2. A released 12,288 by 7,168 q-projection must report exact-match ratio, maximum/mean absolute error, and latency against B-0050's materialized path.
3. A fresh official layer-0 run must report routing-independent output and KDA-state divergence before any complete run.
4. Reference mode must remain bit-identical to B-0050.
5. The direct path is not a default until the complete token and quality gates justify its measured divergence.

## Performance gate

The released tensor must materially reduce decode-plus-matvec wall time and CUDA peak allocation. If it does not, reject this bridge before a complete 93-layer run. If it passes, measure the complete graph once and use the result to decide whether Q8, MXFP4, or storage residency is next.

# K3X Device-Side Q8 Decode Design

## Goal

Reduce official K3X trunk load time without changing the stored group-128 Q8 representation or the verified official graph. Compressed Q8 codes and BF16 scales cross PCIe, and CUDA reconstructs the logical BF16 tensor.

## Boundary

- Keep the existing CPU decoder as the reference and fallback path.
- Select the device decoder only when `K3XTensorStore.load` targets CUDA and the tensor uses `GROUPWISE_8BIT`.
- Validate scale bytes on CPU before transfer so malformed artifacts fail with the existing error boundary.
- Transfer int8 codes and BF16 scales directly to the requested CUDA device.
- Decode with vectorized CUDA tensor operations, trim storage padding, reshape to the recorded dimensions, and return the requested logical dtype.
- Do not alter BF16, FP32, or MXFP4 loading in this milestone.

## Correctness contract

For every valid Q8 tensor, CUDA decode converted to BF16 must be bit-identical to the existing CPU decoder converted to BF16. The complete official token must retain all 93 output/state digests, token 9689, final logit, and final hidden/state digests from B-0049.

## Measurement contract

Record a focused released-tensor decode comparison and one uninterrupted full-token B-0050 run. Compare wall time and load/decode time against B-0049. This bridge is accepted as the default only if correctness passes and the measured boundary does not regress materially.

## Deliberate limitation

The bridge still materializes a BF16 weight and the current graph may subsequently materialize FP32 for matvec. A direct-packed fused Q8 matvec remains the next optimization if this bridge alone cannot approach the 5 tok/s warm-decode target.

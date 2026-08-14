# K3X Packed Q8 Residency Design

## Goal

Expose the 40.30x resident direct-Q8 kernel benefit by avoiding repeated K3X reads, Python byte copies, and H2D for admitted packed matrices across repeated execution in one runtime context.

## Policy

Use explicit byte budgets and stable first admission.

- L0 budget stores int8 codes and BF16 scales on one CUDA device.
- L1 budget stores validated CPU code/scale tensors when L0 has insufficient room.
- L0 hit performs no file read or H2D.
- L1 hit performs no file read but still copies packed tensors to CUDA.
- A miss reads and validates the K3X extents, then attempts L0 followed by L1 admission.
- Default budgets are zero, preserving B-0050/B-0051 behavior.

LRU is deliberately not the first policy. A deterministic 93-layer sequential scan with a cache smaller than the working set can evict every item just before its next-token reuse. Stable first admission guarantees that an admitted subset remains warm. Frequency/profile/deadline policies can replace admission after real multi-token traces exist.

## Ownership and identity

One `OfficialRuntimeContext` owns one shared packed-Q8 cache across all lazily opened fragment stores. Cache keys include resolved fragment path, tensor ID, and CUDA device. The cache never changes tensor contents or routing and validates Q8 metadata/scales before admission.

## Telemetry

Record L0/L1 hits, misses, admissions, rejected bytes, resident bytes, and configured budgets. No cache-hit or traffic claim may be made from requested logical bytes alone.

## Gates

1. A synthetic K3X matrix must read once across two L0-resident matvecs and retain the same direct-Q8 output.
2. Zero-budget behavior must retain repeated reads and B-0051 semantics.
3. Two fresh official layer-0 executions sharing one context must retain identical direct-Q8 outputs while the second execution reports cache hits and lower wall time.
4. A complete multi-token run is deferred until recurrent-state ownership can consume the cache without replaying a first-token graph.

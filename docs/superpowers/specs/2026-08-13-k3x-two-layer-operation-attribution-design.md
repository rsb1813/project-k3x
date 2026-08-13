# K3X Milestone 35 Two-Layer Operation Attribution Design

## Goal

Identify which already-timed CUDA operation classes dominate the exact two-layer device-closure front and tail regions before choosing any fusion or synchronization optimization.

## Evidence boundary

B-0035 attributes 62.934% of device-closure wall time to front and 37.010% to tail. Canonical host routing is only 0.035%. The backend already emits successful `ProfileEvent` records for the official KDA call, device route preparation, MoE FFN, and zero-device-time transfer accounting. M35 will classify those existing events inside the wrapper's current front and tail snapshots. It will not create CUDA events, add synchronization, or change execution order.

## Alternatives

1. Aggregate existing `ProfileEvent::operation` values within each front/tail snapshot. This is selected because it is synchronization-free, bounded, and directly testable against the profiler event stream.
2. Add new per-kernel CUDA events inside official KDA and MoE kernels. This could produce finer data, but it changes instrumentation and synchronization risk before the current coarse operations are measured.
3. Use Nsight Systems or Nsight Compute only. External traces remain useful later, but they do not provide a repository-owned, reproducible JSON/CSV evidence contract.

## Data contract

The caller-owned attribution result gains explicit device-time buckets.

- Front KDA is successful `dense_matvec` device time observed only inside a front snapshot.
- Front route preparation is successful `moe_mix` device time observed only inside a front snapshot.
- Tail FFN is successful `moe_mix` device time observed only inside a tail snapshot.
- Front and tail unclassified device time are checked remainders between the existing regional totals and recognized buckets.

Unknown or future successful operations do not disappear. They accumulate in the matching unclassified bucket. Arithmetic overflow or a classified sum above the regional device total fails the transaction. The caller accumulator remains unchanged on failure.

## Compatibility

Attribution disabled preserves `k3x-official-two-layer-bench-v1`. The M34 schema remains testable and unchanged for callers using the current harness switch. A new explicit operation-attribution switch emits `k3x-official-two-layer-operation-attribution-v1`; it is never the default.

## Measurement

B-0036 will reuse the B-0035 artifact, manifest, oracle, 4 GiB admission limit, exact natural Top-16 routing, three warmups, and twenty measured two-position sequences per row. It will verify all B-0035 correctness, identity, residency, traffic, digest, and atomic-publication invariants, plus regional device-time closure.

## Non-goals

M35 does not implement fusion, alter routing, change precision, claim token throughput, measure physical PCIe/NVMe traffic, widen the layer boundary, download a complete shard/checkpoint, or provision cloud resources.

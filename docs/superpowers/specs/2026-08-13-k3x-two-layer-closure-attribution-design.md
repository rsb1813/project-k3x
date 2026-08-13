# K3X Milestone 34 Two-Layer Closure Attribution Design

## Status and purpose

Milestone 33 proved that exact device closure removes 57,344 logical inter-layer bytes in each direction per two-position sequence but regresses the median by 13.823803%. Milestone 34 measures where that time is spent before any fusion or wider closure is attempted.

This milestone is attribution-only. It must not change mathematical execution, routing, residency, default mode, K3X v1, or production dispatch.

## Alternatives considered

1. Add new CUDA events around every front and tail sub-kernel. This provides the finest breakdown but adds event objects and synchronization to the path being measured before the coarse bottleneck is known.
2. Reimplement the two-layer sequence inside the benchmark harness. This avoids runtime API changes but duplicates ownership and cleanup logic and risks benchmarking a different graph.
3. Reuse the existing backend `Profiler` events and take snapshots around each existing front and tail call. This measures device time already produced by the exact operations, adds no new CUDA synchronization, and keeps one orchestration authority.

Option 3 is accepted. Per-kernel events are deferred unless the coarse attribution proves that one front or tail region dominates.

## Interface and data flow

`OfficialTwoLayerAttribution` is an optional caller-owned accumulator passed to `official_two_layer_cuda`. It contains a pointer to the same `Profiler` supplied to `make_cuda_backend` and cumulative nanoseconds for these mutually interpretable regions.

- `front_wall_nanoseconds` is host elapsed time inside four `official_layer_front` calls.
- `front_device_nanoseconds` is the sum of new successful profiler-event device time emitted by those calls.
- `route_wall_nanoseconds` covers canonical host Top-16 selection and exact expert view resolution between front and tail.
- `tail_wall_nanoseconds` is host elapsed time inside four `official_layer_tail` calls.
- `tail_device_nanoseconds` is the sum of new successful profiler-event device time emitted by those calls.
- `unattributed_wall_nanoseconds` is the nonnegative remainder of the complete wrapper wall time after front, route, and tail wall regions. It includes wrapper validation, state publication/moves, result assembly, and cleanup bookkeeping.

The accumulator is reset at call entry. Disabled attribution passes no pointer and executes the historical path without profiler snapshots or new JSON fields.

The benchmark harness adds explicit `--attribution true|false`, default `false`. False preserves the exact B-0034 schema. True requires a profiler-backed backend and emits schema `k3x-official-two-layer-attribution-v1` with the six attribution fields. Token, TPS, TTFT, quality, utilization, bandwidth, and physical traffic fields remain forbidden.

## Correctness and failure behavior

Attribution never changes routing, token ownership, state ownership, weight admission, or cleanup. A failed execution returns its original error and does not publish a partial attribution record. Device nanoseconds include only successful events added inside the measured region. Counter overflow and profiler event shrinkage fail closed with `INVALID_STATE`.

The existing default CLI output must remain byte-for-byte schema-compatible with B-0034 tooling. B-0034 committed evidence is reverified after the change.

## Measurement contract

B-0035 uses the same bounded artifact, oracle, 4 GiB admission, exact modes, three warmups, and twenty measured sequences as B-0034. It publishes two attribution rows but does not rank modes from a new wall sample alone. The decision is based on the share of wall time and device time assigned to front, route, tail, and remainder.

The first follow-up optimization may target only a region that explains a material share of the measured regression. If attribution is diffuse or profiler overhead changes the result materially, no fusion is accepted.

## Non-goals

- No kernel fusion.
- No wider than two-layer closure.
- No device-side Top-K.
- No production token execution.
- No full checkpoint, complete shard, paid cloud resource, or SKYFORGE execution.
- No projected TPS or physical PCIe/NVMe claim.

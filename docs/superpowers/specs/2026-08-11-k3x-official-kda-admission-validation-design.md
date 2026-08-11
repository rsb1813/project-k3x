# K3X Milestone 30 Official KDA Admission-Validation Design

## Status

Accepted on 2026-08-11 under the standing authorization to continue non-billable work before Cloud Run provisioning. This milestone attributes the official KDA host-orchestration gap identified by B-0030. It changes no model graph, tensor bytes, routing decision, KDA state, quality mode, production artifact capability, or default validation policy.

## Goal

Measure how much repeated host-side finiteness scanning of the official KDA immutable weights contributes to the resident complete-layer wall time. Reuse the admission-validation authority implemented in Milestone 23 instead of creating a second KDA-specific cache.

The milestone must prove all of the following.

1. Dynamic hidden inputs and KDA convolution/recurrent state remain shape- and finiteness-validated on every call.
2. The eight BF16 and six F32 immutable KDA views are fully scanned on first admission and may be reused only under the exact same tensor ID, host pointer, byte length, rows, and columns.
3. A failed first admission publishes no partial registry entries and performs no resident upload or CUDA launch.
4. Per-call and admission modes produce identical output, final V-first state, routes, contributions, and resident traffic on the bounded official fixture.
5. A fixed B-0031 experiment separates validation counters and time from kernel and residual orchestration time without claiming token throughput or changing a default.

## Evidence boundary

B-0030 measured a 53.772681 ms median wall-time gap between resident A-to-B incremental and resident A+B full execution, while their aggregate device time differed by only 0.416216% per sequence. That supports an attribution experiment, not a causal conclusion. The official KDA backend currently rechecks finiteness across 887,800,832 immutable weight bytes on every KDA call, including eight BF16 views totaling 887,160,832 bytes and six F32 views totaling 640,000 bytes.

M30 does not download any new tensor payload. It reuses the ignored 1,829,310,720-byte M29 artifact and its checksum-bound route/state manifest. It remains a non-executable layer fixture with no token semantics.

## Approaches considered

### Accepted: extend the existing backend admission registry

Use `CudaWeightValidationMode::admission`, `ImmutableWeightIdentity`, and the existing runtime counters for the fourteen official KDA immutable views. Structural validation and duplicate-ID rejection still run every call. Admission classifies every view as a registry hit, a required scan, or an identity conflict before scanning. All required scans complete successfully before any new identity is inserted.

This keeps one validation policy and one telemetry vocabulary across resident MoE and KDA execution.

### Rejected: a separate official-KDA validation cache

A dedicated cache would make the local implementation smaller at first, but it would duplicate identity, atomicity, telemetry, and policy semantics already established by M23. The two caches could diverge under later complete-layer or multi-layer work.

### Rejected: harness-only prevalidation and backend bypass

Prevalidating in `k3x_cuda_official_layer_bench` could expose a timing result quickly, but it would weaken the backend contract and would not be safe for a future runtime caller. The benchmark must exercise the same validation boundary that production-capable code would use.

## Validation contract

The official KDA call has two classes of data.

### Dynamic data validated on every call

- FP32 hidden input.
- Three BF16 convolution histories.
- FP32 V-first recurrent state.
- Configuration and all derived dimensions.

No admission mode may cache or skip these checks.

### Immutable data eligible for admission

- BF16 matrices `q_proj`, `k_proj`, `v_proj`, `f_a_proj`, `f_b_proj`, `b_proj`, `g_proj`, and `o_proj`.
- F32 matrices or vectors `q_conv`, `k_conv`, `v_conv`, `A_log`, `dt_bias`, and `o_norm`.

Every call validates nonzero unique tensor IDs, expected rows and columns, exact element counts, and checked byte totals. In `per-call` mode, all fourteen payloads are scanned for finiteness. In `admission` mode, each payload is represented by the existing identity tuple.

```text
(tensor_id, host_pointer, byte_length, rows, columns)
```

An absent tensor ID requires a scan. An exact tuple is a hit. A matching tensor ID with any different identity field fails before scanning, upload, scratch allocation, or launch.

Admission is atomic at the call boundary. The backend first classifies all fourteen views, then scans every new view, then inserts all new identities only if every scan succeeds. A non-finite view therefore cannot seed the registry for other views from the same failed call.

Admission mode is supported for this boundary only with exact resident weights. The benchmark rejects `admission + transient`; `per-call` remains available for both transient and resident execution. The global default remains `per-call` until measured evidence and a separate decision justify otherwise.

## Telemetry contract

Reuse the existing cumulative backend counters.

- `immutable_validation_scans` counts payload views actually scanned.
- `immutable_validation_hits` counts exact admitted identity hits.
- `immutable_validation_bytes` counts host payload bytes actually scanned.
- `immutable_validation_nanoseconds` measures scanning work and is zero for a measured interval containing only hits.

The official harness publishes cold deltas separately from measured deltas. The first resident admission call must report fourteen cold scans, zero cold hits, and 887,800,832 cold scanned bytes. After cold admission and warmups, a measured resident admission interval must report zero scans and bytes, fourteen hits per KDA call, and zero validation nanoseconds. A per-call interval must report fourteen scans and 887,800,832 bytes per KDA call, with zero hits.

The harness also publishes `validation: per-call|admission`. Existing B-0030 evidence and schemas remain immutable historical records.

## Correctness and failure gates

- Tiny official KDA CUDA tests cover first admission, repeated hit, per-call scans, exact output/state parity, identity conflict, duplicate tensor ID, non-finite BF16 and F32 payloads, and dynamic non-finite state.
- A failed non-finite first admission leaves the registry empty, resident bytes unchanged, and launch counters unchanged; a subsequent valid call performs all fourteen scans rather than reporting partial hits.
- A different host allocation under a previously admitted tensor ID fails before upload or launch.
- The actual bounded official fixture preserves the B-0030 output and state digests and maximum absolute error tolerance.
- Compute Sanitizer must report zero errors for the admission-mode resident incremental fixture.
- Production `k3x_run` must continue to reject the fixture with `NON_EXECUTABLE_ARTIFACT`.

## B-0031 measurement contract

B-0031 is a fixed four-row resident-only transaction.

1. `ab-incremental-resident-per-call`.
2. `ab-incremental-resident-admission`.
3. `ab-full-resident-per-call`.
4. `ab-full-resident-admission`.

Each row uses three warmups and twenty measured sequences. The runner writes one canonical raw JSON per row, canonical summary JSON, LF-only summary CSV, and checksums for the artifact, manifest, runner, raw rows, aggregate, and CSV. It verifies exact cross-row route, contribution, output, final-state, resident-byte, and warm-zero-weight-H2D parity before publication.

The experiment records wall latency, kernel time, residual orchestration time, validation scans/hits/bytes/time, KDA calls, state traffic, cache counters, VRAM, process RSS, and logical Reader traffic. It does not emit decode or prefill tokens per second, TTFT, physical NVMe/PCIe traffic, GPU utilization, memory bandwidth, or quality results.

No row is rerun or selected for favorable timing. A failed transaction leaves no published partial evidence.

## Completion boundary

M30 is complete only after the design and TDD plan are committed, RED and GREEN tests demonstrate the validation contract, the full local verification matrix and actual-artifact Compute Sanitizer pass, B-0031 is published exactly once from a reviewed implementation, README and the TITAN Ledger are synchronized, and the public pull request plus post-merge CI pass.

The measured result determines whether the next non-cloud step should remove another host orchestration cost, widen official graph coverage, or revisit the default validation policy. No choice is made in advance.

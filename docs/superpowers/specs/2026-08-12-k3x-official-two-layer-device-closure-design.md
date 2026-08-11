# K3X Official Two-Layer Device Closure Design

Date: 2026-08-12  
Status: accepted design; not implemented or measured  
Decision: D-071

## Objective

Milestone 33 will close the smallest real cross-layer CUDA boundary in the released Kimi K3 text graph. The bounded target is official decoder layers 1 and 2, which are both KDA plus Stable LatentMoE layers. It must preserve exact per-layer KDA recurrence, canonical natural Top-16 routing, exact native MXFP4 experts, and the existing host/default path.

The milestone is successful only if a real layer-1 output becomes the layer-2 input without an intervening host activation round trip. Replaying layer 1 twice is not multi-layer evidence.

## Evidence that selects the boundary

- The released configuration has 93 decoder layers. Layers 1 and 2 are KDA layers; layer 3 is MLA. The synthetic reference graph uses the same KDA/KDA/KDA/MLA prefix.
- B-0033 measures a one-layer split boundary. Device route preparation lowers host orchestration but adds four kernels and two router-logit synchronizations per two-position sequence, leaving the wall median nearly unchanged.
- The existing official converter, oracle, artifact name, harness, and materialization guards are explicitly fixed to layer 1.
- The CUDA backend owns only one KDA device-state slot. It cannot retain independent recurrent states for layers 1 and 2.
- The current wrapper computes the front Attention Residual and input RMSNorm on the CPU, returns KDA output to the host, and returns final MoE output to the host. Chaining two existing calls would therefore preserve the measured round trips rather than close them.
- K3X v1 tensor and layer directories already identify tensors by canonical names and layer IDs. A new file-format version is not required for a bounded two-layer artifact.

## Alternatives

### Accepted: real layers 1 and 2 with a staged device bridge

Manufacture one bounded, checksum-bound two-layer fixture. Retain the layer-1 final hidden vector and the shared Attention Residual block source on the device. Use a capacity-two KDA state registry keyed by exact layer identity. Return raw router logits at each layer so the canonical host routing rule and dynamic expert-residency decision remain visible.

This is the smallest design that tests real cross-layer activation residency without changing routing semantics or introducing an arbitrary production state manager.

### Rejected: replay layer 1 twice

This would reuse one layer's weights and state identity. It cannot prove layer-2 tensor selection, independent KDA recurrence, cross-layer token ownership, or real layer transition behavior. It remains useful only as a tiny unit-test technique and cannot support the M33 claim.

### Rejected: monolithic two-layer GPU call with device Top-K

This would combine front-end kernels, routing policy, expert lookup, and two layers in one large interface. It would remove the scheduling point needed by exact cold rescue and future cache policy work, duplicate the canonical natural Top-16 rule, and make failures difficult to attribute. It is too large for the next measured boundary.

## Bounded manufacturing contract

The source planner accepts exactly layers 1 and 2 for this milestone and validates that both are released KDA layers with the pinned Kimi K3 configuration and revision. Each layer may reside in a different safetensors shard; every planned tensor remains bound to its own pinned shard header, range, and digest.

Materialization is dependency ordered and resumable.

1. Materialize and verify both layers' KDA and always-active MoE tensors as content-addressed range objects.
2. Evaluate position A through layer 1, fetch only its selected exact experts, and compute the exact layer-1 output.
3. Feed that output and the unchanged block source into layer 2, derive its canonical route, and fetch only its selected exact experts.
4. Repeat the interleaved layer-1/layer-2 flow for position B with independent recurrent state per layer.
5. Assemble one execution-ordered source microshard and one K3X v1 artifact containing both layer directories. Bind the two-layer route/state/oracle manifest, source fingerprint, tensor digests, and K3X root before publication.

No complete checkpoint, complete source shard payload, or unrelated expert is downloaded. Existing content-addressed layer-1 objects may be reused only after rehashing. A dry-run reports all accepted ranges and worst-case requested bytes before payload transfer.

## Runtime interface

The new experimental path is a staged boundary rather than a production graph API.

### Opaque activation token

`OfficialLayerHiddenToken` identifies backend owner, generation, producing layer, hidden width, and one of two bounded ping-pong activation slots. It exposes no device pointer. The slot retains both the final hidden vector and the sequence's Attention Residual block source.

- The first layer front accepts host hidden and block vectors.
- A later layer front consumes exactly one valid preceding-layer token.
- A non-final tail returns a new token instead of copying final hidden to the host.
- The final tail publishes the host output.
- Stale, cross-backend, wrong-layer, wrong-width, double-consumed, or unexpectedly live tokens fail closed.
- Any route, expert-resolution, FFN, or next-front failure explicitly discards the live token.

### Capacity-two KDA state registry

The backend owns exactly two official KDA state slots keyed by layer IDs 1 and 2. Each slot has an independent generation and configuration identity. A state-mutating operation consumes the prior token and returns a new token for that same layer.

This registry exists only to execute the bounded A/B-by-layer schedule. It is not a multi-session registry, eviction policy, VAULT persistence layer, or arbitrary 93-layer state manager.

### Layer front and tail

For each layer, the device front performs exact self Attention Residual, input RMSNorm, KDA, residual prefix update, MLP Attention Residual, post RMSNorm, and the raw 896-row router matvec. It returns router logits and a single-use prepared token. The host applies the existing canonical sigmoid, correction, natural Top-16 ordering, tie break, selected-mass calculation, and contribution normalization.

The tail resolves the selected exact experts through the existing reader/residency boundary, consumes the prepared token, executes the exact native-MXFP4 and shared FFN, and either retains or publishes the final hidden vector.

The first implementation may keep one router-logit synchronization per layer. M33 does not claim to remove both B-0033 synchronization points. Kernel fusion is considered only after the complete two-layer boundary is measured.

## Execution and ownership sequence

For two positions A and B, execution is interleaved in model order.

1. A/layer 1 seeds KDA state 1 and retains hidden token 1.
2. A/layer 2 consumes hidden token 1, seeds KDA state 2, and publishes or retains the position output.
3. B/layer 1 continues KDA state 1 and retains hidden token 2.
4. B/layer 2 consumes hidden token 2, continues KDA state 2, and publishes the final output and both final states.

Layer state and activation generations are independent. An error invalidates only live tokens belonging to the failed bounded sequence, without silently publishing partial state.

## Modes and compatibility

- Host round-trip KDA state and host route preparation remain defaults.
- The M31 single-slot and M32 prepared-token interfaces remain source compatible until the two-layer path proves replacement coverage.
- The new path is explicit, resident-weight, admission-validated, incremental, and benchmark-only.
- Production `k3x_run` continues to reject the bounded artifact as `NON_EXECUTABLE_ARTIFACT`.
- Natural Top-16, exact expert IDs and contributions, exact rescue semantics, and native MXFP4 decoding do not change.

## TDD and verification gates

Implementation proceeds in four independently reversible stages.

1. Generalize bounded planning/oracles to real layer 2 and build a tiny exact two-layer fixture. Verify per-layer routes, interleaved states, output, source hashes, resume behavior, and production rejection.
2. Replace the single KDA state slot only for the explicit path with a capacity-two layer registry. Verify stale, cross-layer, cross-backend, overwrite, publish, discard, and partial-failure behavior.
3. Add the opaque hidden token and full front/tail bridge. Verify host/device parity at every layer boundary, exact route/contribution/output/final-state identity, zero inter-layer hidden D2H/H2D, and cleanup under every downstream failure.
4. Add fixed B-0034 tooling and run one sealed host-versus-device two-layer transaction after CUDA correctness, actual-artifact, Compute Sanitizer, production-guard, and full regression gates pass.

## B-0034 measurement contract

B-0034 will report two-layer sequence wall time, kernel time, orchestration time, logical H2D/D2H by category, resident and peak VRAM, validation attribution, route/logit synchronizations, exact route and state digests, and maximum numerical error. Raw JSON, LF CSV, summary, runner, artifact/manifest, and aggregate hashes are sealed atomically.

It will not report decode tok/s, prefill tok/s, TTFT, quality, physical PCIe/NVMe traffic, native-Linux authority, or full-model cache behavior because the bounded fixture cannot measure those quantities honestly.

## Explicit non-goals

- No full Kimi K3 checkpoint download.
- No paid Cloud Run or other billable resource provisioning.
- No MLA layer, arbitrary 93-layer graph, multi-session state, VAULT persistence, or ORBIT prediction.
- No device-owned natural Top-K policy, reduced precision, adaptive Top-K, proxy, or pruning.
- No production default change or TPS projection before measurement.

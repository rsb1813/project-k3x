# K3X Milestone 31 Official KDA Device-State Handoff Design

## Status

Accepted on 2026-08-11 under D-069 and the standing authorization to continue non-billable work before Cloud Run provisioning. This milestone is a bounded exact experiment. It changes no K3 graph, tensor bytes, routing decision, quality mode, production artifact capability, or default host-state behavior.

## Goal

Keep the official KDA convolution and V-first recurrent state on one CUDA backend between two incremental calls, then publish the final state explicitly. The experiment must remove only the intermediate state D2H plus next-call state H2D while preserving the current host-state path as the default and correctness oracle.

The milestone must prove all of the following.

1. Host round-trip execution remains byte-for-byte compatible with M30.
2. Device seed, continuation, and final publication preserve output, routes, contributions, convolution histories, and recurrent state.
3. No raw CUDA pointer crosses the backend interface.
4. Stale, cross-backend, wrong-layer, wrong-config, and already-consumed state tokens fail before upload or launch.
5. The device path records one initial state H2D and one final state D2H across A-to-B incremental execution, with no intermediate state transfer.
6. A fixed B-0032 transaction measures the bounded difference without claiming token throughput or changing a default.

## Evidence boundary

B-0031 admission-mode incremental/full medians are 70.584413 and 67.236923 ms. Their kernel time is nearly identical at 33.889030 and 33.958984 ms per sequence. Incremental execution performs two 6,512,640-byte state transfers in each direction; full execution performs one. The remaining 3.347490 ms wall gap supports a state-handoff experiment but does not prove the entire gap is transfer cost.

M31 reuses the ignored 1,829,310,720-byte M29 artifact and its checksum-bound manifest. It downloads no new tensor payload and retains the production `NON_EXECUTABLE_ARTIFACT` guard.

## Approaches considered

### Accepted: one backend-owned slot with an opaque token

Add a dedicated KDA state allocation owned by the CUDA backend. A successful device seed returns an opaque token containing backend ownership and generation identity. Continuation consumes exactly that token and returns a new generation. Final publication consumes the token, executes the next call, and copies the resulting state to host.

The token contains no device pointer. The backend binds the active generation to the exact layer and KDA configuration. Only one device state is active per backend in this bounded milestone.

### Rejected: implicit reuse inside the existing scratch allocation

The current recurrent-state offset depends on sequence-sized intermediate buffers. A later call with a different sequence length may place state at a different address, and scratch growth may reallocate the entire block. Implicit reuse also cannot reject stale or cross-session callers. State therefore moves to a dedicated allocation.

### Deferred: a multi-session device-state registry

A registry with eviction, independent streams, and concurrent session ownership is eventually useful for a complete runtime, but it broadens lifetime, memory-pressure, and concurrency policy before the bounded transfer hypothesis is measured. M31 implements one explicit slot and leaves multi-session residency proposed.

## API and lifetime contract

The existing host-state call remains source-compatible and default. New control values express four operations.

- `host_roundtrip`: upload host state, execute, and publish host state.
- `device_seed`: upload host state, execute, retain resulting state, and return a token without publishing host state.
- `device_continue`: consume a token, execute, retain resulting state, and return a new token.
- `device_publish`: consume a token, execute, and publish final host state without retaining a token.

The result explicitly reports whether host state was published and may carry an opaque token. Unpublished state vectors must be empty. Continuation modes do not accept or inspect host state payloads.

Tokens are single-use. A successful continuation consumes generation N and returns generation N+1. A host round trip or new seed invalidates any previous active token before mutating the slot. Once a validated operation is about to upload or launch, the prior token is invalidated; any later CUDA failure leaves no reusable token. Structural, immutable-weight, dynamic-hidden, and token checks still fail before state mutation.

M31 does not claim concurrent use of one backend. Cross-backend owner identity still prevents accidental token reuse across independent backend instances.

## Memory and transfer contract

The dedicated allocation holds three BF16 convolution histories and one FP32 V-first recurrent state. For the official layer-1 configuration this is 6,512,640 bytes. It is tracked as reusable scratch memory, not resident weight memory.

For a two-call A-to-B sequence.

| Path | State H2D | State D2H | Output D2H |
|---|---:|---:|---:|
| Host round trip | 13,025,280 B | 13,025,280 B | unchanged |
| Device handoff with final publish | 6,512,640 B | 6,512,640 B | unchanged |

Weights, activations, output publication, host Attention Residual, routing, and MoE execution remain unchanged. The expected logical state-transfer reduction is not a physical PCIe measurement.

## Telemetry contract

Add cumulative counters for successful device seeds, continuations, publications, and invalidations. Existing `official_kda_state_h2d_bytes` and `official_kda_state_d2h_bytes` remain the byte authorities. The harness publishes cold and measured deltas only under an explicit state-transfer option so historical B-0030/B-0031 schemas remain closed.

## Correctness and failure gates

- Tiny CUDA tests cover host default parity, seed/continue/publish parity, single-use generations, wrong owner, wrong layer/config, empty or unexpected host state, and invalidation after a state-mutating failure.
- The official-layer wrapper proves A-to-B device handoff matches the independent portable full result and host incremental result.
- The actual bounded artifact preserves M30 output/state digests, natural routes, contribution vectors, resident bytes, and maximum-error tolerance.
- Compute Sanitizer reports zero errors on the device-handoff path.
- Production `k3x_run` continues to reject the fixture.

## B-0032 measurement contract

B-0032 is a fixed three-row exact-resident admission transaction.

1. A-to-B incremental with host state.
2. A-to-B incremental with device handoff and final publication.
3. A+B full with host state as the same-transaction lower-bound comparator.

Each row uses three warmups and twenty measured sequences. The transaction records wall, kernel, orchestration, state traffic, state-operation counters, output/state/route identities, cache counters, VRAM, RSS, and logical Reader traffic. It writes canonical raw JSON, summary JSON, LF CSV, and strict hashes atomically. No row is rerun or selected for favorable timing.

The benchmark does not emit decode/prefill tok/s, TTFT, quality, physical NVMe/PCIe traffic, utilization, or bandwidth. A favorable bounded result does not make device state the default or prove full-model benefit.

## Completion boundary

M31 is complete only after the design and plan are committed, RED and GREEN tests prove lifetime and transfer contracts, actual-artifact sanitizer passes, B-0032 is published exactly once, the full local verification matrix passes, the TITAN Ledger is synchronized with `PROJECT_STATE.md` last, and public pull-request plus post-merge CI pass.

# K3X Official MoE Device Routing Design

## Scope

Milestone 32 measures one exact, opt-in CUDA route-preparation boundary inside the existing bounded official layer-1 fixture. It moves MLP Attention Residual, post-residual RMSNorm, and the 896-row router matvec to the device, retains the prepared prefix and normalized hidden vector on the backend, and returns only canonical routing evidence needed to select exact resident MXFP4 experts.

The milestone does not add a second official layer, download another tensor range, change natural Top-16 routing, change the production runtime default, or claim token throughput, quality, physical PCIe traffic, native-Linux authority, or full-model behavior.

## Decision

Use a two-stage opaque prepared-activation handoff.

1. `prepare_official_moe_route` accepts host prefix/block vectors plus immutable residual, normalization, and router weights. It executes the exact BF16 residual preparation and router matvec on one CUDA stream.
2. The backend returns 896 raw router logits and a single-use owner/generation token. It does not return a CUDA pointer or the prepared activation.
3. The existing canonical host routing rule converts logits to sigmoid scores, applies correction, selects natural Top-16 with expert-ID tie breaking, and normalizes the selected contributions.
4. `official_mxfp4_moe_ffn_prepared` consumes the token and the selected exact expert views, then executes the existing shared/routed MXFP4 FFN using the retained prefix and normalized hidden vector.
5. Any route, expert lookup, or FFN failure discards the token. Successful consumption and explicit discard both make the generation stale.

This is preferred over adding another official layer because B-0032 leaves host residual/routing/API orchestration as the measured local bottleneck, while a second layer would add payload and lifetime scope without removing it. A monolithic whole-layer call is deferred because expert selection must remain an observable scheduling boundary for later cache and cold-rescue work.

## Interfaces

`OfficialMoeRoutePrepareView` owns views for residual norm, residual projection, post norm, and router. Correction remains a canonical host-policy input and never enters the CUDA preparation API. `OfficialMoePreparedToken` contains only `owner` and `generation`. `OfficialMoeRoutePrepareResult` contains `executed`, the token, and one finite raw logit per router row, which is 896 on the bounded official fixture.

The backend gains three opt-in virtual operations.

- `prepare_official_moe_route(prefix, block, weights, epsilon, layer, phase)`.
- `official_mxfp4_moe_ffn_prepared(token, ffn_weights, experts, expert_ids, contributions, epsilon, situ_beta, situ_linear_beta, layer, phase)`.
- `discard_official_moe_prepared(token)`.

The portable backend keeps the default `backend_unavailable` behavior. Existing `official_mxfp4_moe_ffn` and the host routing path remain unchanged.

## Exactness

CUDA kernels preserve the released boundary order.

- Round prefix and block elements to BF16.
- Compute the two RMS-normalized residual projection scores in deterministic index order.
- Apply the two-way stable softmax and round the mixed residual to BF16.
- Apply post RMSNorm and round each output to BF16.
- Compute every BF16 router dot product in deterministic column order and return raw F32 logits.
- Run sigmoid, correction, natural Top-16 selection, expert-ID tie breaking, and contribution normalization through one shared canonical CPU helper for both host and device paths.

Tiny CUDA tests require exact route IDs and bounded numerical parity for logits, contributions, prepared activation, and final output. Formal B-0033 additionally requires identical route IDs, output digest, final KDA-state digest, resident-byte identity, and zero warm weight H2D across paired host/device-routing rows. A tolerance change or routing mismatch fails the transaction rather than weakening the oracle.

## Lifetime and failure rules

The backend owns one grow-only prepared-activation slot independent of operation scratch and KDA state. Preparation validates and admits the immutable route weights before seeding the slot. A token is valid only for the exact backend owner, generation, layer, and hidden width that created it.

- A new preparation invalidates any older unconsumed generation before mutation.
- Consumption validates every field before upload or kernel launch and consumes the token once.
- Cross-backend, stale, zero, wrong-layer, wrong-width, duplicate-consumption, and mismatched-weight calls fail closed.
- Host/default FFN calls invalidate an outstanding prepared activation so implicit reuse is impossible.
- Wrapper failures after preparation call explicit discard. Destruction releases the slot through existing RAII device storage.

This milestone intentionally keeps one slot. Multi-session registries, overlap, eviction, VAULT persistence, and cross-layer ownership remain outside M32.

## Telemetry and B-0033

Runtime telemetry records preparation calls and kernels, router-logit D2H bytes, prepared seeds/consumes/discards/invalidations, and prepared-slot bytes. Existing activation, weight, synchronization, state, and resident-weight counters remain authoritative and are not redefined.

B-0033 fixes two rows in one process, in order.

1. Device-state incremental with host routing, resident weights, and admission validation.
2. Device-state incremental with device route preparation, resident weights, and admission validation.

The transaction uses three warmups and twenty measured sequences per row. It records raw JSON, LF-only CSV, summary JSON/CSV, artifact/manifest/runner hashes, raw hashes, and an aggregate digest through the existing atomic evidence pattern. It is non-ranking and runs exactly once after smoke, sanitizer, and strict verifier gates pass.

## Verification

- Controlled CUDA RED proves the new interface and lifetime rules do not already exist.
- Tiny CPU/CUDA parity covers residual preparation, all 896 logits, Top-16 IDs/contributions, final FFN output, and token failures.
- The unchanged host path remains covered by historical B-0030 through B-0032 regressions.
- The actual bounded fixture must pass host/device routing parity, production `NON_EXECUTABLE_ARTIFACT`, and Compute Sanitizer before B-0033.
- Completion requires CPU, liburing/direct, ASan/UBSan, CUDA, actual-artifact, evidence, and documentation gates plus public correctness and CodeQL.

## Rejected or deferred

- A second official layer is deferred until the measured single-layer host routing boundary is closed or rejected by B-0033.
- A monolithic whole-layer kernel is deferred because it hides the dynamic expert scheduling boundary.
- Device Top-K selection is deferred; canonical CPU selection keeps policy identity while this milestone attributes preparation and router matvec only.
- Multiple prepared slots, asynchronous expert fetch overlap, reduced precision, adaptive Top-K, proxy experts, pruning, and default changes are not part of M32.

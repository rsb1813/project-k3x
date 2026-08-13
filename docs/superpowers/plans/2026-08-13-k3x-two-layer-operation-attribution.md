# K3X Milestone 35 Two-Layer Operation Attribution Plan

**Goal:** Expose existing profiler device-time operations inside the exact two-layer front and tail regions, then seal one fixed B-0036 measurement without changing execution.

**Tech stack:** C++20, CUDA 13.3, K3X `Profiler`, pytest, CTest, Compute Sanitizer.

## Task 1 — RED ownership and classification contract

- Extend the CUDA two-layer test to require explicit front-KDA, front-route-preparation, tail-FFN, and unclassified device buckets.
- Assert exact regional closure, disabled-path zero/default behavior, and caller accumulator non-publication on failure.
- Verify RED compilation or assertion failure before implementation.

## Task 2 — Minimal wrapper aggregation

- Classify only successful existing profiler events between each front/tail snapshot.
- Use checked accumulation and checked regional remainders.
- Add no CUDA event, synchronization, backend call, or execution branch.
- Pass focused CUDA and portable CTest plus strict warning compilation.

## Task 3 — Opt-in harness schema

- Add a strict operation-attribution mode while retaining the historical and M34 schemas unchanged.
- Emit the new buckets only under `k3x-official-two-layer-operation-attribution-v1`.
- Verify unknown values, disabled compatibility, exact identities, and malformed arithmetic rejection.

## Task 4 — B-0036 evidence transaction

- Add a fail-closed runner/verifier with raw JSON, summary JSON, and CSV digest parity.
- Verify fixed row names, 3/20 sampling, artifact/manifest/oracle/runner hashes, exact routes and identities, numerical tolerances, zero warm weight H2D, traffic formulas, and operation closure.
- Run synthetic mutation tests before using the actual artifact.

## Task 5 — Actual gates and formal run

- Run portable, liburing, ASan/UBSan, CUDA, focused actual-artifact, Compute Sanitizer, production non-executable, and evidence regression gates.
- Run B-0036 once after all gates pass and atomically publish only complete evidence.
- Record measured values without treating B-0035 and B-0036 as a paired timing comparison.

## Task 6 — Publish

- Update README, ARCHITECTURE, DECISIONS, BENCHMARKS, checklist, context notes, and PROJECT_STATE last.
- Self-review the diff, commit semantic units, publish a public PR, and require pull-request plus post-merge correctness and CodeQL.

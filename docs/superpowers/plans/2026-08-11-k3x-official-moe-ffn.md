# K3X Official MoE FFN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a dependency-closed official Kimi K3 layer-1 MoE FFN slice with real BF16 trunk/router/shared weights and the natural exact Top-16 native-MXFP4 experts, then publish bounded CPU/CUDA parity and traffic evidence.

**Architecture:** K3X v1 gains a fail-closed native BF16 tensor feature without changing record sizes. A two-phase bounded source compiler first materializes always-active tensors and derives two deterministic natural routes, then fetches only the selected expert union into one non-executable K3X fixture. A dedicated portable oracle and CUDA benchmark validate the pinned fixture, execute the exact BF16/MXFP4 boundary, and publish B-0029 without making token or full-layer claims.

**Tech Stack:** Python 3.12, PyTorch reference calculations, safetensors-compatible byte ranges, K3X v1, C++20, CUDA 13.3 `sm_120`, cuBLASLt, pytest, CMake/CTest, Compute Sanitizer, JSON/CSV, SHA-256 and CRC32C.

## Global Constraints

- Keep the K3X major/minor version at 1.0 and keep every fixed record size unchanged.
- Keep `k3x_run` fail-closed for the official MoE fixture through `OPTIONAL_STORAGE_FIXTURE`.
- Preserve BF16 source payload bytes exactly; never expand stored BF16 weights to FP32.
- Bind repository `moonshotai/Kimi-K3` to revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- Use layer 1, all 896 router scores, natural Top-16, exact selected experts, and the real shared expert.
- Use deterministic cases A and B from the accepted design and require their selected expert sets to differ.
- Cap every network read and local copy chunk at 8 MiB.
- Do not download a complete shard or checkpoint and do not provision paid cloud resources.
- Keep all real source bytes and generated K3X artifacts ignored below `artifacts/`; commit only bounded evidence.
- Do not report decode tok/s, prefill tok/s, quality, or physical NVMe traffic from this sublayer benchmark.
- Every new Python, C++, or CUDA source file begins with a one-line Korean role comment.
- Witness RED before production code, run focused GREEN and regressions, self-review, and make one semantic commit per task.

---

### Task 1: Native BF16 K3X v1 contract

**Files:**
- Modify: `converter/k3x_converter/format.py`
- Modify: `converter/k3x_converter/source_manifest.py`
- Modify: `converter/k3x_converter/writer.py`
- Modify: `runtime/include/k3x/format.hpp`
- Modify: `runtime/src/reader.cpp`
- Modify: `tests/python/test_k3x_format.py`
- Modify: `tests/python/test_source_manifest_integrity.py`
- Modify: `tests/python/test_cpp_reader.py`
- Modify: `tests/cpp/test_reader_contract.cpp`

**Interfaces:**
- Produces: `DType.BF16 = 3`, `REQUIRED_BF16_TENSORS = 1 << 0`, `OPTIONAL_OFFICIAL_MOE_FIXTURE = 1 << 1`, and matching C++ constants.
- Produces: source format `k3-official-moe-slice-v1` with artifact kind `official_moe_fixture`.
- Consumes: existing `SourceTensor`, `_TensorPlan`, `Superblock`, `TensorRecord`, and Reader validation paths.

- [x] **Step 1: Add failing Python format and writer tests**

Create a minimal local safetensors shard containing one `BF16` tensor with shape `(2, 3)` and the exact bytes `00 00 80 3f 00 40 40 40 80 40 a0 40`. Require conversion to emit dtype 3, quantization NONE, data length 12, logical length 12, no auxiliary extent, and required feature bit 0. Require `K3XReader.read_tensor_extents()` to return the exact twelve bytes.

Add independent negative tests for a BF16 tensor with an MXFP4 auxiliary, odd byte length, logical length unequal to `2 * product(dimensions)`, missing required bit, and an unknown required bit. Each mutation must fail with the existing stable error family rather than being accepted as FP32.

- [x] **Step 2: Run Python RED**

Run:

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_k3x_format.py \
  tests/python/test_source_manifest_integrity.py -q
```

Expected: the new source manifest is rejected and `DType.BF16` is absent.

- [x] **Step 3: Implement the minimal Python contract**

Add these exact constants and recognition rules.

```python
REQUIRED_BF16_TENSORS = 1 << 0
SUPPORTED_REQUIRED_FEATURES = REQUIRED_BF16_TENSORS
OPTIONAL_STORAGE_FIXTURE = 1 << 0
OPTIONAL_OFFICIAL_MOE_FIXTURE = 1 << 1

class DType(enum.IntEnum):
    FP32 = 1
    UINT8 = 2
    BF16 = 3
```

For plain source tensors, map `F32` to FP32 and `BF16` to BF16. Set `required_features` if any plan is BF16. For NONE tensors compute logical length as `product(dimensions) * 4` for FP32 and `product(dimensions) * 2` for BF16. The official source format sets both optional fixture bits and preserves physical plan order supplied by its manifest; all earlier source formats retain their current order and feature bits.

- [x] **Step 4: Add failing C++ Reader tests**

Generate the minimal BF16 K3X fixture in Python and require the C++ Reader to open it, expose dtype 3, return the exact bytes, and retain required feature bit 0. Mutate the superblock to clear the BF16 bit while repairing its CRC, and mutate it to set bit 63; both must fail before tensor consumption. Add a directory mutation with BF16 plus MXFP4 quantization and require `invalid_directory`.

- [x] **Step 5: Run C++ RED**

Run:

```bash
cmake --build build --target test_reader_contract test_reader
ctest --test-dir build -R 'reader_contract|reader' --output-on-failure
PYTHONPATH=converter:reference K3X_BUILD_DIR=build \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cpp_reader.py -q
```

Expected: C++ rejects required feature bit 0 and dtype 3.

- [x] **Step 6: Implement the C++ Reader contract and run GREEN**

Define `required_bf16_tensors`, `supported_required_features`, and `optional_official_moe_fixture` in `format.hpp`. In `Reader::open`, reject only bits outside `supported_required_features`; accept dtype 3 only when quantization is zero, auxiliary fields are zero, `data_length == logical_length`, and logical length equals twice the checked dimension product. Require every file containing dtype 3 to set the BF16 feature and every file setting the feature to contain at least one BF16 tensor.

Run the commands from Steps 2 and 5, then the complete CPU CTest suite. Expected: all focused tests and CPU CTests pass.

- [x] **Step 7: Self-review and commit**

Confirm FP32/MXFP4 bytes and prior fixture identities are unchanged, no record size/version changed, and unsupported required bits still fail at the superblock. Commit:

```bash
git add converter/k3x_converter/format.py converter/k3x_converter/source_manifest.py \
  converter/k3x_converter/writer.py runtime/include/k3x/format.hpp \
  runtime/src/reader.cpp tests/python/test_k3x_format.py \
  tests/python/test_source_manifest_integrity.py tests/python/test_cpp_reader.py \
  tests/cpp/test_reader_contract.cpp
git commit -m "feat: add native BF16 K3X tensors"
```

---

### Task 2: Two-phase official MoE source planner and materializer

**Files:**
- Create: `converter/k3x_converter/official_moe.py`
- Create: `tests/python/test_official_moe.py`
- Modify: `converter/k3x_converter/official_source.py`
- Modify: `tools/discover_official_kimi_k3.py`
- Modify: `tests/python/test_official_discovery_cli.py`

**Interfaces:**
- Consumes: `OfficialSnapshot`, `OfficialIndex`, `OfficialConfig`, `OfficialShardHeader`, `Transport`, `_fetch_exact_range()`, `inspect_official_shard_header()`, and `convert()`.
- Produces: `OfficialMoePlan`, `OfficialMoeRoutes`, `OfficialMoeMaterializationReport`, `plan_official_moe_slice(...)`, and `materialize_official_moe_slice(...)`.
- Produces: CLI modes `--scope moe-ffn --dry-run` and explicit `--materialize` with `--output-dir`.

- [x] **Step 1: Add failing config, tensor-set, and input tests**

Require `OfficialConfig` to bind these official values: hidden 7168, experts 896, top-k 16, shared experts 2, latent 3584, expert intermediate 3072, SiTU beta 4, linear beta 25, latent norm enabled, and RMS epsilon `1e-5`.

Require `plan_official_moe_slice()` to select the eleven always-active layer-1 tensors listed in the accepted design and no expert payload during dry-run. Require deterministic little-endian FP32 cases:

```python
case_a_prefix[i] = (((17 * i + 3) % 257) - 128) / 1024
case_a_block[i] = (((29 * i + 11) % 251) - 125) / 1024
case_b_prefix[i] = (((31 * i + 7) % 263) - 131) / 1024
case_b_block[i] = (((43 * i + 19) % 269) - 134) / 1024
```

Require all planned ranges to belong to shard 2, remain within the pinned file size, and preserve the accepted physical first-use order.

- [x] **Step 2: Run planner RED**

Run:

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_moe.py \
  tests/python/test_official_source.py \
  tests/python/test_official_discovery_cli.py -q
```

Expected: `official_moe` and the extended config fields do not exist.

- [x] **Step 3: Implement pure planning and routing helpers**

Add frozen dataclasses whose serialized fields contain repository, resolved revision, index/config/shard identities, exact tensor metadata, input SHA-256 values, selected route IDs, canonical contribution values, and selected union. Implement the official route calculation in FP32 with BF16-decoded router weights:

```python
scores = torch.sigmoid(hidden.float() @ gate_weight.float().T)
adjusted = scores + correction_bias.float()
selected = torch.topk(adjusted, 16, sorted=False).indices
contributions = scores[selected]
contributions = contributions / contributions.sum()
canonical = sorted(selected.tolist(), key=lambda e: (-float(adjusted[e]), e))
```

The selected set must equal the official `topk` set, canonical ordering only controls deterministic accumulation, and cases A/B must have different sets. Never substitute or search for preferred experts.

- [ ] **Step 4: Add failing bounded materialization and recovery tests**

Use a fake range transport that rejects any request above 8 MiB and records every request. Simulate interruption after each content object and after source-manifest publication. Require restart to reuse SHA-256-verified objects, discard a corrupt partial object, fetch only missing bytes, and publish no final manifest before all objects are durable.

Require phase 1 to download all always-active tensors, derive both routes, and persist the route manifest. Require phase 2 to fetch exactly the selected expert union and produce one local safetensors-compatible source shard plus `source-manifest.json`. Verify source tensor SHA-256 values and final K3X conversion. No test may contact the network.

- [ ] **Step 5: Implement bounded content-addressed materialization**

Write downloads to `<sha256>.partial`, update the digest while requesting consecutive chunks of at most `8 * 1024 * 1024`, `fsync`, verify length and SHA-256, then atomically replace `<sha256>.blob`. Build the final local shard from verified blobs in physical plan order using the same chunk cap. Publish route and source manifests through temporary files plus atomic replace. Every manifest records converter version, `transport-pinned-ranges`, exact requested ranges, source identities, tensor digests, deterministic input digests, and route union.

The CLI keeps dry-run as its no-payload default. `--materialize` requires an output directory and prints one canonical JSON report; it must not upload, provision, or copy artifacts outside that directory.

- [ ] **Step 6: Run GREEN and regression tests**

Run the Step 2 suite plus:

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_transport.py \
  tests/python/test_safetensors_integrity.py \
  tests/python/test_converter_resume.py \
  tests/python/test_source_manifest_integrity.py -q
```

Expected: all tests pass and fake transport reports no request above 8 MiB.

- [ ] **Step 7: Self-review and commit**

Confirm dry-run downloads zero payload bytes, materialization cannot escape its output directory, every resumed object is rehashed, and no full-shard path exists. Commit:

```bash
git add converter/k3x_converter/official_moe.py \
  converter/k3x_converter/official_source.py tools/discover_official_kimi_k3.py \
  tests/python/test_official_moe.py tests/python/test_official_discovery_cli.py
git commit -m "feat: materialize bounded official MoE slice"
```

---

### Task 3: Portable official BF16/MXFP4 MoE oracle

**Files:**
- Create: `runtime/include/k3x/official_moe.hpp`
- Create: `runtime/src/official_moe.cpp`
- Create: `tests/cpp/test_official_moe.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/python/test_cpp_parity.py`

**Interfaces:**
- Produces: `Bf16WeightView`, `Bf16VectorView`, `OfficialMoeWeights`, `OfficialRoute`, `official_moe_inputs()`, `prepare_official_moe_input(...)`, `route_official_moe(...)`, and `official_moe_cpu(...)`.
- Consumes: exact K3 SiTU, MXFP4 decode, Reader tensor bytes, and Task 2 route manifest.

- [ ] **Step 1: Add failing scalar BF16 and official-boundary tests**

Test exact BF16 word-to-FP32 decoding for zero, signed zero, finite normals, infinity, and NaN bit patterns. With tiny literal matrices, require Attention Residual, RMSNorm, router sigmoid/correction selection, contribution renormalization, latent down, two MXFP4 experts, routed norm/up, shared SiTU-GLU/down, routed/shared add, and final prefix add to match independent Python/PyTorch values at every named boundary.

Mutate dimensions, route count, duplicate expert IDs, non-finite contribution, contribution sum, missing tensor, and BF16 byte alignment independently; require failure before any output is published.

- [ ] **Step 2: Run portable RED**

Run:

```bash
cmake -S . -B build -G Ninja
cmake --build build --target test_official_moe
```

Expected: CMake or compilation fails because the official MoE files and target do not exist.

- [ ] **Step 3: Implement minimal portable graph**

Represent BF16 weights as `std::span<const std::uint16_t>` plus rows/columns and decode explicitly with a 16-bit left shift into the FP32 bit pattern. Apply the accepted output boundaries exactly: BF16-round Attention Residual and normalization outputs, BF16-round each BF16 Linear output, BF16-round each expert output, accumulate contributions in FP32 then BF16-round the mixed latent, and BF16-round routed/shared and final prefix additions.

Use exact official dimensions only in the pinned identity validator; keep the pure tiny oracle dimension-driven so tests remain bounded. Return `Result<T>` and leave production `ModelSession` untouched.

- [ ] **Step 4: Run GREEN and Python parity**

Run:

```bash
cmake --build build --target test_official_moe
ctest --test-dir build -R official_moe --output-on-failure
PYTHONPATH=converter:reference K3X_BUILD_DIR=build \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cpp_parity.py -q
```

Expected: every intermediate boundary and final vector passes; malformed inputs fail closed.

- [ ] **Step 5: Run CPU regression and commit**

Run complete CPU CTest. Confirm the helper has no network, filesystem, CUDA, global state, or production dispatch. Commit:

```bash
git add CMakeLists.txt runtime/include/k3x/official_moe.hpp \
  runtime/src/official_moe.cpp tests/cpp/test_official_moe.cpp \
  tests/python/test_cpp_parity.py
git commit -m "feat: add official MoE portable oracle"
```

---

### Task 4: Native BF16 CUDA official MoE boundary

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/cuda/resident_weights.cuh`
- Modify: `runtime/cuda/resident_weights.cu`
- Modify: `runtime/cuda/moe_layer.cuh`
- Modify: `runtime/cuda/moe_layer.cu`
- Create: `tests/cuda/test_cuda_official_moe.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `Bf16MlpView`, `OfficialMoeFfnView`, `OfficialMoeFfnResult`, and `ComputeBackend::official_mxfp4_moe_ffn(...)`.
- Consumes: Task 3 normalized BF16 hidden input, prefix residual, canonical routes, BF16 K3X views, and existing `Mxfp4MlpView` experts.

- [ ] **Step 1: Add failing synthetic CUDA boundary tests**

Create tiny BF16 literal routed/shared weights and two literal MXFP4 experts. Compare transient and resident CUDA results against Task 3 CPU oracle. Require one final output vector, finite values, maximum absolute error at most `2e-2`, exact selected expert order, and no mutation of caller buffers.

For resident mode, snapshot stats around a second call and require zero weight H2D, all BF16 and MXFP4 tensors resident, and unchanged output. Require invalid BF16 byte count, aliasing tensor IDs, duplicate experts, wrong contribution count, and insufficient resident capacity to fail before execution.

- [ ] **Step 2: Run CUDA RED**

Run:

```bash
cmake -S . -B build-cuda -G "Unix Makefiles" -DK3X_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target test_cuda_official_moe
```

Expected: target or API is absent.

- [ ] **Step 3: Implement native BF16 views and residency**

Add views over raw `std::uint16_t` BF16 storage. Admission copies those bytes once and keys residency by tensor ID, byte count, dimensions, and immutable digest/validation state. Do not construct an intermediate host `std::vector<float>` or rebuild BF16 host storage on calls.

The dedicated method accepts normalized hidden and prefix residual prepared by Task 3, executes routed down, selected expert FFNs, weighted mix, routed norm/up, shared gate/up/SiTU/down, routed/shared add, and final prefix add on one stream, then performs exactly one final D2H. Retain CPU Attention Residual, postnorm, and router outside the CUDA method.

- [ ] **Step 4: Run CUDA GREEN and traffic assertions**

Run:

```bash
cmake --build build-cuda --target test_cuda_official_moe
ctest --test-dir build-cuda -R cuda_official_moe --output-on-failure
```

Expected: transient and resident outputs pass, the second resident call records zero weight H2D, and one D2H vector is recorded per call.

- [ ] **Step 5: Run CUDA regression and sanitizer**

Run complete CUDA CTest, then:

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./build-cuda/test_cuda_official_moe
```

Expected: all CUDA CTests pass and Compute Sanitizer reports `ERROR SUMMARY: 0 errors`.

- [ ] **Step 6: Self-review and commit**

Confirm the existing FP32 `ResidentMoeLayerView` behavior and production defaults are unchanged, oracle-only allocations are destroyed before measured execution, and reported peak residency covers every live device allocation. Commit:

```bash
git add CMakeLists.txt runtime/include/k3x/backend.hpp \
  runtime/cuda/backend_cuda.cu runtime/cuda/resident_weights.cuh \
  runtime/cuda/resident_weights.cu runtime/cuda/moe_layer.cuh \
  runtime/cuda/moe_layer.cu tests/cuda/test_cuda_official_moe.cu
git commit -m "feat: execute official BF16 MoE on CUDA"
```

---

### Task 5: Pinned official fixture harness

**Files:**
- Create: `runtime/src/cuda_official_moe_bench.cpp`
- Create: `tests/python/test_cuda_official_moe.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: Task 2 manifest/artifact, Task 3 portable oracle, Task 4 CUDA boundary, and Reader checksum mode.
- Produces: `k3x_cuda_official_moe_bench` with cases `a`, `b`, and `alternating`, and weight modes `transient` and `resident`.

- [ ] **Step 1: Add failing CLI and identity tests**

Require paired options `--model`, `--manifest`, `--case`, `--weight-mode`, `--warmup`, and `--iterations`. Reject missing paths, unknown cases/modes, zero iterations, trailing arguments, a generic storage fixture, and every independent mutation of repository, revision, K3X root, tensor identities/shapes/dtypes, input digests, route IDs, route contributions, and selected union before backend construction.

- [ ] **Step 2: Run harness RED**

Run:

```bash
cmake --build build-cuda --target k3x_cuda_official_moe_bench
PYTHONPATH=converter:reference K3X_BUILD_DIR=build-cuda \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_official_moe.py -q
```

Expected: source/target is absent.

- [ ] **Step 3: Implement strict load, oracle, and execution flow**

Open the artifact with full checksums, parse the canonical manifest with duplicate-key and non-finite-number rejection, validate every fixed identity, and recompute both CPU routes from artifact bytes. Require exact route-set equality and contribution error at most `1e-6` before CUDA construction.

Compute and retain the portable CPU output, destroy temporary oracle-only weight ownership that is not needed by the selected backend, then run the selected CUDA case. `alternating` executes A then B in each iteration using one resident table and reports their exact union behavior.

- [ ] **Step 4: Emit and test the canonical JSON schema**

Emit identity and scope fields, case/mode/warmup/iterations, input and output elements, exact route IDs, selected union, source/K3X bytes, cold and measured latency, p05/median/p95, Attention Residual/router/orchestration/kernel times, BF16/MXFP4/activation H2D, D2H, resident bytes, cache hits/misses/bypasses, allocation/synchronization counts, peak VRAM, maximum absolute error, and finite status.

Set `token_semantics=false`, `full_transformer_layer=false`, `quality_measured=false`, and omit tok/s, quality score, and physical NVMe fields. Enforce transient traffic formulas and resident warm weight H2D zero inside the C++ process.

- [ ] **Step 5: Run synthetic GREEN and actual ignored-artifact smoke**

Run the Step 2 tests. If the actual M28 artifact exists, run cases A and B once in transient mode and alternating once in resident mode with zero warmups and one iteration. Expected: all identities, routes, parity, finite values, and traffic invariants pass. If it does not exist yet, record the smoke as pending Task 7 rather than weakening tests.

- [ ] **Step 6: Self-review and commit**

Confirm identity rejection occurs before CUDA construction, no real artifact is copied or tracked, and the harness cannot call production generation. Commit:

```bash
git add CMakeLists.txt runtime/src/cuda_official_moe_bench.cpp \
  tests/python/test_cuda_official_moe.py
git commit -m "feat: validate pinned official MoE fixture"
```

---

### Task 6: B-0029 runner and strict evidence verifier

**Files:**
- Create: `tools/ablate_official_moe.py`
- Create: `tests/python/test_official_moe_ablation.py`

**Interfaces:**
- Consumes: Task 5 canonical JSON.
- Produces: three raw JSON rows, `summary.json`, LF-only `summary.csv`, `run_ablation(...)`, and `verify_summary(...)`.

- [ ] **Step 1: Add failing controlled-runner tests**

Use a fake executable that emits schema-complete rows. Require fixed order `a-transient`, `a-resident`, `alternating-resident`, exact forwarding of 3 warmups/20 iterations, compact sorted LF JSON, LF-only CSV, raw/runner/artifact/manifest digests, canonical aggregate, and CSV parity.

Add independent mutations for forbidden tok/s/quality/NVMe keys, wrong identity, route mismatch, contribution drift, resident warm H2D, transient formula, D2H formula, parity above `2e-2`, non-finite value, raw digest, runner digest, aggregate, CSV digest, and case order.

- [ ] **Step 2: Run runner RED**

Run:

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_moe_ablation.py -q
```

Expected: import fails because the tool does not exist.

- [ ] **Step 3: Implement the fixed matrix and verifier**

Use exactly:

```python
CASES = (
    ("a-transient", "a", "transient"),
    ("a-resident", "a", "resident"),
    ("alternating-resident", "alternating", "resident"),
)
```

Parse exactly one JSON object per process. Validate the complete schema and formulas before writing any result. Strict mode rehashes the actual artifact, manifest, and runner, fixes 3/20, and verifies every raw file plus summary/CSV bytes. Test mode may skip only real-file identities and 3/20; it may not skip schema, formulas, parity, or forbidden-field checks.

- [ ] **Step 4: Run GREEN and commit**

Run the Step 2 suite. Confirm the tool never reruns or ranks timings and never copies source/artifact bytes. Commit:

```bash
git add tools/ablate_official_moe.py \
  tests/python/test_official_moe_ablation.py
git commit -m "bench: add official MoE ablation"
```

---

### Task 7: Actual bounded materialization, B-0029, and public integration

**Files:**
- Create ignored: `artifacts/m28-official-moe/live/**`
- Create: `results/b0029-official-moe-wsl/a-transient.json`
- Create: `results/b0029-official-moe-wsl/a-resident.json`
- Create: `results/b0029-official-moe-wsl/alternating-resident.json`
- Create: `results/b0029-official-moe-wsl/summary.json`
- Create: `results/b0029-official-moe-wsl/summary.csv`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `PROJECT_STATE.md` last

**Interfaces:**
- Consumes: Tasks 1-6 and the pinned public source.
- Produces: one ignored official MoE fixture, canonical B-0029 evidence, synchronized TITAN Ledger, and a public GitHub integration.

- [ ] **Step 1: Run official dry-run and inspect the bounded request plan**

Run the `moe-ffn` dry-run at the pinned revision. Require zero payload bytes, exact eleven always-active tensors, exact shard identity, estimated always-on bytes 379,900,416, and maximum possible two-case bytes no greater than 941 MiB. Stop if any identity, tensor, dimension, dtype, or shard placement differs from the design.

- [ ] **Step 2: Materialize once and verify every identity**

Run explicit materialization into `artifacts/m28-official-moe/live`. Allow only the planned capped ranges. Verify content objects, route sets differ, union size is 16 through 32, final tensor set equals always-active plus union experts, both optional fixture bits are set, BF16 required bit is set, all CRC/root/source digests pass, and `k3x_run` returns `NON_EXECUTABLE_ARTIFACT`.

- [ ] **Step 3: Run actual correctness and sanitizer gates before timing**

Run the Task 5 actual-artifact focused tests, direct A/B/alternating smoke, complete CUDA CTest, and Compute Sanitizer on one resident alternating iteration. Require exact routes, contribution error at most `1e-6`, final maximum error at most `2e-2`, all finite output, zero sanitizer errors, and zero resident warm weight H2D.

- [ ] **Step 4: Run B-0029 exactly once**

Invoke the Task 6 tool with the actual artifact/manifest/runner, output directory `results/b0029-official-moe-wsl`, 3 warmups, and 20 iterations. Do not rerun to select timing. Rerun only after documenting a correctness/evidence defect and replace every affected raw/summary file together.

- [ ] **Step 5: Independently cross-check evidence**

Run the strict verifier, recompute every tracked file digest from staged Git blob bytes, compare raw rows to summary/CSV, validate formulas and forbidden-field absence, and add a committed-evidence regression binding artifact, manifest, runner, raw, aggregate, and CSV digests.

- [ ] **Step 6: Run the complete verification matrix**

Run CPU CTest/pytest, liburing/direct CTest/pytest, ASan/UBSan CTest, CUDA CTest/pytest, actual-artifact focused tests, committed B-0029 tests, and Compute Sanitizer. Record fresh pass/skip counts and exact environment; do not reuse M27 counts.

- [ ] **Step 7: Update README and every TITAN Ledger document**

Record only measured sublayer latency, traffic, residency, routes, parity, and scope. Update architecture status and D-055 or later decision evidence. Mark token metrics, quality, physical NVMe, and full-layer behavior unmeasured. Update `PROJECT_STATE.md` after every other document and identify the next measured bottleneck.

- [ ] **Step 8: Final review, semantic commit, and public integration**

Run `git diff --check`, inspect every changed line against the accepted design, verify no real bytes are tracked, and fix only Critical/Important findings in one correction cycle. Commit evidence/docs, push, open a public PR, wait for correctness and CodeQL, rebase-merge only when clean, and verify post-merge correctness and CodeQL before beginning M29.

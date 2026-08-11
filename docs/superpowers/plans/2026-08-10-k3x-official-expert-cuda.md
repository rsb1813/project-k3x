# K3X Official Expert CUDA Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove exact RTX 5080 execution of the pinned official Kimi K3 layer-1 expert-0 FFN against the portable CPU oracle and publish strict transient/resident B-0028 evidence.

**Architecture:** A pure runtime identity helper owns the fixed B-0027 contract. A separate CUDA-only benchmark validates that identity before backend construction, computes a CPU oracle, and measures transient or exact-capacity resident execution. A Python runner/verifier publishes two bounded rows without committing the real K3X artifact or changing production generation.

**Tech Stack:** C++20, CUDA 13.3 `sm_120`, existing K3X Reader and CPU/CUDA backends, Python 3.12, pytest, CMake/CTest, JSON/CSV, SHA-256.

## Global Constraints

- Keep `k3x_run` and its `NON_EXECUTABLE_ARTIFACT` guard unchanged.
- Do not download another official byte range, complete shard, or full checkpoint.
- Do not provision or invoke paid cloud resources.
- Official identity is fixed in code and is never supplied by a caller-selected label.
- Use full Reader checksum verification before identity validation or backend construction.
- Treat the exploratory 7.212 ms smoke as non-benchmark evidence; B-0028 must be freshly measured.
- Report one expert FFN only. Set token, routing, and full-MoE-layer semantics to false.
- Every new C++ or Python source file starts with a one-line Korean role comment.
- Use witnessed RED/GREEN cycles and one semantic commit per task.

---

### Task 1: Pinned official expert identity contract

**Files:**
- Create: `runtime/include/k3x/official_expert.hpp`
- Create: `runtime/src/official_expert.cpp`
- Create: `tests/cpp/test_official_expert.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `k3x::OfficialExpertObservation`, `k3x::OfficialExpertIdentity`, `k3x::official_kimi_k3_expert()`, and `k3x::verify_official_kimi_k3_expert(const OfficialExpertObservation&)`.
- Consumes: `optional_storage_fixture`, `Result<T>`, and `ErrorCode::invalid_mxfp4` from the existing runtime.

- [x] **Step 1: Add the failing identity test and CMake target**

Define the wished-for observation with exact literals and verify success plus one mutation for every field.

```cpp
const auto identity = k3x::official_kimi_k3_expert();
k3x::OfficialExpertObservation observation{
    identity.k3x_root_sha256,
    identity.ordered_sha256,
    k3x::optional_storage_fixture,
    1,
    0,
    17'547'264,
    {{{3072, 3584}, {3072, 3584}, {3584, 3072}}},
};
if (!k3x::verify_official_kimi_k3_expert(observation)) return 1;
auto bad = observation;
bad.k3x_root_sha256[0] ^= std::byte{1};
if (k3x::verify_official_kimi_k3_expert(bad)) return 2;
```

Repeat independent mutations for ordered digest, optional features, layer, expert, payload bytes, and each shape. Add `test_official_expert` to CTest and add `runtime/src/official_expert.cpp` to `k3x_runtime` before the file exists.

- [x] **Step 2: Run RED**

Run:

```bash
cmake -S . -B build -G Ninja
cmake --build build --target test_official_expert
```

Expected: compilation fails because `k3x/official_expert.hpp` and the declared symbols do not exist.

- [x] **Step 3: Implement the minimal pure contract**

Use fixed-size values only.

```cpp
struct OfficialExpertShape {
    std::uint64_t rows{};
    std::uint64_t columns{};
};

struct OfficialExpertIdentity {
    std::array<std::byte, 32> k3x_root_sha256{};
    std::array<std::byte, 32> ordered_sha256{};
    std::uint64_t optional_features{};
    std::uint32_t layer_id{};
    std::uint32_t expert_id{};
    std::uint64_t payload_bytes{};
    std::array<OfficialExpertShape, 3> shapes{};
};

using OfficialExpertObservation = OfficialExpertIdentity;

const OfficialExpertIdentity& official_kimi_k3_expert();
Result<OfficialExpertIdentity> verify_official_kimi_k3_expert(
    const OfficialExpertObservation& observation);
```

The implementation compares every field and returns `invalid_mxfp4` with `official Kimi K3 expert identity mismatch` on any difference. Parse the two literal 64-hex digests once into byte arrays; reject no caller input because the API accepts bytes, not text.

- [x] **Step 4: Run GREEN and the CPU CTest suite**

Run:

```bash
cmake --build build --target test_official_expert
ctest --test-dir build --output-on-failure
```

Expected: `official_expert` passes and the complete CPU CTest suite has zero failures.

- [x] **Step 5: Self-review and commit**

Confirm the helper performs no I/O, allocation, environment lookup, or CUDA work. Then commit:

```bash
git add CMakeLists.txt runtime/include/k3x/official_expert.hpp runtime/src/official_expert.cpp tests/cpp/test_official_expert.cpp
git commit -m "feat: bind official expert identity"
```

---

### Task 2: Dedicated official expert CUDA harness

**Files:**
- Create: `runtime/src/cuda_official_expert_bench.cpp`
- Create: `tests/python/test_cuda_official_expert.py`
- Modify: `CMakeLists.txt`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_ffn.cu`

**Interfaces:**
- Consumes: Task 1 identity API, `Reader`, `load_storage_expert`, `make_cpu_backend`, `make_cuda_backend`, and `mxfp4_situ_mlp_group`.
- Produces: executable `k3x_cuda_official_expert_bench` and one canonical JSON object on stdout.

- [x] **Step 1: Add failing CLI and pre-CUDA identity tests**

Add the CUDA executable target before its source exists. In Python, require `K3X_BUILD_DIR=build-cuda` and test these exact failures.

```python
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((), "model path is required"),
        (("--weight-mode", "other"), "unknown weight mode: other"),
        (("--iterations", "0"), "iterations must be positive"),
    ],
)
def test_official_expert_bench_rejects_invalid_arguments(arguments, message):
    result = subprocess.run([str(runner), *arguments], capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stderr.strip() == message
```

Create a normal synthetic storage fixture with the existing writer, invoke the bench with it, and require exit 4 plus `INVALID_MXFP4: official Kimi K3 expert identity mismatch`. This proves rejection occurs before the CUDA backend is needed.

- [x] **Step 2: Run RED**

Run:

```bash
cmake -S . -B build-cuda -G "Unix Makefiles" -DK3X_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target k3x_cuda_official_expert_bench
PYTHONPATH=converter:reference K3X_BUILD_DIR=build-cuda /home/jolib/.venvs/k3x-m1/bin/python -m pytest tests/python/test_cuda_official_expert.py -q
```

Expected: build fails because `cuda_official_expert_bench.cpp` does not exist.

- [x] **Step 3: Implement parsing and fail-before-backend validation**

Support only paired options `--model`, `--weight-mode`, `--warmup`, and `--iterations`. Open with `VerifyMode::checksums`, load layer 1 expert 0, construct `OfficialExpertObservation` from the Reader superblock and loaded digest, validate it, and only then construct backends.

Use the gate/up/down extents in this exact mapping.

```cpp
const k3x::Mxfp4MlpView expert{
    {101, extents[0], extents[1], 3072, 3584, 32},
    {102, extents[2], extents[3], 3072, 3584, 32},
    {103, extents[4], extents[5], 3584, 3072, 32},
};
```

- [x] **Step 4: Implement CPU oracle and CUDA correctness flow**

Build the deterministic 3,584-element input, time one CPU `mxfp4_situ_mlp_group` call, and retain its only output. Construct CUDA options with reused allocation, FFN block, synchronous transfer, no fusion, and the selected weight mode. Resident capacity is exactly 17,547,264; transient capacity is zero.

Record runtime/profiler snapshots around the cold call and around measured calls separately. After every CUDA call, require output length 3,584, all finite values, and maximum absolute error at most `1.0e-6F`.

- [x] **Step 5: Emit the exact JSON schema**

Include these keys and no token/quality fields.

```text
artifact_kind, repository, resolved_revision, token_semantics,
routing_semantics, full_moe_layer, layer_id, expert_id, weight_mode,
k3x_root_sha256, ordered_sha256, expert_payload_bytes, input_elements,
output_elements, warmup, iterations, cpu_oracle_nanoseconds,
cold_latency_nanoseconds, cold_kernel_nanoseconds, cold_weight_h2d_bytes,
cold_activation_h2d_bytes, cold_device_to_host_bytes,
latency_nanoseconds_median, latency_nanoseconds_p05,
latency_nanoseconds_p95, kernel_nanoseconds, weight_h2d_bytes,
activation_h2d_bytes, device_to_host_bytes, device_allocation_count,
stream_synchronization_count, weight_cache_hits, weight_cache_misses,
weight_cache_bypasses, resident_weight_bytes, peak_resident_weight_bytes,
peak_vram_bytes, maximum_absolute_error, all_finite
```

Transient measured weight H2D must equal `iterations * 17_547_264`. Resident cold H2D must equal 17,547,264, measured H2D must be zero, resident bytes must equal 17,547,264, and bypasses must be zero; otherwise exit 4.

- [x] **Step 6: Run GREEN against synthetic rejection and the ignored official artifact**

Run:

```bash
cmake --build build-cuda --target k3x_cuda_official_expert_bench
PYTHONPATH=converter:reference K3X_BUILD_DIR=build-cuda /home/jolib/.venvs/k3x-m1/bin/python -m pytest tests/python/test_cuda_official_expert.py -q
./build-cuda/k3x_cuda_official_expert_bench --model artifacts/m26-official/live/expert-l1-e0.k3x --weight-mode transient --warmup 0 --iterations 1
./build-cuda/k3x_cuda_official_expert_bench --model artifacts/m26-official/live/expert-l1-e0.k3x --weight-mode resident --warmup 0 --iterations 1
```

Expected: tests pass; both real runs return zero, report `maximum_absolute_error <= 1e-6`, and satisfy their traffic invariants.

- [x] **Step 7: Run CUDA regression and sanitizer gates**

Run:

```bash
ctest --test-dir build-cuda --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=99 ./build-cuda/k3x_cuda_official_expert_bench --model artifacts/m26-official/live/expert-l1-e0.k3x --weight-mode resident --warmup 0 --iterations 1
```

Expected: all CUDA CTests pass and Compute Sanitizer reports `ERROR SUMMARY: 0 errors`.

- [x] **Step 8: Self-review and commit**

Confirm `k3x_run`, `model.cpp`, format files, and production defaults are untouched. Then commit:

```bash
git add CMakeLists.txt runtime/src/cuda_official_expert_bench.cpp tests/python/test_cuda_official_expert.py
git commit -m "feat: execute pinned official expert on CUDA"
```

---

### Task 3: B-0028 runner and strict verifier

**Files:**
- Create: `tools/ablate_official_expert_cuda.py`
- Create: `tests/python/test_official_expert_cuda_ablation.py`

**Interfaces:**
- Consumes: Task 2 canonical JSON executable output.
- Produces: `run_ablation(...)`, `verify_summary(...)`, raw `transient.json`, raw `resident.json`, `summary.json`, and LF-only `summary.csv`.

- [x] **Step 1: Write RED tests with a controlled fake runner**

The fake runner emits one schema-complete row selected by `--weight-mode`. Test exact case order, warmup/iteration forwarding, raw JSON digests, LF-only CSV, canonical aggregate digest, and `strict_official=False` verification. Add independent mutations for wrong identity, forbidden `decode_tok_s`, resident measured H2D, transient H2D formula, parity above `1e-6`, raw digest, CSV parity, and case order.

- [x] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest tests/python/test_official_expert_cuda_ablation.py -q
```

Expected: import fails because `tools.ablate_official_expert_cuda` does not exist.

- [x] **Step 3: Implement the two-case runner**

Use this fixed matrix.

```python
CASES = (("transient", "transient"), ("resident", "resident"))
```

Invoke the runner with `--model`, `--weight-mode`, `--warmup`, and `--iterations`. Parse exactly one JSON object, validate its full schema and invariants, write each raw row with sorted compact JSON plus one LF, then build summary records containing `name` and `raw_json_sha256`.

The strict summary binds artifact file SHA-256 `e08293cd854ed11913bd8f1bc3a51d1eb577202fd5fd9b5b7e3c96ef1bccecc7`, B-0027 summary JSON SHA-256 `57ebd9d85ed3ae55a4e2ab01f023bc451faf02cd7b6e69f478d11e3ea73e982a`, all fixed C++ identities, warmup 3, iterations 20, raw and runner digests, aggregate digest, CSV digest/parity, and forbidden-field absence. `strict_official=False` skips only the real artifact and fixed 3/20 gates for synthetic unit tests.

- [x] **Step 4: Run GREEN and mutation tests**

Run:

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest tests/python/test_official_expert_cuda_ablation.py -q
```

Expected: all runner, parity, digest, invariant, and mutation tests pass.

- [x] **Step 5: Self-review and commit**

Confirm the tool never copies the artifact and never emits token, quality, or physical-NVMe fields. Then commit:

```bash
git add tools/ablate_official_expert_cuda.py tests/python/test_official_expert_cuda_ablation.py
git commit -m "bench: add official expert CUDA ablation"
```

---

### Task 4: Fresh B-0028 measurement and evidence publication

**Files:**
- Create: `results/b0028-official-expert-cuda-wsl/transient.json`
- Create: `results/b0028-official-expert-cuda-wsl/resident.json`
- Create: `results/b0028-official-expert-cuda-wsl/summary.json`
- Create: `results/b0028-official-expert-cuda-wsl/summary.csv`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md` last
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: Tasks 1-3 and the ignored B-0027 K3X artifact.
- Produces: canonical B-0028 evidence and a measured M28 boundary decision.

- [x] **Step 1: Build the exact native CUDA target and verify artifact identity**

Run:

```bash
cmake --build build-cuda --target k3x_cuda_official_expert_bench
sha256sum artifacts/m26-official/live/expert-l1-e0.k3x
```

Expected artifact SHA-256: `e08293cd854ed11913bd8f1bc3a51d1eb577202fd5fd9b5b7e3c96ef1bccecc7`.

- [x] **Step 2: Run B-0028 once**

Run:

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python tools/ablate_official_expert_cuda.py \
  --artifact artifacts/m26-official/live/expert-l1-e0.k3x \
  --runner build-cuda/k3x_cuda_official_expert_bench \
  --output-dir results/b0028-official-expert-cuda-wsl \
  --warmup 3 --iterations 20
```

Do not rerun to select a preferred timing. Rerun only after a documented correctness or evidence defect, and replace every affected raw/summary artifact together.

- [x] **Step 3: Independently verify staged evidence**

Run the strict verifier against summary JSON/CSV, recompute every tracked file digest from Git blobs, and compare raw rows with the summary. Require both rows to satisfy parity and traffic invariants. Record measured values exactly; do not infer tok/s.

- [x] **Step 4: Run the complete verification matrix**

Run CPU CTest/pytest, liburing/direct CTest/pytest, ASan/UBSan CTest, CUDA CTest/pytest, the real transient/resident tests, and Compute Sanitizer. Capture exact pass/skip counts and do not reuse M26 counts.

- [x] **Step 5: Update README and TITAN Ledger**

Record B-0028 date, commit, hardware, WSL2 environment, official revision, one-expert scope, transient/resident cold and warm latency, CPU error, H2D, D2H, VRAM, cache residency, kernel time, enabled options, and every unmeasured charter metric. Add D-053 benchmark evidence and decide only the next M28 slice. Update `PROJECT_STATE.md` after every other document.

- [x] **Step 6: Final review and semantic evidence commit**

Review the complete diff for Critical/Important correctness, evidence, identity, timing, and claim-boundary issues. Apply at most one focused correction/re-review cycle unless a verified Critical remains. Commit evidence and synchronized documents with a single benchmark-scoped message.

- [x] **Step 7: Publish and integrate**

Push the branch, open a public PR, wait for correctness and CodeQL, rebase-merge only when clean, verify post-merge `main` correctness and CodeQL, and reconcile any remaining publication-pending documentation before starting M28.

Completed through PR #46 at public integration head `ec08b827`. Branch correctness `31455570571`, pull-request correctness `31455597581`, pull-request CodeQL `31455597565`, post-merge correctness `31455776634`, and post-merge CodeQL `31455776673` all passed.

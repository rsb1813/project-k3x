# K3X Persistent Official Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the complete validated official K3X token graph in one opt-in Python process while retaining the subprocess reference path.

**Architecture:** Existing official graph scripts expose callable entrypoints. A thin driver dispatches layer 0, KDA, MLA, and head stages either directly or through the unchanged subprocess reference boundary, validates publications, and emits timing evidence.

**Tech Stack:** Python 3.12, PyTorch CUDA, K3X Python Reader, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-k3x-persistent-official-runtime-design.md`

## Global Constraints

- Preserve natural Top-16 routing, exact cold-expert loading, and the sealed K3X identity checks.
- Keep subprocess execution available as the reference mode.
- Do not claim steady decode TPS from one first-token execution.
- Do not download checkpoint payloads or provision paid resources.
- Add no new quantization or quality loss in this milestone.

---

### Task 1: Callable official graph stages

**Files:**
- Modify: `tools/run_official_layer0.py`
- Modify: `tools/run_official_layer1.py`
- Modify: `tools/run_official_layer3.py`
- Modify: `tools/run_official_head.py`
- Test: `tests/python/test_official_persistent_runner.py`

**Interfaces:**
- Produces: `run(args: argparse.Namespace) -> int` in every official stage module.
- Preserves: each module's command-line `main() -> int` behavior.

- [ ] Write an import-level test requiring callable `run` entrypoints.
- [ ] Run the focused test and observe failure because the entrypoints do not exist.
- [ ] Extract only argument parsing from each current `main`, leaving its execution body unchanged in `run`.
- [ ] Run the focused test and existing official CLI tests.
- [ ] Commit the callable-stage change.

### Task 2: In-process remaining-layer dispatcher

**Files:**
- Modify: `tools/run_official_remaining.py`
- Test: `tests/python/test_official_persistent_runner.py`

**Interfaces:**
- Consumes: callable KDA and MLA `run` entrypoints.
- Produces: `--execution-mode subprocess|in-process`, defaulting to `subprocess`.

- [ ] Write a test proving in-process dispatch invokes the selected callable and does not invoke a child process.
- [ ] Run the focused test and observe the missing-mode failure.
- [ ] Implement the minimum explicit dispatch branch and publication validation.
- [ ] Run the focused tests and CLI help check.
- [ ] Commit the dispatcher change.

### Task 3: One-process token driver and measurement

**Files:**
- Create: `tools/run_official_token.py`
- Test: `tests/python/test_official_persistent_runner.py`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`

**Interfaces:**
- Consumes: the four callable official stages and authenticated prefix state.
- Produces: one command for resumable first-token execution plus stage and total wall timing.

- [ ] Write a bounded fake-stage test for resume, stage ordering, and publication validation.
- [ ] Run it and observe failure because the driver is absent.
- [ ] Implement the driver without duplicating graph math.
- [ ] Run focused regressions and Python compilation.
- [ ] Run the real sealed K3X graph in-process, compare token and digests with B-0046, and record only measured fields.
- [ ] Update TITAN Ledger documents, with `PROJECT_STATE.md` last.
- [ ] Run `git diff --check`, commit evidence, and push the branch.

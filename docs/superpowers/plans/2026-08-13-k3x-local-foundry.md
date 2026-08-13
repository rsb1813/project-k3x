# K3X Local Foundry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline. Subagents are disabled by user instruction. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the checksum-bound local pipeline that manufactures a runnable 1.28 TB K3X checkpoint with authenticated high-performance Xet shard staging.

**Architecture:** A deterministic Conductor ledger coordinates two bounded HDD shard slots and direct NVMe K3X extent publication. A tested 3-bit expert codec and precision recipe are required before any full official shard is downloaded.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face Hub with `hf_xet`, K3X v1, Windows PowerShell orchestration, Linux-native converter/runtime boundaries.

**Spec:** `docs/superpowers/specs/2026-08-13-k3x-local-foundry-design.md`

## Global Constraints

- Destination reserve is 200 GiB and staging reserve is 100 GiB.
- The output budget is 1,280,000,000,000 bytes.
- Full official downloads remain disabled until both synthetic gates pass.
- No subagents and no repeated broad regression runs.

---

### Task 1: Disk-safe manifest and IMMORTAL ledger

**Files:** `converter/k3x_converter/local_foundry.py`, `tools/run_local_foundry.py`, and `tests/python/test_local_foundry.py`.

- [ ] Write one focused failing test for the output budget, two shard slots, reserve rejection, canonical unit identity, and resume validation.
- [ ] Run only that test and confirm the missing module failure.
- [ ] Implement the minimal manifest, disk guard, atomic ledger, and `--dry-run` CLI.
- [ ] Re-run the focused test and commit the passing boundary.

### Task 2: Deterministic 3-bit expert codec

**Files:** `reference/k3x_ref/quant3.py`, `tests/python/test_local_foundry.py`, and `K3X_FORMAT.md`.

- [ ] Add failing synthetic round-trip, byte-budget, determinism, corruption, and sensitive-tensor passthrough assertions.
- [ ] Confirm the focused test fails because the codec is absent.
- [ ] Implement the smallest reference codec and explicit K3X quantization metadata contract.
- [ ] Re-run the focused test and commit the codec boundary.

### Task 3: Xet staging and synthetic integration gate

**Files:** `converter/k3x_converter/local_foundry.py`, `tools/run_local_foundry.py`, `tests/python/test_local_foundry.py`, and the TITAN Ledger documents.

- [ ] Add failing assertions for authenticated Xet command construction, verified-source deletion eligibility, and interrupted-unit resume.
- [ ] Implement staging without exposing the token and run the focused test once.
- [ ] Convert the synthetic model through the local pipeline and compare layer outputs and greedy tokens in one integration gate.
- [ ] Update the ledger, record measured Xet throughput, and commit before enabling the official manifest.

### Task 4: Official manufacture launch gate

**Files:** `tools/run_local_foundry.py` and `PROJECT_STATE.md`.

- [ ] Recheck live disk reserves, authenticated account, official revision, and output-byte budget.
- [ ] Enable the official launch only when every gate is recorded in the ledger.
- [ ] Start the bounded two-slot pipeline, publish progress per shard, and stop on the first correctness or resource violation.

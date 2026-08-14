# K3X Shared Official Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Own immutable official metadata and sealed-set directory state once per persistent token process.

**Architecture:** A small context object provides authenticated metadata plus lazy header/store caches. In-process stages consume it; standalone subprocess stages remain unchanged.

**Tech Stack:** Python 3.12, PyTorch CUDA, pytest, K3X Python Reader.

**Spec:** docs/superpowers/specs/2026-08-14-k3x-shared-official-context-design.md

## Global Constraints

- Preserve B-0048 token, route, state, logit, and hidden digests.
- Keep standalone and subprocess behavior unchanged.
- Do not retain decoded model tensors in this milestone.
- Do not claim decode TPS from a first-token run.

### Task 1: Context ownership

**Files:**
- Create: tools/official_runtime_context.py
- Modify: tools/run_official_token.py
- Test: tests/python/test_official_persistent_runner.py

- [ ] Write a failing test proving one context instance reaches every in-process stage.
- [ ] Implement one authenticated context creation and driver propagation.
- [ ] Write a failing test proving repeated shard store lookup opens once.
- [ ] Implement lazy header and store caches.
- [ ] Run focused tests and commit.

### Task 2: Stage integration and measurement

**Files:**
- Modify: tools/run_official_layer0.py
- Modify: tools/run_official_layer1.py
- Modify: tools/run_official_layer3.py
- Modify: tools/run_official_head.py
- Modify: TITAN Ledger documents and README.md

- [ ] Route in-process metadata/header/store access through the context.
- [ ] Run focused regressions and Python compilation.
- [ ] Execute the complete sealed set from fresh state and compare all 93 records with B-0048.
- [ ] Record measured B-0049 evidence, update PROJECT_STATE.md last, commit, and push.

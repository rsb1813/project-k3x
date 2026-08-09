# B-0015 — Exact expert-major speculative verification

This directory contains the measured Milestone 14 synthetic CPU ablation. The run used the WSL2 CPU build, `pread + buffered`, disabled L1, natural routing, three warmups, and twenty measured samples per row.

| Case | Decode tok/s | Reader bytes | Reader calls | Evaluated positions | Discarded positions | Unique payload loads | Assignments | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 163.1535 | 665,616 | 428 | 0 | 0 | 0 | 0 | n/a |
| Token-major perfect block-2 | 160.1659 | 665,616 | 428 | 0 | 0 | 0 | 0 | 1.00 |
| Expert-major perfect block-2 | 201.5550 | 655,824 | 392 | 5 | 0 | 24 | 30 | 1.00 |
| Token-major mixed block-2 | 163.0028 | 665,616 | 428 | 0 | 0 | 0 | 0 | 0.25 |
| Expert-major mixed block-2 | 122.6010 | 680,304 | 482 | 8 | 3 | 39 | 48 | 0.25 |

Every row preserves the greedy generated tokens, final KDA/MLA state, and committed routing trace. In the perfect case, expert-major execution reuses six of thirty expert assignments, reduces Reader bytes by 1.47% and Reader calls by 8.41% relative to token-major, and measures 25.84% higher decode throughput. In the mixed case, rejected suffix work raises Reader bytes by 2.21%, Reader calls by 12.62%, and measures 24.79% lower decode throughput.

These results demonstrate the intended reuse and rejection-cost tradeoff only on the tiny warm synthetic CPU fixture. They do not establish full-model throughput, native-Linux P44 Pro traffic, RTX 5080 performance, or a production default. Token-major remains the default.

The artifact SHA-256 is `29f3fd10c95dcde9f2b012e10e36962363b5cdd79dfeda5f5e3bbaca0cb89b75`. The canonical aggregate-record SHA-256 is `cb95eff274713a21b821695d75ff2655da735513c99215ec5ec14f5ed995b813`. `summary.json` records the SHA-256 of every raw JSON and CSV file; the Python ablation test independently recomputes them and cross-checks selected raw, CSV, and summary fields.

# K3X Device-Side Q8 Decode Implementation Plan

1. Add one CUDA-gated store test proving device decode matches CPU decode at the official BF16 boundary.
2. Add the minimum device decoder and preserve the current CPU path.
3. Run the focused Q8/store and official persistent-runner tests.
4. Measure one released Q8 tensor and then run one full official K3X token.
5. Update the architecture, decision, benchmark, README, checklist, context notes, and project state from measured evidence.

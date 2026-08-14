# K3X Direct-Packed Q8 Matvec Implementation Plan

1. Add a CUDA extension smoke/parity test and witness its failure before the extension exists.
2. Implement the minimal `sm_120` group-128 Q8 matvec extension and lazy loader.
3. Add a packed Q8 matrix handle and `_bf16_matvec` dispatch while preserving the tensor reference path.
4. Add and propagate the explicit `--direct-q8` switch through the official one-process driver.
5. Benchmark one released q-projection, then run layer 0 only if the component gate passes.
6. Run a complete token only if the layer-0 divergence/performance gate is acceptable.
7. Record the measured decision in the TITAN Ledger before moving to MXFP4 residency.

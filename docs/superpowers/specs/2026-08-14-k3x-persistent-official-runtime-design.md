# K3X Persistent Official Runtime Design

## Objective

Replace the 93-process compatibility chain with one opt-in process that executes the already validated official Kimi K3 layer functions. Preserve the subprocess path as the reference mode, then measure the warm boundary before moving graph operations into the production C++/CUDA runtime.

## Selected approach

The first production-facing step is a hybrid persistent Python driver. The existing layer 0, KDA, MLA, and head implementations remain the mathematical authority, but each exposes a callable `run(args)` entrypoint. A new token driver owns their lifetime and calls them in sequence without interpreter restarts.

This is preferred over an immediate whole-graph C++ rewrite because B-0046 already proves the Python graph against the original checkpoint, while the C++ executable still implements only the synthetic graph. It is preferred over more subprocess caching because subprocess ownership cannot retain parsed metadata, host cache state, or CUDA resources across layers.

## Runtime boundary

- `subprocess` remains the default reference execution mode.
- `in-process` is explicit and calls the same layer functions with the same arguments.
- Resume remains driven by the authenticated prefix-state manifest. A completed layer is never recomputed.
- The driver validates every result publication before advancing.
- The K3X set identity continues to be checked by each layer entrypoint.
- The first version keeps state publication on disk. In-memory state and reusable CUDA allocations are the next measured optimization, not an unverified simultaneous rewrite.

## Measurement

The driver records wall time per invoked layer and total resumed execution time. A full warm run must retain token 9689 for input token 1 before its timing is compared with B-0046. The result is reported as TTFT until at least two incremental generated tokens are timed in one process; no inverse-TTFT value is labeled decode TPS.

## Follow-on stages toward 5 TPS

1. Retain parsed topology, index, config, shard directories, prefix state, and CUDA buffers in the same process.
2. Connect the existing packed Q8 trunk and native MXFP4 expert kernels to the official dimensions without host dequantization.
3. Add measured L0/L1 expert residency and deadline-aware prefetch.
4. Add expert-major speculative verification so trunk and expert traffic are amortized across accepted tokens.
5. Compare quality modes against natural Top-16 and original-precision token/logit gates.

Five TPS remains a measured end goal, not a projection from this process-boundary milestone.

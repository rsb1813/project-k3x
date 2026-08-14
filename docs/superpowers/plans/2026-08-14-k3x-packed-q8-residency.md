# K3X Packed Q8 Residency Implementation Plan

1. Add a RED synthetic repeated-matvec test for one-read L0 residency and zero-budget fallback.
2. Implement the bounded shared cache, stable admission, identity key, and telemetry.
3. Attach the cache to `OfficialRuntimeContext` and every lazily opened K3X store.
4. Add explicit host/device byte-budget CLI flags while keeping zero as default.
5. Run two official layer-0 executions in one context and measure cold/warm wall, reads, H2D source bytes, output divergence, and VRAM.
6. Record the result before extending residency to MXFP4 experts or the multi-token loop.

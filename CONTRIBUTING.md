# Contributing to K3X

## Development principles

1. Correctness before performance.
2. Optimizations must retain a reference path.
3. Performance claims require reproducible benchmarks.
4. Do not silently change routing semantics.
5. Do not report synthetic benchmark results as full Kimi K3 performance.

## Before submitting a PR

- Build succeeds
- CTest passes
- Python tests pass
- New behavior has tests
- Performance changes include before/after measurements
- Documentation is updated when behavior changes

# K3X Shared Official Context Design

## Objective

Remove repeated immutable official metadata and sealed-set parsing from the one-process compatibility graph without changing graph math, state publication, routing, precision, or the subprocess reference path.

## Design

One OfficialRuntimeContext is created by the persistent token driver. It authenticates topology against the pinned snapshot, index, and config once; reads the K3XSET manifest once; and lazily caches shard headers and sealed K3X tensor stores by official shard name.

The callable layer stages accept the context only through their in-process argument namespace. Without a context, each stage executes its existing standalone validation path. The context never caches decoded weights in this milestone, so memory remains bounded and B-0048 output parity remains the acceptance gate.

## Measurement

The same input token 1 full graph is executed from a fresh state directory. Every layer output/state digest, token, logit, and final hidden digest must match B-0048. Wall time is compared only against the 1,156.152598-second B-0048 one-process baseline. Decode TPS remains unmeasured.

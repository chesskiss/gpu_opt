          ┌────────────────────────────┐
          │  User Model (PyTorch/ONNX) │
          └────────────┬───────────────┘
                       │
             [Graph Extractor]
                       ↓
           ┌──────────────────────┐
           │    Computation Graph │
           └──────────────────────┘
                       ↓
         [Operator Profiler / Lookup Table]
                       ↓
               [Partitioning Engine]
         (assign ops to CPU / GPU / NPU)
                       ↓
        ┌────────────┬─────────────┐
        │ Subgraph A │  Subgraph B │
        │   on GPU   │   on CPU    │
        └─────┬──────┴─────┬───────┘
              ↓            ↓
   [Executor Engine]   [Executor Engine]
   (e.g. ONNX EP)        (e.g. CPU)
              ↓            ↓
        ┌──────────────────────────┐
        │   Runtime Orchestrator   │
        │ (executes + manages data │
        │  transfer & scheduling)  │
        └──────────────────────────┘
                       ↓
               Final Model Output


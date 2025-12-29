### POC Architecture
```
┌─────────────────────────────────────────┐
│  Client Request                         │
│  [user_id, item_ids, features]          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Router (Python/FastAPI)                 │
│  - Parse request                         │
│  - Decide: CPU or GPU path?              │
│  - If cold start: trigger CPU prefill    │
└──────────────┬───────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌─────────────┐ ┌─────────────┐
│ CPU Worker  │ │ GPU Worker  │
│             │ │             │
│ Embedding   │ │ Full model  │
│ lookup only │ │ (baseline)  │
└──────┬──────┘ └─────────────┘
       │
       │ Send dense vectors
       ▼
┌─────────────┐
│ GPU Worker  │
│ MLP only    │
└──────┬──────┘
       │
       ▼
  Response
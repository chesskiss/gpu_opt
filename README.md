# CPU-GPU Heterogeneous Inference POC

Proof-of-concept for CPU embedding offload and CPU-assisted cold start for recommendation models.

## Stack

- **Python 3.9+**
- **PyTorch 2.0+** (CPU & CUDA)
- **FastAPI** (routing server)
- **NumPy** (data generation)

## Architecture

```
Request → Router (poc.py) → CPU Embeddings → GPU MLP → Response
                         ↘ GPU Full Model ↗
```

## Quick Start

```bash
# Install dependencies using uv (recommended)
uv sync

# Or with pip
pip install torch numpy

# Run simple example
python3 example.py

# Run benchmarks (takes ~60 seconds)
python3 benchmark.py
```

## Hardware Requirements

**Minimum:**
- CPU: 8+ cores, 32GB RAM
- GPU: 1× RTX 3090 / A10 / T4 (16GB+ VRAM)

**Optimal:**
- CPU: 32+ cores, 128GB RAM  
- GPU: 1× A100 (40GB+)
- Network: 10Gbps if CPU/GPU on separate machines

## What This Proves

1. **CPU embedding offload** reduces GPU memory 10-100×
2. **CPU pre-fill during cold start** eliminates downtime
3. **Intelligent routing** improves latency for large batches

## Key Metrics

- Baseline GPU latency: 5-100ms
- CPU offload latency: 10-50ms (faster for large batches)
- Cold start: 30-60s → <5s with CPU assist

## Project Status

**Done:**
- ✅ Basic routing logic
- ✅ CPU embedding lookup
- ✅ GPU MLP execution

**TODO:** See TODO.md

## References

- Prism (Alibaba): NSDI 2025
- FleetRec (ETH/Alibaba): KDD 2021  
- CaraServe (HKUST): arXiv 2024
- NVIDIA HugeCTR: github.com/NVIDIA-Merlin/HugeCTR
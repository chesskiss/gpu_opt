"""
Benchmark CPU-GPU Heterogeneous Inference
Compares three scenarios:
1. GPU-only: Everything on GPU
2. CPU-only: Everything on CPU
3. Hybrid: CPU embeddings + GPU MLP
"""

import torch
import torch.nn as nn
import numpy as np
import time


class SimpleMLP(nn.Module):
    """MLP for recommendation scoring"""
    def __init__(self, input_dim=128, hidden_dim=512, hidden_dim2=256):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class GPUOnlyModel:
    """All components on GPU (or MPS)"""
    def __init__(self, num_embeddings=10_000_000, embedding_dim=128, device='cuda'):
        self.device = device
        self.embedding_table = nn.Embedding(num_embeddings, embedding_dim).to(device)
        self.mlp = SimpleMLP(input_dim=embedding_dim).to(device)

    def forward(self, user_ids, item_ids):
        # Everything happens on GPU
        user_ids_tensor = torch.tensor(user_ids, device=self.device)
        item_ids_tensor = torch.tensor(item_ids, device=self.device)

        user_embeddings = self.embedding_table(user_ids_tensor)
        item_embeddings = self.embedding_table(item_ids_tensor)
        combined = (user_embeddings + item_embeddings) / 2.0

        scores = self.mlp(combined)
        return scores


class CPUOnlyModel:
    """All components on CPU"""
    def __init__(self, num_embeddings=10_000_000, embedding_dim=128):
        self.embedding_table = nn.Embedding(num_embeddings, embedding_dim).to('cpu')
        self.mlp = SimpleMLP(input_dim=embedding_dim).to('cpu')

    def forward(self, user_ids, item_ids):
        # Everything happens on CPU
        user_ids_tensor = torch.tensor(user_ids, device='cpu')
        item_ids_tensor = torch.tensor(item_ids, device='cpu')

        user_embeddings = self.embedding_table(user_ids_tensor)
        item_embeddings = self.embedding_table(item_ids_tensor)
        combined = (user_embeddings + item_embeddings) / 2.0

        scores = self.mlp(combined)
        return scores


class HybridModel:
    """CPU embeddings + GPU MLP"""
    def __init__(self, num_embeddings=10_000_000, embedding_dim=128, device='cuda'):
        self.device = device
        # Embeddings stay on CPU
        self.embedding_table = nn.Embedding(num_embeddings, embedding_dim).to('cpu')
        # MLP on GPU
        self.mlp = SimpleMLP(input_dim=embedding_dim).to(device)

    def forward(self, user_ids, item_ids):
        # Step 1: Lookup on CPU
        user_ids_tensor = torch.tensor(user_ids, device='cpu')
        item_ids_tensor = torch.tensor(item_ids, device='cpu')

        user_embeddings = self.embedding_table(user_ids_tensor)
        item_embeddings = self.embedding_table(item_ids_tensor)
        combined = (user_embeddings + item_embeddings) / 2.0

        # Step 2: Transfer to GPU
        combined = combined.to(self.device)

        # Step 3: MLP on GPU
        scores = self.mlp(combined)
        return scores


def benchmark_model(model, batch_sizes, num_embeddings, num_warmup=5, num_runs=20):
    results = {}
    
    for batch_size in batch_sizes:
        user_ids = np.random.randint(0, num_embeddings, size=batch_size)
        item_ids = np.random.randint(0, num_embeddings, size=batch_size)
        
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model.forward(user_ids, item_ids)
        
        # Synchronize before timing (important!)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()  # ← ADD THIS!
        
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model.forward(user_ids, item_ids)
            
            # Synchronize after work (critical for accurate timing!)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()  # ← ADD THIS!
            
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
        
        median_latency = np.median(latencies)
        results[batch_size] = median_latency
    
    return results


def get_memory_usage():
    """Get GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, return 0
        return 0
    else:
        return 0


def print_results(scenario_name, results, memory_mb=None):
    """Print benchmark results in a nice format"""
    print(f"\n{scenario_name}")
    print("-" * 50)

    for batch_size, latency in results.items():
        print(f"  Batch {batch_size:>4}: {latency:>8.2f} ms")

    if memory_mb is not None and memory_mb > 0:
        print(f"  GPU Memory: {memory_mb:.1f} MB")


def main():
    """Run all benchmarks"""
    print("\n" + "="*60)
    print("Benchmarking CPU-GPU Heterogeneous Inference")
    print("="*60)

    # Configuration
    num_embeddings = 10_000_000
    embedding_dim = 128
    batch_sizes = [1, 10, 50, 100, 500]

    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        print(f"\nUsing GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        device_name = "Apple Silicon MPS"
        print(f"\nUsing MPS: {device_name}")
    else:
        device = 'cpu'
        device_name = "CPU only"
        print(f"\nWARNING: No GPU available, using CPU for all scenarios")
        print(f"GPU-only and Hybrid scenarios will actually run on CPU")

    print(f"Embedding table: {num_embeddings:,} entries x {embedding_dim} dims")
    print(f"Batch sizes: {batch_sizes}")

    # Scenario 1: GPU-only
    print(f"\n{'='*60}")
    print("Scenario 1: GPU-only (everything on GPU)")
    print(f"{'='*60}")
    print("Creating model...")

    try:
        gpu_model = GPUOnlyModel(num_embeddings, embedding_dim, device=device)
        print("Running benchmark...")
        gpu_results = benchmark_model(gpu_model, batch_sizes, num_embeddings)
        gpu_memory = get_memory_usage()
        print_results("GPU-only Results", gpu_results, gpu_memory)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("GPU-only scenario failed (likely out of memory)")
        gpu_results = None

    # Scenario 2: CPU-only
    print(f"\n{'='*60}")
    print("Scenario 2: CPU-only (baseline)")
    print(f"{'='*60}")
    print("Creating model...")

    cpu_model = CPUOnlyModel(num_embeddings, embedding_dim)
    print("Running benchmark...")
    cpu_results = benchmark_model(cpu_model, batch_sizes, num_embeddings)
    print_results("CPU-only Results", cpu_results)

    # Scenario 3: Hybrid
    print(f"\n{'='*60}")
    print("Scenario 3: Hybrid (CPU embeddings + GPU MLP)")
    print(f"{'='*60}")
    print("Creating model...")

    hybrid_model = HybridModel(num_embeddings, embedding_dim, device=device)
    print("Running benchmark...")
    hybrid_results = benchmark_model(hybrid_model, batch_sizes, num_embeddings)
    hybrid_memory = get_memory_usage()
    print_results("Hybrid Results", hybrid_results, hybrid_memory)

    # Summary comparison
    print(f"\n{'='*60}")
    print("Summary Comparison")
    print(f"{'='*60}")
    print(f"\n{'Batch':>6} | {'CPU-only':>10} | {'Hybrid':>10} | {'GPU-only':>10} | {'Winner'}")
    print("-" * 60)

    for batch_size in batch_sizes:
        cpu_lat = cpu_results[batch_size]
        hybrid_lat = hybrid_results[batch_size]

        if gpu_results is not None:
            gpu_lat = gpu_results[batch_size]
            winner = min(
                [('CPU', cpu_lat), ('Hybrid', hybrid_lat), ('GPU', gpu_lat)],
                key=lambda x: x[1]
            )[0]
            print(f"{batch_size:>6} | {cpu_lat:>8.2f}ms | {hybrid_lat:>8.2f}ms | {gpu_lat:>8.2f}ms | {winner}")
        else:
            winner = 'Hybrid' if hybrid_lat < cpu_lat else 'CPU'
            print(f"{batch_size:>6} | {cpu_lat:>8.2f}ms | {hybrid_lat:>8.2f}ms | {'N/A':>10} | {winner}")

    # Key findings
    print(f"\n{'='*60}")
    print("Key Findings")
    print(f"{'='*60}")

    # Find crossover point where hybrid beats CPU
    crossover_batch = None
    for batch_size in batch_sizes:
        if hybrid_results[batch_size] < cpu_results[batch_size]:
            crossover_batch = batch_size
            break

    if crossover_batch:
        speedup = cpu_results[crossover_batch] / hybrid_results[crossover_batch]
        print(f"✓ Hybrid faster than CPU-only starting at batch size {crossover_batch}")
        print(f"  Speedup: {speedup:.2f}x at batch {crossover_batch}")
    else:
        print(f"✗ Hybrid not faster than CPU-only at tested batch sizes")
        print(f"  Transfer overhead dominates for small batches")

    # Compare hybrid vs GPU-only for large batches
    if gpu_results is not None:
        large_batch = batch_sizes[-1]
        if hybrid_results[large_batch] < gpu_results[large_batch]:
            speedup = gpu_results[large_batch] / hybrid_results[large_batch]
            print(f"✓ Hybrid faster than GPU-only at batch {large_batch}")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Reason: Avoids loading large embedding table to GPU")
        else:
            print(f"✗ GPU-only faster at batch {large_batch}")

    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

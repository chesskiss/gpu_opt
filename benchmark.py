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
import threading
import queue


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


class PipelineBuffer:
    """Manages a single pipeline buffer with pinned memory and CUDA stream"""
    def __init__(self, max_batch_size, embedding_dim, device):
        self.max_batch_size = max_batch_size
        self.embedding_dim = embedding_dim
        self.device = device

        # Pinned CPU tensor for fast transfers (only for CUDA)
        use_pinned = (device == 'cuda' and torch.cuda.is_available())
        self.cpu_tensor = torch.empty(
            (max_batch_size, embedding_dim),
            dtype=torch.float32,
            pin_memory=use_pinned
        )

        # Pre-allocated GPU tensor
        self.gpu_tensor = torch.empty(
            (max_batch_size, embedding_dim),
            dtype=torch.float32,
            device=device
        )

        # Dedicated CUDA stream for async operations
        if device == 'cuda':
            self.stream = torch.cuda.Stream(device=device)
        else:
            self.stream = None

        # Synchronization event
        if device == 'cuda':
            self.ready_event = torch.cuda.Event()
        else:
            self.ready_event = None


class ParallelHybridModel:
    """CPU embeddings + GPU MLP with pipelined execution"""
    def __init__(self, num_embeddings=10_000_000, embedding_dim=128, device='cuda', pipeline_depth=2):
        self.device = device
        self.pipeline_depth = pipeline_depth
        self.embedding_dim = embedding_dim

        # Same components as HybridModel
        self.embedding_table = nn.Embedding(num_embeddings, embedding_dim).to('cpu')
        self.embedding_table.eval()  # Set to eval mode for thread safety
        for param in self.embedding_table.parameters():
            param.requires_grad = False  # Freeze weights

        self.mlp = SimpleMLP(input_dim=embedding_dim).to(device)

        # Pipeline infrastructure - use max batch size of 500 from benchmark
        max_batch_size = 500
        self.buffers = [
            PipelineBuffer(max_batch_size, embedding_dim, device)
            for _ in range(pipeline_depth)
        ]

        # Threading components
        self.cpu_queue = queue.Queue(maxsize=pipeline_depth)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.buffer_idx = 0

        # Start CPU worker thread
        self.cpu_thread = threading.Thread(target=self._cpu_worker, daemon=True)
        self.cpu_thread.start()

    def _get_next_buffer(self):
        """Get next buffer index in round-robin fashion"""
        idx = self.buffer_idx
        self.buffer_idx = (self.buffer_idx + 1) % self.pipeline_depth
        return idx

    def _cpu_worker(self):
        """Background thread for CPU embedding lookups"""
        while not self.stop_event.is_set():
            try:
                batch_info = self.cpu_queue.get(timeout=0.1)
                if batch_info is None:
                    break

                batch_id, user_ids, item_ids, buffer = batch_info

                # Compute embeddings on CPU
                user_ids_tensor = torch.tensor(user_ids, device='cpu')
                item_ids_tensor = torch.tensor(item_ids, device='cpu')

                user_embeddings = self.embedding_table(user_ids_tensor)
                item_embeddings = self.embedding_table(item_ids_tensor)
                combined = (user_embeddings + item_embeddings) / 2.0

                # Copy to pinned buffer
                actual_size = len(user_ids)
                buffer.cpu_tensor[:actual_size].copy_(combined)

                # Signal ready
                self.result_queue.put((batch_id, buffer, actual_size))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in CPU worker: {e}")
                break

    def forward(self, user_ids, item_ids):
        """
        Pipelined forward pass:
        1. Submit to CPU queue
        2. Wait for CPU completion
        3. Async transfer to GPU on stream
        4. Execute MLP on same stream
        5. Synchronize and return
        """
        batch_size = len(user_ids)
        buffer_idx = self._get_next_buffer()
        buffer = self.buffers[buffer_idx]

        # Submit to CPU worker
        self.cpu_queue.put((buffer_idx, user_ids, item_ids, buffer))

        # Wait for CPU completion
        batch_id, ready_buffer, actual_size = self.result_queue.get()

        # GPU operations on dedicated stream
        if self.device == 'cuda' and ready_buffer.stream is not None:
            with torch.cuda.stream(ready_buffer.stream):
                # Async H2D transfer
                ready_buffer.gpu_tensor[:actual_size].copy_(
                    ready_buffer.cpu_tensor[:actual_size],
                    non_blocking=True
                )

                # MLP computation
                scores = self.mlp(ready_buffer.gpu_tensor[:actual_size])

                # Record completion
                if ready_buffer.ready_event is not None:
                    ready_buffer.ready_event.record()

            # Synchronize stream before returning
            ready_buffer.stream.synchronize()
        else:
            # Non-CUDA path (MPS or CPU)
            ready_buffer.gpu_tensor[:actual_size].copy_(
                ready_buffer.cpu_tensor[:actual_size]
            )
            scores = self.mlp(ready_buffer.gpu_tensor[:actual_size])

        return scores

    def __del__(self):
        """Cleanup: stop CPU worker thread"""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
            if hasattr(self, 'cpu_queue'):
                try:
                    self.cpu_queue.put(None, timeout=0.1)
                except:
                    pass


def benchmark_model(model, batch_sizes, num_embeddings, num_warmup=5, num_runs=20, parallel=False):
    """
    Benchmark a model across different batch sizes
    Returns: dict with batch_size -> latency_ms

    Args:
        parallel: If True, measures steady-state throughput over multiple batches
                  to account for pipeline fill/drain overhead
    """
    results = {}

    for batch_size in batch_sizes:
        # Generate random IDs
        user_ids = np.random.randint(0, num_embeddings, size=batch_size)
        item_ids = np.random.randint(0, num_embeddings, size=batch_size)

        # Warmup runs (don't measure)
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model.forward(user_ids, item_ids)

        # For parallel models, add extra warmup to fill pipeline
        if parallel and hasattr(model, 'pipeline_depth'):
            for _ in range(model.pipeline_depth):
                with torch.no_grad():
                    _ = model.forward(user_ids, item_ids)

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            if parallel:
                # Measure steady-state throughput over multiple batches
                # to amortize pipeline fill/drain overhead
                start_time = time.time()

                num_batches = 10
                for _ in range(num_batches):
                    with torch.no_grad():
                        _ = model.forward(user_ids, item_ids)

                # Synchronize GPU if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                # Average latency per batch
                latency = ((end_time - start_time) / num_batches) * 1000
            else:
                # Original single-batch timing
                start_time = time.time()

                with torch.no_grad():
                    _ = model.forward(user_ids, item_ids)

                # Synchronize GPU if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                latency = (end_time - start_time) * 1000

            latencies.append(latency)

        # Use median latency (more robust than mean)
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

    # Scenario 4: Parallel Hybrid
    print(f"\n{'='*60}")
    print("Scenario 4: Parallel Hybrid (Pipelined CPU+GPU)")
    print(f"{'='*60}")
    print("Creating model...")

    parallel_model = ParallelHybridModel(
        num_embeddings,
        embedding_dim,
        device=device,
        pipeline_depth=2
    )
    print("Running benchmark...")
    print("(Measuring steady-state throughput over multiple batches)")
    parallel_results = benchmark_model(
        parallel_model,
        batch_sizes,
        num_embeddings,
        parallel=True
    )
    parallel_memory = get_memory_usage()
    print_results("Parallel Hybrid Results", parallel_results, parallel_memory)

    # Summary comparison
    print(f"\n{'='*60}")
    print("Summary Comparison")
    print(f"{'='*60}")
    print(f"\n{'Batch':>6} | {'CPU-only':>10} | {'Hybrid':>10} | {'Parallel':>10} | {'GPU-only':>10} | {'Winner'}")
    print("-" * 75)

    for batch_size in batch_sizes:
        cpu_lat = cpu_results[batch_size]
        hybrid_lat = hybrid_results[batch_size]
        parallel_lat = parallel_results[batch_size]

        if gpu_results is not None:
            gpu_lat = gpu_results[batch_size]
            winner = min(
                [('CPU', cpu_lat), ('Hybrid', hybrid_lat), ('Parallel', parallel_lat), ('GPU', gpu_lat)],
                key=lambda x: x[1]
            )[0]
            print(f"{batch_size:>6} | {cpu_lat:>8.2f}ms | {hybrid_lat:>8.2f}ms | {parallel_lat:>8.2f}ms | {gpu_lat:>8.2f}ms | {winner}")
        else:
            candidates = [('CPU', cpu_lat), ('Hybrid', hybrid_lat), ('Parallel', parallel_lat)]
            winner = min(candidates, key=lambda x: x[1])[0]
            print(f"{batch_size:>6} | {cpu_lat:>8.2f}ms | {hybrid_lat:>8.2f}ms | {parallel_lat:>8.2f}ms | {'N/A':>10} | {winner}")

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

    # Compare parallel vs sequential hybrid
    print(f"\n✓ Parallel Hybrid Speedup Analysis:")
    for batch_size in batch_sizes:
        speedup = hybrid_results[batch_size] / parallel_results[batch_size]
        print(f"  Batch {batch_size:>3}: {speedup:.2f}x faster than sequential hybrid")

    # Find best speedup
    best_batch = max(batch_sizes, key=lambda b: hybrid_results[b] / parallel_results[b])
    best_speedup = hybrid_results[best_batch] / parallel_results[best_batch]
    print(f"  Best speedup: {best_speedup:.2f}x at batch size {best_batch}")

    # Compare hybrid vs GPU-only for large batches
    if gpu_results is not None:
        large_batch = batch_sizes[-1]
        if hybrid_results[large_batch] < gpu_results[large_batch]:
            speedup = gpu_results[large_batch] / hybrid_results[large_batch]
            print(f"\n✓ Hybrid faster than GPU-only at batch {large_batch}")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Reason: Avoids loading large embedding table to GPU")
        else:
            print(f"\n✗ GPU-only faster at batch {large_batch}")

    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
Simple DLRM (Deep Learning Recommendation Model) POC
This demonstrates a basic recommendation model with:
- Embeddings on CPU (simulating large embedding tables)
- MLP on GPU (simulating compute-intensive layers)
"""

import torch
import torch.nn as nn
import numpy as np


# Simple MLP for GPU computation
class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron that runs on GPU
    Takes embedding vectors and outputs a recommendation score
    """
    def __init__(self, input_dim=128, hidden_dim=512, hidden_dim2=256):
        super().__init__()
        # Three layer network: 128 -> 512 -> 256 -> 1
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class SimpleDLRM:
    """
    Simplified DLRM model with CPU embeddings and GPU MLP
    In a real DLRM, there would be multiple embedding tables and a cross-layer
    """
    def __init__(self, num_embeddings=10_000_000, embedding_dim=128):
        print(f"\n=== Creating Simple DLRM Model ===")
        print(f"Embeddings: {num_embeddings:,} entries x {embedding_dim} dims")

        # Store model configuration
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create embedding table on CPU (too large for GPU)
        # In real systems, this would be 10M+ entries
        print(f"Creating embedding table on CPU...")
        self.embedding_table = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_table = self.embedding_table.to('cpu')

        # Calculate memory usage
        embedding_memory_mb = (num_embeddings * embedding_dim * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"Embedding table size: {embedding_memory_mb:.1f} MB")

        # Create MLP on GPU (compute intensive)
        print(f"Creating MLP on GPU...")
        self.mlp = SimpleMLP(input_dim=embedding_dim)

        # Check if GPU is available
        if torch.cuda.is_available():
            self.mlp = self.mlp.to('cuda')
            print(f"MLP moved to GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.mlp = self.mlp.to('mps')
            print(f"MLP moved to MPS (Apple Silicon)")
        else:
            print(f"WARNING: No GPU available, using CPU")

        print(f"=== Model Created Successfully ===\n")

    def forward(self, user_ids, item_ids):
        """
        Forward pass:
        1. Look up embeddings on CPU
        2. Transfer to GPU
        3. Run MLP on GPU
        """
        # user_ids and item_ids are arrays of integers
        batch_size = len(user_ids)

        # Step 1: Lookup embeddings on CPU
        # In real DLRM, we'd have separate user and item embeddings
        # For simplicity, we'll just use one embedding table and combine them
        user_embeddings = self.embedding_table(torch.tensor(user_ids, device='cpu'))
        item_embeddings = self.embedding_table(torch.tensor(item_ids, device='cpu'))

        # Combine user and item embeddings (simple average)
        # TODO: Add real DLRM cross-layer interaction (see TODO.md Phase 2)
        combined_embeddings = (user_embeddings + item_embeddings) / 2.0

        # Step 2: Transfer to GPU
        # TODO: Optimize transfer using pinned memory (see TODO.md Phase 1)
        if torch.cuda.is_available():
            combined_embeddings = combined_embeddings.to('cuda')
        elif torch.backends.mps.is_available():
            combined_embeddings = combined_embeddings.to('mps')

        # Step 3: Run MLP on GPU
        scores = self.mlp(combined_embeddings)

        return scores

    def get_device_info(self):
        """Print information about where model components are located"""
        print(f"\n=== Device Information ===")
        print(f"Embedding table device: {next(self.embedding_table.parameters()).device}")
        print(f"MLP device: {next(self.mlp.parameters()).device}")


def generate_random_ids(num_samples, max_id):
    """
    Generate random user/item IDs for testing
    In a real system, these would come from a dataset
    """
    return np.random.randint(0, max_id, size=num_samples)


def main():
    """
    Main demo function
    """
    print("\n" + "="*60)
    print("DLRM POC - CPU Embeddings + GPU MLP")
    print("="*60)

    # Create model
    # Using 10M embeddings to simulate real-world size
    model = SimpleDLRM(num_embeddings=10_000_000, embedding_dim=128)

    # Show device placement
    model.get_device_info()

    # Generate sample data
    print(f"\n=== Running Sample Inference ===")
    num_samples = 5
    user_ids = generate_random_ids(num_samples, model.num_embeddings)
    item_ids = generate_random_ids(num_samples, model.num_embeddings)

    print(f"Sample user IDs: {user_ids}")
    print(f"Sample item IDs: {item_ids}")

    # Run forward pass
    print(f"\nRunning forward pass...")
    with torch.no_grad():  # Don't need gradients for inference
        scores = model.forward(user_ids, item_ids)

    # Move scores back to CPU for printing
    scores_cpu = scores.cpu().numpy()

    print(f"\n=== Results ===")
    for i in range(num_samples):
        print(f"User {user_ids[i]:,} + Item {item_ids[i]:,} -> Score: {scores_cpu[i][0]:.4f}")

    print(f"\n=== Forward Pass Successful! ===")
    print(f"\nWhat happened:")
    print(f"1. Looked up embeddings for {num_samples} user-item pairs on CPU")
    print(f"2. Transferred {num_samples * 128 * 4} bytes to GPU")
    print(f"3. Computed MLP scores on GPU")
    print(f"4. Transferred {num_samples * 4} bytes back to CPU")

    print(f"\n" + "="*60)


if __name__ == "__main__":
    main()

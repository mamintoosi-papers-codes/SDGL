import numpy as np
import matplotlib.pyplot as plt
import os
from SDGL.Pems4.util import estimate_adjacency_with_dagma

# Settings
dataset_name = "PEMSD4"
cache_dir = "./dagma_adj"

# Try different lambda1 values
lambda1_values = [0.01, 0.03, 0.05, 0.1, 0.2]
alpha_values = [0.3, 0.5, 0.7, 1.0]  # 1.0 means only DAGMA

# Create a grid of plots
fig, axs = plt.subplots(len(lambda1_values), len(alpha_values), figsize=(15, 15))

for i, lambda1 in enumerate(lambda1_values):
    for j, alpha in enumerate(alpha_values):
        # Get adjacency matrix with these parameters
        adj = estimate_adjacency_with_dagma(
            dataset_name=dataset_name,
            lambda1=lambda1,
            use_distance=True,
            alpha=alpha
        )
        
        # Plot the adjacency matrix
        im = axs[i, j].imshow(adj, cmap='viridis')
        axs[i, j].set_title(f'λ={lambda1}, α={alpha}')
        axs[i, j].axis('off')
        
        # Print statistics
        print(f"λ={lambda1}, α={alpha}: Non-zero elements: {np.count_nonzero(adj)}, Sparsity: {np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1]):.4f}")

plt.tight_layout()
plt.savefig(f"./dagma_parameter_tuning.png", dpi=300, bbox_inches='tight')
plt.show()

# Find the best parameters based on sparsity
# We want a reasonably sparse matrix (not too dense, not too sparse)
results = []
for lambda1 in lambda1_values:
    for alpha in alpha_values:
        cache_file = f"{cache_dir}/{dataset_name}_adj_dagma_mode1_lambda{lambda1:.4f}_alpha{alpha:.2f}.npy"
        if os.path.exists(cache_file):
            adj = np.load(cache_file)
            sparsity = np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1])
            results.append((lambda1, alpha, sparsity))

# Sort by sparsity
results.sort(key=lambda x: abs(x[2] - 0.1))  # Target sparsity around 10%
print("\nBest parameters based on sparsity:")
for lambda1, alpha, sparsity in results[:3]:
    print(f"λ={lambda1}, α={alpha}: Sparsity={sparsity:.4f}")
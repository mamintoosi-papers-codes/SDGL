import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from SDGL.Pems4.util import estimate_adjacency_with_dagma

# Settings
dataset_name = "PEMSD4"
cache_dir = "./dagma_adj"
cache_file = f"c:/git/mamintoosi-papers-codes/SDGL/SDGL/Pems4/dagma_adj/{dataset_name}_adj_dagma_mode1.npy"
distance_file = "c:/git/mamintoosi-papers-codes/SDGL/SDGL/Pems4/data/PeMSD4/distance.csv"

# Function to create adjacency matrix from distance.csv
def create_adj_from_distance(distance_file, threshold=None):
    """
    Create adjacency matrix from distance.csv file
    
    Args:
        distance_file: Path to distance.csv file
        threshold: Distance threshold to create binary adjacency matrix (optional)
        
    Returns:
        adj_mx: Adjacency matrix
    """
    # Read distance.csv file
    df = pd.read_csv(distance_file)
    print(f"Distance file shape: {df.shape}")
    
    # Get number of nodes
    nodes = set(df['from'].unique()) | set(df['to'].unique())
    num_nodes = len(nodes)
    print(f"Number of nodes: {num_nodes}")
    
    # Create node mapping
    node_mapping = {node: i for i, node in enumerate(sorted(nodes))}
    
    # Create empty adjacency matrix
    adj_mx = np.zeros((num_nodes, num_nodes))
    
    # Fill adjacency matrix with distances
    for _, row in df.iterrows():
        from_node = node_mapping[row['from']]
        to_node = node_mapping[row['to']]
        distance = row['cost']
        
        # Set distance in adjacency matrix
        adj_mx[from_node, to_node] = distance
        # Make it symmetric (undirected graph)
        adj_mx[to_node, from_node] = distance
    
    # Convert to binary adjacency matrix if threshold is provided
    if threshold is not None:
        # Invert distances (closer nodes have higher values)
        max_distance = np.max(adj_mx[adj_mx > 0])
        adj_mx_inv = np.zeros_like(adj_mx)
        mask = adj_mx > 0
        adj_mx_inv[mask] = max_distance - adj_mx[mask]
        
        # Apply threshold
        adj_mx = (adj_mx_inv > threshold).astype(float)
    
    return adj_mx

# Load DAGMA adjacency matrix
if os.path.exists(cache_file):
    print(f"Loading DAGMA adjacency matrix from {cache_file}")
    dagma_adj = np.load(cache_file)
else:
    print("DAGMA adjacency matrix not found. Computing it...")
    dagma_adj = estimate_adjacency_with_dagma(dataset_name=dataset_name)

# Create adjacency matrix from distance.csv
# First, create a weighted adjacency matrix
distance_adj_weighted = create_adj_from_distance(distance_file)

# Then create a binary adjacency matrix with a threshold
# We'll use the median of non-zero distances as threshold
non_zero_distances = distance_adj_weighted[distance_adj_weighted > 0]
threshold = np.median(non_zero_distances)
distance_adj_binary = create_adj_from_distance(distance_file, threshold)

# Make sure both matrices have the same shape
if dagma_adj.shape != distance_adj_binary.shape:
    print(f"Warning: Matrix shapes don't match. DAGMA: {dagma_adj.shape}, Distance: {distance_adj_binary.shape}")
    # Use the smaller dimension
    min_dim = min(dagma_adj.shape[0], distance_adj_binary.shape[0])
    dagma_adj = dagma_adj[:min_dim, :min_dim]
    distance_adj_binary = distance_adj_binary[:min_dim, :min_dim]
    distance_adj_weighted = distance_adj_weighted[:min_dim, :min_dim]

# Normalize weighted distance matrix for visualization
if np.max(distance_adj_weighted) > 0:
    distance_adj_norm = distance_adj_weighted / np.max(distance_adj_weighted)
else:
    distance_adj_norm = distance_adj_weighted

# Display matrices
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Display distance-based weighted adjacency matrix
axs[0, 0].imshow(distance_adj_norm, cmap='viridis')
axs[0, 0].set_title('Distance-based Weighted Adjacency Matrix')
axs[0, 0].axis('off')

# Display distance-based binary adjacency matrix
axs[0, 1].imshow(distance_adj_binary, cmap='viridis')
axs[0, 1].set_title('Distance-based Binary Adjacency Matrix')
axs[0, 1].axis('off')

# Display DAGMA adjacency matrix
axs[1, 0].imshow(dagma_adj, cmap='viridis')
axs[1, 0].set_title('DAGMA Adjacency Matrix')
axs[1, 0].axis('off')

# Create overlay image (Red: Distance-based, Green: DAGMA)
overlay_image = np.zeros((dagma_adj.shape[0], dagma_adj.shape[1], 3))
overlay_image[..., 0] = distance_adj_binary  # Red channel
overlay_image[..., 1] = dagma_adj            # Green channel
overlay_image[..., 2] = 0                    # Blue channel

# Display overlay image
axs[1, 1].imshow(overlay_image)
axs[1, 1].set_title('Overlay (Red: Distance-based, Green: DAGMA)')
axs[1, 1].axis('off')

plt.tight_layout()
plt.savefig(f"./comparison_{dataset_name}_adjacency.png", dpi=300, bbox_inches='tight')
plt.show()

# Print statistical information
print(f"DAGMA Adjacency Matrix Shape: {dagma_adj.shape}")
print(f"DAGMA Adjacency Matrix Min: {np.min(dagma_adj)}, Max: {np.max(dagma_adj)}")
print(f"DAGMA Adjacency Matrix Non-zero elements: {np.count_nonzero(dagma_adj)}")
print(f"DAGMA Adjacency Matrix Sparsity: {np.count_nonzero(dagma_adj) / (dagma_adj.shape[0] * dagma_adj.shape[1]):.4f}")

print(f"Distance-based Binary Adjacency Matrix Shape: {distance_adj_binary.shape}")
print(f"Distance-based Binary Adjacency Matrix Non-zero elements: {np.count_nonzero(distance_adj_binary)}")
print(f"Distance-based Binary Adjacency Matrix Sparsity: {np.count_nonzero(distance_adj_binary) / (distance_adj_binary.shape[0] * distance_adj_binary.shape[1]):.4f}")

# Calculate correlation between matrices
correlation = np.corrcoef(distance_adj_binary.flatten(), dagma_adj.flatten())[0, 1]
print(f"Correlation between matrices: {correlation:.4f}")

# Save the distance-based adjacency matrix for future use
os.makedirs(cache_dir, exist_ok=True)
np.save(f"{cache_dir}/{dataset_name}_adj_distance.npy", distance_adj_binary)
print(f"Saved distance-based adjacency matrix to {cache_dir}/{dataset_name}_adj_distance.npy")
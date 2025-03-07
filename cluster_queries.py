import os
import pickle
import argparse
import numpy as np
import faiss
from openai import AzureOpenAI  # Changed to AzureOpenAI
import networkx as nx
from collections import defaultdict

# Set up argument parser for configurable parameters
parser = argparse.ArgumentParser(description="Cluster similar queries using embeddings and FAISS.")
parser.add_argument("query_file", type=str, help="Path to the text file containing queries, one per line.")
parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold for clustering (default 0.9).")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generating embeddings (default 100).")
parser.add_argument("--top_n", type=int, default=10, help="Number of largest clusters to display (default 10).")
parser.add_argument("--deployment", type=str, default="text-embedding-3-large", help="Azure OpenAI embeddings deployment name (default: text-embedding-3-large)")
args = parser.parse_args()

# Initialize Azure OpenAI client with API key and endpoint from environment variables
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=args.deployment
)

# Function to generate embeddings in batches
def generate_embeddings(queries, batch_size=100):
    embeddings = []
    for i in range(0, len(queries), batch_size):
        print(f"Processing batch {i // batch_size + 1} of {len(queries) // batch_size + 1}...")
        batch = queries[i:i + batch_size]
        response = client.embeddings.create(model=args.deployment,input=batch)  
        batch_embeddings = [embedding.embedding for embedding in response.data]
        embeddings.extend(batch_embeddings)
        print(f"Processed {len(batch)} queries.")
    print("All queries processed.")
    return np.array(embeddings, dtype=np.float32)

# Load or generate embeddings with caching
# Use the input file name to create a cache file name
base_name = os.path.splitext(os.path.basename(args.query_file))[0]
cache_file = f"{base_name}.embeddings.pkl"

if os.path.exists(cache_file):
    print(f"Loading cached embeddings from {cache_file}...")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    queries = data["queries"]
    embeddings = data["embeddings"]
else:
    print(f"Generating embeddings for {args.query_file}...")
    # Add encoding parameter to handle non-ASCII characters
    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    embeddings = generate_embeddings(queries, batch_size=args.batch_size)
    with open(cache_file, "wb") as f:
        pickle.dump({"queries": queries, "embeddings": embeddings}, f)
    print(f"Embeddings cached to {cache_file}")

# Normalize embeddings for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized_embeddings = embeddings / norms

# Create and populate FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity (with normalized embeddings)
index.add(normalized_embeddings)

# Save the FAISS index to disk for persistence
faiss.write_index(index, "faiss_index.bin")
print("FAISS index saved to faiss_index.bin")

# Perform range search to find similar query pairs
threshold = args.threshold
lims, D, I = index.range_search(normalized_embeddings, threshold)

# Build a graph of similar queries
graph = defaultdict(list)
for i in range(len(queries)):
    start = lims[i]
    end = lims[i + 1]
    for j in range(start, end):
        neighbor = I[j]
        if neighbor != i:  # Avoid self-loops
            graph[i].append(neighbor)

# Find clusters using connected components
G = nx.Graph()
G.add_nodes_from(range(len(queries)))
for node, neighbors in graph.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

clusters = list(nx.connected_components(G))

# Sort clusters by size to find the most common questions
# Add secondary sorting criterion (by min node ID) for deterministic ordering
clusters = sorted(clusters, key=lambda c: (-len(c), min(c)))

# Display the top N largest clusters
print(f"\nTop {args.top_n} largest clusters (representing the most common questions):")
for i, cluster in enumerate(clusters[:args.top_n], 1):
    print(f"Cluster {i} (size {len(cluster)}):")
    # Sort nodes within each cluster for consistent display
    for idx in sorted(list(cluster))[:20]:  # Show up to 20 example queries per cluster
        try:
            print(f" - {queries[idx]}")
        except UnicodeEncodeError:
            # Handle unicode encoding errors by replacing problematic characters
            print(f" - {queries[idx].encode('cp1252', errors='replace').decode('cp1252')}")
    if len(cluster) > 20:
        print("   ...")

# Calculate percentage of queries that have at least one neighbor
nodes_with_neighbors = sum(1 for node in G.nodes() if G.degree(node) > 0)
total_nodes = len(queries)
percentage_with_neighbors = (nodes_with_neighbors / total_nodes) * 100

print(f"\nAnalysis Summary:")
print(f"Total queries: {total_nodes}")
print(f"Queries with at least one neighbor (similarity > {args.threshold}): {nodes_with_neighbors}")
print(f"Percentage of queries with neighbors: {percentage_with_neighbors:.2f}%")
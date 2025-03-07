import os
import pickle
import argparse
import numpy as np
import faiss
from openai import OpenAI
import networkx as nx  # Move import to top
from collections import defaultdict

# Set up argument parser for configurable parameters
parser = argparse.ArgumentParser(description="Cluster similar queries using embeddings and FAISS.")
parser.add_argument("query_file", type=str, help="Path to the text file containing queries, one per line.")
parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold for clustering (default 0.9).")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generating embeddings (default 100).")
parser.add_argument("--top_n", type=int, default=10, help="Number of largest clusters to display (default 10).")
args = parser.parse_args()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to generate embeddings in batches
def generate_embeddings(queries, batch_size=100):
    embeddings = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        response = client.embeddings.create(model="text-embedding-3-large", input=batch)
        batch_embeddings = [embedding.embedding for embedding in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings, dtype=np.float32)

# Load or generate embeddings with caching
cache_file = "embeddings.pkl"
if os.path.exists(cache_file):
    print("Loading cached embeddings...")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    queries = data["queries"]
    embeddings = data["embeddings"]
else:
    print("Generating embeddings...")
    with open(args.query_file, "r") as f:
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
clusters.sort(key=len, reverse=True)

# Display the top N largest clusters
print(f"\nTop {args.top_n} largest clusters (representing the most common questions):")
for i, cluster in enumerate(clusters[:args.top_n], 1):
    print(f"Cluster {i} (size {len(cluster)}):")
    for idx in list(cluster)[:5]:  # Show up to 5 example queries per cluster
        print(f" - {queries[idx]}")
    if len(cluster) > 5:
        print("   ...")
# Query Clustering with Azure OpenAI Embeddings and FAISS

This script processes a text file of queries, generates embeddings using Azure OpenAI's embedding models, stores them in a local FAISS vector database, and clusters similar queries to identify the most common questions. It's designed for large datasets (e.g., 100k+ queries) and uses a graph-based clustering approach to find meaningful clusters without forcing a predefined number of clusters (unlike K-means).

## Setup

### Prerequisites
- **Python 3.7+**: Ensure you have a compatible Python version installed.
- **Azure OpenAI Service**: Access to Azure OpenAI with an appropriate deployment.

### Install Dependencies
Install the required Python packages:
```bash
pip install openai faiss-cpu numpy networkx
```
- **`openai`**: For accessing the Azure OpenAI API to generate embeddings.
- **`faiss-cpu`**: For efficient similarity search and vector storage (use `faiss-gpu` for GPU support if desired).
- **`numpy`**: For numerical operations on embeddings.
- **`networkx`**: For graph-based clustering operations.

### Set Azure OpenAI Environment Variables
Set your Azure OpenAI credentials as environment variables:
```bash
export AZURE_OPENAI_API_KEY='your-api-key'
export AZURE_OPENAI_ENDPOINT='https://your-resource-name.openai.azure.com/'
export AZURE_OPENAI_API_VERSION='2023-05-15'
```
Replace with your actual values. The script will retrieve these automatically.

## Usage

Run the script with the path to your query file (one query per line):
```bash
python cluster_queries.py path/to/queries.txt
```

### Command-Line Parameters
The script accepts several configurable parameters:
- **`query_file`** (required, string): Path to the text file containing queries, one per line.
  - Example: `queries.txt`
- **`--threshold`** (optional, float, default: 0.9): Cosine similarity threshold for clustering. Queries with similarity above this value are grouped together.
  - Range: 0.0 to 1.0
  - Higher values (e.g., 0.95) create tighter, more specific clusters; lower values (e.g., 0.85) allow more variation.
  - Tuning: Start with 0.9 and adjust based on whether clusters are too strict or too loose.
- **`--batch_size`** (optional, int, default: 100): Number of queries processed per API call when generating embeddings.
  - Adjust based on API rate limits or available memory. Larger values reduce API calls but require more memory.
- **`--top_n`** (optional, int, default: 10): Number of largest clusters to display in the output.
  - Set to any positive integer to control how many common question groups you see.
- **`--deployment`** (optional, string, default: "text-embedding-3-large"): Azure OpenAI embeddings deployment name.
  - This should match the name of your embedding model deployment in Azure OpenAI.

Example with custom parameters:
```bash
python cluster_queries.py queries.txt --threshold 0.95 --batch_size 50 --top_n 5 --deployment my-embedding-deployment
```

## How the Script Works

The script performs the following steps to cluster your queries:

1. **Embedding Generation**:
   - **Input**: Reads queries from the specified text file, one per line.
   - **Process**: Uses Azure OpenAI's embedding model to generate embeddings for each query. These embeddings are fixed-length vectors optimized for cosine similarity.
   - **Batching**: Processes queries in batches (controlled by `--batch_size`) to optimize API usage and memory.
   - **Caching**: Saves queries and embeddings to a cache file named after your input file (e.g., `yourfile.embeddings.pkl`). If this file exists, it loads the cached data instead of regenerating embeddings, saving time and API costs on subsequent runs.

2. **Vector Database Setup**:
   - **Tool**: Uses FAISS (Facebook AI Similarity Search) with an `IndexFlatIP` index, which computes inner products (equivalent to cosine similarity for normalized embeddings).
   - **Storage**: Adds all embeddings to the FAISS index and saves it to `faiss_index.bin` for persistence. This allows reuse without rebuilding the index.
   - **Why FAISS?**: FAISS is highly efficient for large-scale similarity searches, avoiding the \( O(n^2) \) complexity of brute-force pairwise comparisons.

3. **Clustering**:
   - **Similarity Search**: Performs a range search with FAISS to find all pairs of queries with cosine similarity above the `--threshold`. This is much faster than computing all pairwise similarities (e.g., 10 billion comparisons for 100k queries).
   - **Graph Construction**: Builds an undirected graph where:
     - Nodes are queries.
     - Edges connect pairs of queries with similarity ≥ `--threshold`.
   - **Cluster Detection**: Uses NetworkX to find connected components in the graph. Each component is a cluster of similar queries, representing different wordings of the same question.
   - **Why This Method?**: 
     - Doesn't require specifying the number of clusters (unlike K-means).
     - Naturally identifies small, meaningful clusters based on similarity.
     - Scales well with large datasets due to approximate similarity search.

4. **Output**:
   - Sorts clusters by size (largest first) to highlight the most common questions.
   - Displays the top `--top_n` clusters, showing up to 20 example queries per cluster (with an ellipsis for larger clusters).
   - Provides summary statistics about the percentage of queries with at least one similar match.

## Output Files
- **`[inputfile].embeddings.pkl`**: A pickled file containing the list of queries and their embeddings. Used for caching to avoid redundant API calls.
- **`faiss_index.bin`**: The FAISS index stored on disk for persistence. You can reload it with `faiss.read_index("faiss_index.bin")` if you extend the script.

## Example Output
```
Top 5 largest clusters (representing the most common questions):
Cluster 1 (size 15):
 - How do I reset my password?
 - Password reset instructions
 - How to change my password
 - Resetting my account password
 - Help with password reset
   ...
Cluster 2 (size 10):
 - What's the weather like today?
 - Current weather forecast
 - Today's weather update
 - How's the weather now?
 - Weather report for today
   ...
...

Analysis Summary:
Total queries: 5000
Queries with at least one neighbor (similarity > 0.9): 3750
Percentage of queries with neighbors: 75.00%
```

## Notes and Tuning Tips
- **Threshold Tuning**: 
  - Default (0.9) balances precision and inclusivity. Increase to 0.95 for stricter clusters (e.g., nearly identical queries) or decrease to 0.85 for broader groupings.
  - Experiment by sampling clusters to ensure they align with your definition of "same question."
- **Batch Size**: 
  - Default (100) works well for most systems. Decrease if you hit memory limits or API errors; increase for faster processing if resources allow.
- **Embedding Quality**: The clustering relies on the embedding model's ability to capture semantic similarity. Test a few query pairs manually to confirm it meets your needs.
- **Azure OpenAI Deployment**: Ensure your deployment name matches what you've set up in the Azure portal.
- **Reusing the Index**: To skip rebuilding the FAISS index, modify the script to load `faiss_index.bin` with `index = faiss.read_index("faiss_index.bin")` instead of creating a new one.
- **Scalability**: For millions of queries, consider FAISS's GPU support or distributed options, though the CPU version should handle 100k+ queries efficiently.

## Why Not Alternatives?
- **K-means**: Requires specifying the number of clusters upfront, which doesn't suit your goal of finding natural groupings.
- **Brute-Force Nearest Neighbors**: Computing top 100 neighbors for each query without a vector database would take \( O(n^2) \) time (e.g., 10 billion calculations for 100k queries), making it impractical.

## Extending the Script
- Add a `--force` flag to regenerate embeddings even if a cache exists.
- Output clusters to a file instead of printing them.
- Filter clusters by minimum size (e.g., ignore clusters with fewer than 3 queries).

This script provides a scalable, efficient solution for clustering queries to uncover the most common questions in your dataset!

#!/usr/bin/env python3
"""
Benchmark script for BM25 and Hybrid Search in Puffer VectorDB.
Measures performance and recall for text, vector, and hybrid search.
"""

import requests
import time
import random
import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import statistics

BASE_URL = "http://127.0.0.1:8080"

# Sample documents for testing (realistic text with semantic meaning)
DOCUMENTS = [
    # Technology
    ("Machine learning algorithms power modern artificial intelligence systems", ["tech", "ai", "ml"]),
    ("Deep neural networks enable complex pattern recognition tasks", ["tech", "ai", "deep-learning"]),
    ("Natural language processing helps computers understand human text", ["tech", "nlp", "ai"]),
    ("Computer vision systems can identify objects in images and videos", ["tech", "cv", "ai"]),
    ("Reinforcement learning trains AI agents through reward signals", ["tech", "rl", "ai"]),
    ("Transformer architectures revolutionized NLP with attention mechanisms", ["tech", "nlp", "transformers"]),
    ("Convolutional neural networks excel at image classification tasks", ["tech", "cnn", "cv"]),
    ("Recurrent neural networks process sequential data like time series", ["tech", "rnn", "deep-learning"]),
    ("Generative adversarial networks create realistic synthetic images", ["tech", "gan", "ai"]),
    ("AutoML automates the process of selecting and tuning ML models", ["tech", "automl", "ml"]),

    # Database
    ("Vector databases store high-dimensional embeddings for similarity search", ["database", "vector", "search"]),
    ("SQL databases use structured query language for data management", ["database", "sql", "relational"]),
    ("NoSQL databases provide flexible schemas for unstructured data", ["database", "nosql", "flexible"]),
    ("Graph databases model relationships between connected entities", ["database", "graph", "relationships"]),
    ("In-memory databases offer ultra-fast data access speeds", ["database", "memory", "performance"]),
    ("Distributed databases scale horizontally across multiple servers", ["database", "distributed", "scale"]),
    ("Time-series databases optimize storage for temporal data patterns", ["database", "timeseries", "iot"]),
    ("Document databases store data as JSON-like documents", ["database", "document", "json"]),
    ("Key-value stores provide simple fast lookups by unique keys", ["database", "keyvalue", "simple"]),
    ("Column-family databases optimize for analytical query workloads", ["database", "columnar", "analytics"]),

    # Search
    ("Full-text search indexes documents for keyword retrieval", ["search", "fulltext", "keywords"]),
    ("Semantic search understands meaning beyond exact word matches", ["search", "semantic", "meaning"]),
    ("BM25 algorithm ranks documents by term frequency relevance", ["search", "bm25", "ranking"]),
    ("Inverted indexes map terms to document locations efficiently", ["search", "index", "inverted"]),
    ("Hybrid search combines vector and keyword search methods", ["search", "hybrid", "combined"]),
    ("Approximate nearest neighbor search finds similar vectors quickly", ["search", "ann", "vectors"]),
    ("Query expansion improves recall by adding related terms", ["search", "expansion", "recall"]),
    ("Faceted search allows filtering results by categories", ["search", "faceted", "filter"]),
    ("Fuzzy matching finds results despite spelling variations", ["search", "fuzzy", "typos"]),
    ("Relevance tuning adjusts search ranking for better results", ["search", "relevance", "tuning"]),

    # Programming
    ("Python programming language is popular for data science", ["programming", "python", "data"]),
    ("Rust provides memory safety without garbage collection overhead", ["programming", "rust", "safety"]),
    ("JavaScript runs in browsers and powers web applications", ["programming", "javascript", "web"]),
    ("Go language excels at building concurrent server applications", ["programming", "golang", "concurrency"]),
    ("TypeScript adds static typing to JavaScript development", ["programming", "typescript", "types"]),
    ("C++ offers low-level control for performance-critical code", ["programming", "cpp", "performance"]),
    ("Java virtual machine enables cross-platform application deployment", ["programming", "java", "jvm"]),
    ("Functional programming emphasizes immutable data transformations", ["programming", "functional", "immutable"]),
    ("Object-oriented design organizes code into reusable classes", ["programming", "oop", "classes"]),
    ("Microservices architecture splits applications into small services", ["programming", "microservices", "architecture"]),

    # Cloud
    ("Cloud computing provides on-demand scalable infrastructure", ["cloud", "infrastructure", "scale"]),
    ("Kubernetes orchestrates containerized application deployments", ["cloud", "kubernetes", "containers"]),
    ("Serverless functions execute code without managing servers", ["cloud", "serverless", "functions"]),
    ("Container images package applications with all dependencies", ["cloud", "docker", "containers"]),
    ("Load balancers distribute traffic across multiple servers", ["cloud", "loadbalancer", "traffic"]),
    ("CDN networks cache content closer to end users globally", ["cloud", "cdn", "caching"]),
    ("Infrastructure as code automates cloud resource provisioning", ["cloud", "iac", "automation"]),
    ("Service mesh manages communication between microservices", ["cloud", "servicemesh", "networking"]),
    ("Auto-scaling adjusts resources based on demand patterns", ["cloud", "autoscaling", "elastic"]),
    ("Multi-cloud strategies avoid vendor lock-in across providers", ["cloud", "multicloud", "strategy"]),
]

# Generate more documents by combining patterns
def generate_documents(n: int) -> List[Tuple[str, List[str]]]:
    """Generate n documents with realistic text and tags."""
    docs = list(DOCUMENTS)

    adjectives = ["advanced", "modern", "efficient", "powerful", "innovative", "robust", "scalable", "reliable"]
    verbs = ["enables", "provides", "supports", "implements", "delivers", "offers", "facilitates", "achieves"]
    domains = ["enterprise", "startup", "research", "production", "development", "testing", "deployment", "monitoring"]

    while len(docs) < n:
        base = random.choice(DOCUMENTS)
        adj = random.choice(adjectives)
        verb = random.choice(verbs)
        domain = random.choice(domains)

        # Create variation
        words = base[0].split()
        new_text = f"{adj.capitalize()} {domain} systems {verb} {' '.join(words[2:])}"
        new_tags = base[1] + [domain]
        docs.append((new_text, new_tags))

    return docs[:n]

def generate_random_vector(dim: int) -> List[float]:
    """Generate a random unit vector."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x*x for x in vec) ** 0.5
    return [x / norm for x in vec]

def create_collection(name: str, dim: int) -> bool:
    """Create a new collection."""
    try:
        # Delete if exists
        requests.delete(f"{BASE_URL}/v1/collections/{name}")
    except:
        pass

    resp = requests.post(f"{BASE_URL}/v1/collections", json={
        "name": name,
        "dimension": dim,
        "metric": "cosine"
    })
    return resp.status_code == 201

def insert_points(collection: str, points: List[Dict]) -> int:
    """Insert points into collection."""
    resp = requests.post(f"{BASE_URL}/v1/collections/{collection}/points", json={
        "points": points
    })
    if resp.status_code == 200:
        return resp.json().get("inserted", 0)
    return 0

def add_text_documents(collection: str, docs: List[Dict], retry: int = 3) -> int:
    """Add text documents to FTS index."""
    for attempt in range(retry):
        resp = requests.post(f"{BASE_URL}/v1/collections/{collection}/text-documents", json={
            "documents": docs
        })
        if resp.status_code == 200:
            return resp.json().get("indexed", 0)
        if attempt < retry - 1:
            time.sleep(0.5)  # Wait before retry
    print(f"Error adding docs: {resp.status_code} - {resp.text}")
    return 0

def search_vector(collection: str, vector: List[float], top_k: int = 10) -> Tuple[List[str], float]:
    """Vector search, returns (ids, latency_ms)."""
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/collections/{collection}/search", json={
        "vector": vector,
        "top_k": top_k,
        "nprobe": 4
    })
    latency = (time.time() - start) * 1000

    if resp.status_code == 200:
        results = resp.json().get("results", [])
        return [r["id"] for r in results], latency
    return [], latency

def search_bm25(collection: str, query: str, top_k: int = 10) -> Tuple[List[str], float]:
    """BM25 text search, returns (ids, latency_ms)."""
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/collections/{collection}/text-search", json={
        "query": query,
        "top_k": top_k
    })
    latency = (time.time() - start) * 1000

    if resp.status_code == 200:
        results = resp.json().get("results", [])
        return [r["id"] for r in results], latency
    print(f"BM25 error: {resp.status_code} - {resp.text}")
    return [], latency

def search_hybrid(collection: str, query: str, vector: List[float],
                  top_k: int = 10, lambda_val: float = 0.5,
                  fusion: str = "weighted") -> Tuple[List[str], float]:
    """Hybrid search, returns (ids, latency_ms)."""
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/collections/{collection}/hybrid-search", json={
        "text_query": query,
        "vector": vector,
        "top_k": top_k,
        "lambda": lambda_val,
        "fusion_method": fusion,
        "candidates_per_source": 100
    })
    latency = (time.time() - start) * 1000

    if resp.status_code == 200:
        results = resp.json().get("results", [])
        return [r["id"] for r in results], latency
    print(f"Hybrid error: {resp.status_code} - {resp.text}")
    return [], latency

def compute_recall(retrieved: List[str], ground_truth: List[str]) -> float:
    """Compute recall@k."""
    if not ground_truth:
        return 0.0
    hits = len(set(retrieved) & set(ground_truth))
    return hits / len(ground_truth)

def compute_mrr(retrieved: List[str], relevant: str) -> float:
    """Compute Mean Reciprocal Rank."""
    try:
        rank = retrieved.index(relevant) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0

@dataclass
class BenchmarkResult:
    name: str
    latencies: List[float]
    recalls: List[float]
    mrrs: List[float]

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "latency_p50_ms": statistics.median(self.latencies) if self.latencies else 0,
            "latency_p95_ms": np.percentile(self.latencies, 95) if self.latencies else 0,
            "latency_mean_ms": statistics.mean(self.latencies) if self.latencies else 0,
            "recall_mean": statistics.mean(self.recalls) if self.recalls else 0,
            "mrr_mean": statistics.mean(self.mrrs) if self.mrrs else 0,
            "qps": 1000 / statistics.mean(self.latencies) if self.latencies else 0,
        }

def run_benchmark(num_docs: int = 10000, dim: int = 128, num_queries: int = 100):
    """Run the full benchmark suite."""
    collection = "bench_hybrid"

    print(f"\n{'='*60}")
    print(f"PUFFER HYBRID SEARCH BENCHMARK")
    print(f"{'='*60}")
    print(f"Documents: {num_docs}")
    print(f"Dimension: {dim}")
    print(f"Queries: {num_queries}")
    print(f"{'='*60}\n")

    # Create collection
    print("Creating collection...")
    if not create_collection(collection, dim):
        print("Failed to create collection")
        return

    # Generate documents
    print(f"Generating {num_docs} documents...")
    documents = generate_documents(num_docs)

    # Create vectors - use text hash as seed for reproducibility
    # This simulates embeddings that are semantically related to text
    print("Generating vectors (simulated embeddings)...")
    doc_vectors = {}
    for i, (text, tags) in enumerate(documents):
        # Create semi-realistic embeddings based on content
        seed = hash(text) % (2**32)
        random.seed(seed)
        vec = generate_random_vector(dim)
        doc_vectors[f"doc_{i}"] = {
            "text": text,
            "tags": tags,
            "vector": vec
        }

    # Insert vectors and text in batches
    print("Inserting vectors...")
    batch_size = 1000
    total_inserted = 0

    for i in range(0, num_docs, batch_size):
        batch_end = min(i + batch_size, num_docs)
        points = []
        for j in range(i, batch_end):
            doc_id = f"doc_{j}"
            points.append({
                "id": doc_id,
                "vector": doc_vectors[doc_id]["vector"]
            })
        inserted = insert_points(collection, points)
        total_inserted += inserted
        print(f"  Inserted {total_inserted}/{num_docs} vectors...")

    # Flush to ensure vectors are indexed
    print("Flushing vectors...")
    requests.post(f"{BASE_URL}/v1/collections/{collection}/flush")
    time.sleep(1)

    # Insert text documents
    print("Indexing text documents...")
    total_indexed = 0
    for i in range(0, num_docs, batch_size):
        batch_end = min(i + batch_size, num_docs)
        docs = []
        for j in range(i, batch_end):
            doc_id = f"doc_{j}"
            docs.append({
                "vector_id": doc_id,
                "text": doc_vectors[doc_id]["text"],
                "tags": doc_vectors[doc_id]["tags"]
            })
        indexed = add_text_documents(collection, docs)
        total_indexed += indexed
        print(f"  Indexed {total_indexed}/{num_docs} documents...")

    print(f"\nTotal: {total_inserted} vectors, {total_indexed} documents indexed")

    # Create test queries
    print(f"\nGenerating {num_queries} test queries...")
    test_queries = []

    # Sample some documents as queries (self-retrieval test)
    query_indices = random.sample(range(num_docs), min(num_queries, num_docs))

    for idx in query_indices:
        doc_id = f"doc_{idx}"
        text = doc_vectors[doc_id]["text"]
        vector = doc_vectors[doc_id]["vector"]

        # Extract 2-3 keywords from text for BM25 query
        words = text.lower().split()
        query_words = random.sample(words, min(3, len(words)))
        query_text = " ".join(query_words)

        test_queries.append({
            "doc_id": doc_id,
            "text_query": query_text,
            "vector": vector,
            "full_text": text
        })

    # Benchmark Vector Search
    print("\n--- Benchmarking Vector Search ---")
    vector_result = BenchmarkResult("Vector Search", [], [], [])

    for q in test_queries:
        ids, latency = search_vector(collection, q["vector"], top_k=10)
        vector_result.latencies.append(latency)

        # Recall: did we find the original document?
        recall = 1.0 if q["doc_id"] in ids else 0.0
        vector_result.recalls.append(recall)

        # MRR: what rank is the original document?
        mrr = compute_mrr(ids, q["doc_id"])
        vector_result.mrrs.append(mrr)

    # Benchmark BM25 Search
    print("--- Benchmarking BM25 Search ---")
    bm25_result = BenchmarkResult("BM25 Search", [], [], [])

    for q in test_queries:
        ids, latency = search_bm25(collection, q["text_query"], top_k=10)
        bm25_result.latencies.append(latency)

        recall = 1.0 if q["doc_id"] in ids else 0.0
        bm25_result.recalls.append(recall)

        mrr = compute_mrr(ids, q["doc_id"])
        bm25_result.mrrs.append(mrr)

    # Benchmark Hybrid Search (different fusion methods)
    fusion_methods = ["weighted", "rrf", "normalized"]
    hybrid_results = []

    for fusion in fusion_methods:
        print(f"--- Benchmarking Hybrid Search ({fusion}) ---")
        result = BenchmarkResult(f"Hybrid ({fusion})", [], [], [])

        for q in test_queries:
            ids, latency = search_hybrid(
                collection, q["text_query"], q["vector"],
                top_k=10, lambda_val=0.5, fusion=fusion
            )
            result.latencies.append(latency)

            recall = 1.0 if q["doc_id"] in ids else 0.0
            result.recalls.append(recall)

            mrr = compute_mrr(ids, q["doc_id"])
            result.mrrs.append(mrr)

        hybrid_results.append(result)

    # Print Results
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Method':<25} {'P50 (ms)':<12} {'P95 (ms)':<12} {'QPS':<10} {'Recall@10':<12} {'MRR':<10}")
    print("-" * 80)

    all_results = [vector_result, bm25_result] + hybrid_results

    for r in all_results:
        s = r.summary()
        print(f"{s['name']:<25} {s['latency_p50_ms']:<12.2f} {s['latency_p95_ms']:<12.2f} "
              f"{s['qps']:<10.1f} {s['recall_mean']:<12.3f} {s['mrr_mean']:<10.3f}")

    print(f"\n{'='*80}")

    # Lambda sweep for hybrid
    print("\nHYBRID SEARCH LAMBDA SWEEP (text weight)")
    print("-" * 60)
    print(f"{'Lambda':<10} {'Recall@10':<15} {'MRR':<15} {'Latency P50':<15}")
    print("-" * 60)

    for lambda_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        recalls = []
        mrrs = []
        latencies = []

        for q in test_queries[:50]:  # Use subset for sweep
            ids, latency = search_hybrid(
                collection, q["text_query"], q["vector"],
                top_k=10, lambda_val=lambda_val, fusion="weighted"
            )
            latencies.append(latency)
            recalls.append(1.0 if q["doc_id"] in ids else 0.0)
            mrrs.append(compute_mrr(ids, q["doc_id"]))

        print(f"{lambda_val:<10.2f} {statistics.mean(recalls):<15.3f} "
              f"{statistics.mean(mrrs):<15.3f} {statistics.median(latencies):<15.2f}")

    print(f"\n{'='*80}")
    print("Benchmark complete!")

    # Cleanup
    print("\nCleaning up...")
    requests.delete(f"{BASE_URL}/v1/collections/{collection}")

if __name__ == "__main__":
    import sys

    num_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    num_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    run_benchmark(num_docs, dim, num_queries)

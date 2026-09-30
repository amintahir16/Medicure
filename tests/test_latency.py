"""
Response Latency & Profiling Benchmark for Medicure MBBS Engine
Measures latency across:
1. Local PubMedBERT Dense Embedding Generation
2. SQLite FTS5 BM25 Lexical Retrieval
3. SQLite-vec / NumPy Dense Vector KNN
4. Reciprocal Rank Fusion (RRF) & Clinical Intent Scoring
5. Offline Response Generation
6. Full API Roundtrip (Offline and Groq)
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import httpx
import asyncio
from backend.engine.retriever import MBBSHybridRetriever
from backend.engine.generator import MBBSGenerator

def benchmark_pipeline_breakdown():
    print("=" * 65)
    print("  MEDICURE INTERNAL PIPELINE LATENCY PROFILING")
    print("=" * 65)

    retriever = MBBSHybridRetriever()
    generator = MBBSGenerator(retriever=retriever)

    query = "What are sesamoid bones and what are their clinical advantages?"

    # 1. Warm-up embedder
    t0 = time.perf_counter()
    retriever.embedder.embed_query("warmup")
    t_warm = time.perf_counter() - t0
    print(f"[*] Embedding Engine Warmup: {t_warm:.3f}s")

    # 2. Query Embedding Latency
    t0 = time.perf_counter()
    q_emb = retriever.embedder.embed_query(query)
    t_emb = time.perf_counter() - t0
    print(f"[1] PubMedBERT Dense Query Embedding (768d, CPU): {t_emb:.3f}s")

    # 3. BM25 Search Latency
    t0 = time.perf_counter()
    clean_fts = retriever._clean_fts_query(query)
    bm25_matches = retriever._search_sparse_bm25(clean_fts)
    t_bm25 = time.perf_counter() - t0
    print(f"[2] SQLite FTS5 BM25 Lexical Search: {t_bm25:.3f}s ({len(bm25_matches)} candidates)")

    # 4. Dense Vector Search Latency
    t0 = time.perf_counter()
    dense_matches = retriever._search_dense_vectors(query)
    t_dense = time.perf_counter() - t0
    print(f"[3] Dense Vector KNN / Cosine Similarity: {t_dense:.3f}s ({len(dense_matches)} candidates)")

    # 5. Full Hybrid RRF Search Latency
    t0 = time.perf_counter()
    evidence = retriever.search(query, top_k=5)
    t_rrf = time.perf_counter() - t0
    print(f"[4] Full Hybrid RRF Fusion + Subject Intent Routing: {t_rrf:.3f}s ({len(evidence)} evidence passages)")

    # 6. Offline Response Synthesis Latency
    t0 = time.perf_counter()
    offline_res = asyncio.run(generator.generate_response(query, provider="offline"))
    t_offline = time.perf_counter() - t0
    print(f"[5] Offline Grounded Synthesis (Table Matrix + Pearl): {t_offline:.3f}s")

    print(f"--> Total In-Memory Offline Pipeline: {t_rrf + t_offline:.3f}s")
    print("-" * 65)

def benchmark_http_api():
    print("\n" * 1 + "=" * 65)
    print("  HTTP API ENDPOINT BENCHMARK (http://127.0.0.1:8000/api/chat)")
    print("=" * 65)

    test_queries = [
        ("Sesamoid Bones", "What are sesamoid bones and their two main clinical advantages?"),
        ("Hilton's Law", "State Hilton's Law regarding the nerve supply of joints."),
        ("Bone Vascularity", "Explain the vascular supply of a growing long bone and epiphyseal cartilage.")
    ]

    client = httpx.Client(timeout=30.0)

    for name, q in test_queries:
        t0 = time.perf_counter()
        resp = client.post("http://127.0.0.1:8000/api/chat", json={
            "query": q,
            "provider": "offline"
        })
        elapsed = time.perf_counter() - t0
        data = resp.json()
        print(f"[OFFLINE] {name:18} -> {elapsed:.3f}s (Status: {resp.status_code}, Citations: {len(data.get('citations', []))})")

    print("=" * 65)

if __name__ == "__main__":
    benchmark_pipeline_breakdown()
    benchmark_http_api()

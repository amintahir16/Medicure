"""
Automated Test Suite for Medicure MBBS AI Engine
Validates:
1. Local FastEmbed embedding engine and 384-dimensional dense vectors.
2. Multi-year textbook indexing integrity with dense vectors in SQLite.
3. Hybrid retrieval precision: Dense vector KNN + Sparse BM25 + Reciprocal Rank Fusion (RRF).
4. Grounded generation and exact citation formatting.
5. Page inspector data retrieval.
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.engine.embeddings import MBBSEmbeddingEngine, EMBEDDING_DIM
from backend.engine.indexer import MBBSIndexer
from backend.engine.retriever import MBBSHybridRetriever
from backend.engine.generator import MBBSGenerator

def test_dense_embeddings_engine():
    embedder = MBBSEmbeddingEngine()
    assert embedder.dim == 384, f"Expected 384 dimensions, got {embedder.dim}"

    # Query embedding test
    q_vec = embedder.embed_query("fibular neck fracture common peroneal nerve foot drop")
    assert isinstance(q_vec, np.ndarray)
    assert q_vec.shape == (384,)
    assert not np.all(q_vec == 0)

    # Document batch embedding test
    docs = [
        "Common peroneal nerve winds around the neck of the fibula.",
        "Loss of dorsiflexion results in foot drop during the swing phase of gait."
    ]
    doc_vecs = embedder.embed_documents(docs)
    assert len(doc_vecs) == 2
    assert doc_vecs[0].shape == (384,)
    assert doc_vecs[1].shape == (384,)
    print("[PASS] Local FastEmbed 384-dim embedding engine verified.")

def test_textbook_indexing_and_vector_count():
    indexer = MBBSIndexer()
    summary = indexer.get_books_summary()
    assert len(summary) >= 1, f"Expected at least 1 MBBS textbook, found {len(summary)}"
    
    titles = [b["book_title"] for b in summary]
    assert any("Chaurasia" in t for t in titles), f"BD Chaurasia not found in summary: {titles}"

    # Verify that all medical_records have dense vectors
    with indexer.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) as cnt FROM medical_records WHERE embedding IS NOT NULL")
        embedded_cnt = cursor.fetchone()["cnt"]
        cursor.execute("SELECT count(*) as cnt FROM medical_records")
        total_cnt = cursor.fetchone()["cnt"]
        assert embedded_cnt == total_cnt and total_cnt > 0, f"Expected all {total_cnt} records to have embeddings, found {embedded_cnt}"

        # Verify medical_vectors table
        cursor.execute("SELECT count(*) as cnt FROM medical_vectors")
        vec_cnt = cursor.fetchone()["cnt"]
        assert vec_cnt == total_cnt, f"Expected {total_cnt} in medical_vectors, found {vec_cnt}"

    print(f"[PASS] Real MBBS textbook indexing with {embedded_cnt} dense vectors verified.")

def test_hybrid_dense_sparse_retrieval():
    retriever = MBBSHybridRetriever()
    
    # Test 1: Exact keyword query
    bone_results = retriever.search("classification of bones according to shape diaphysis epiphysis", top_k=3)
    assert len(bone_results) > 0, "Failed to retrieve bone classification from BD Chaurasia"
    top_bone = bone_results[0]
    assert "Anatomy" in top_bone["subject"]
    assert any("Chaurasia" in b["book_title"] for b in bone_results)
    assert top_bone["rrf_score"] > 0
    assert "dense_similarity" in top_bone

    # Test 2: Natural language synonym / paraphrased query
    synonym_results = retriever.search("stretching or tearing of ligament fibers without dislocation", top_k=3)
    assert len(synonym_results) > 0, "Failed to retrieve ligament anatomy via hybrid search"
    top_synonym = synonym_results[0]
    assert "Anatomy" in top_synonym["subject"]
    assert top_synonym["dense_similarity"] > 0
    assert top_synonym["page_number"] > 0

    print("[PASS] Hybrid Dense Vector + Sparse BM25 RRF retrieval verified across exact and synonym queries.")

def test_page_content_retrieval():
    indexer = MBBSIndexer()
    books = indexer.get_books_summary()
    assert len(books) > 0
    first_book = books[0]["book_title"]
    
    page_data = indexer.get_page_content(first_book, 44)
    assert page_data, f"Failed to retrieve page 44 for {first_book}"
    assert "content" in page_data
    assert len(page_data["content"]) > 50
    assert page_data["page_number"] == 44
    assert "BONES" in page_data["content"] or "CLASSIFICATION" in page_data["content"]
    print("[PASS] Real page content retrieval verified.")

def test_grounded_generator_response():
    generator = MBBSGenerator()
    result = asyncio.run(generator.generate_response(
        query="What is the classification of bones according to shape in anatomy?",
        provider="offline"
    ))
    
    assert "answer" in result
    assert "citations" in result
    assert len(result["citations"]) > 0
    
    # Check that answer contains clean inline reference and clinical content
    assert "[Ref 1]" in result["answer"]
    assert any(term in result["answer"].lower() for term in ["bone", "long", "shape", "epiphysis", "shaft"])
    
    # Check citation contents
    first_cit = result["citations"][0]
    assert "Anatomy" in first_cit["subject"]
    assert "Chaurasia" in first_cit["book_title"]
    print("[PASS] Grounded response generation with real textbook citations verified.")

if __name__ == "__main__":
    test_dense_embeddings_engine()
    test_textbook_indexing_and_vector_count()
    test_hybrid_dense_sparse_retrieval()
    test_page_content_retrieval()
    test_grounded_generator_response()
    print("\n" + "=" * 60)
    print("  ALL HYBRID DENSE-SPARSE RAG TESTS PASSED 100%!")
    print("=" * 60)

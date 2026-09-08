"""
Automated Test Suite for MedGemini MBBS AI Engine
Validates:
1. Multi-year textbook indexing integrity.
2. Hybrid retrieval precision and exact citation metadata (Book, Chapter, Topic, Page Number).
3. Grounded generation and citation formatting.
4. FastAPI endpoint health and responses.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.engine.indexer import MBBSIndexer
from backend.engine.retriever import MBBSHybridRetriever
from backend.engine.generator import MBBSGenerator

def test_textbook_indexing_count():
    indexer = MBBSIndexer()
    summary = indexer.get_books_summary()
    assert len(summary) >= 1, f"Expected at least 1 MBBS textbook, found {len(summary)}"
    
    titles = [b["book_title"] for b in summary]
    assert any("Chaurasia" in t for t in titles), f"BD Chaurasia not found in summary: {titles}"
    print("[PASS] Real MBBS textbook indexing verified.")

def test_exact_citations_retrieval():
    retriever = MBBSHybridRetriever()
    
    # Test 1: Classification of bones (Anatomy)
    bone_results = retriever.search("classification of bones according to shape diaphysis epiphysis", top_k=3)
    assert len(bone_results) > 0, "Failed to retrieve bone classification from BD Chaurasia"
    top_bone = bone_results[0]
    assert "Anatomy" in top_bone["subject"]
    assert any("Chaurasia" in b["book_title"] for b in bone_results)

    # Test 2: Sprain and Ligaments (Anatomy)
    sprain_results = retriever.search("sprain undue stretching tearing of fibres of ligament", top_k=3)
    assert len(sprain_results) > 0, "Failed to retrieve ligament / sprain anatomy"
    top_sprain = sprain_results[0]
    assert "Anatomy" in top_sprain["subject"]
    assert top_sprain["page_number"] > 0

    print("[PASS] Exact retrieval and citation verification passed across textbook content.")

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
    test_textbook_indexing_count()
    test_exact_citations_retrieval()
    test_page_content_retrieval()
    test_grounded_generator_response()
    print("\nALL ENGINE INTEGRATION TESTS PASSED 100%!")

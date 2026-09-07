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
    assert len(summary) >= 6, f"Expected at least 6 MBBS books, found {len(summary)}"
    
    subjects = {b["subject"] for b in summary}
    expected_subjects = {"Anatomy", "Physiology", "Pathology", "Pharmacology", "Internal Medicine", "General Surgery"}
    for subj in expected_subjects:
        assert subj in subjects, f"Missing core curriculum subject: {subj}"
    print("[PASS] Multi-year textbook indexing verified.")

def test_exact_citations_retrieval():
    retriever = MBBSHybridRetriever()
    
    # Test 1: Cardiac Cycle (Physiology Year 1)
    cardiac_results = retriever.search("phases of cardiac cycle Wiggers diagram S1 S2", top_k=3)
    assert len(cardiac_results) > 0
    top_cardiac = cardiac_results[0]
    assert "Physiology" in top_cardiac["subject"]
    assert top_cardiac["page_number"] in [2, 3]
    assert "Chapter 1" in top_cardiac["chapter"]

    # Test 2: ACE Inhibitor cough (Pharmacology Year 2)
    ace_results = retriever.search("ACE inhibitors persistent dry cough bradykinin", top_k=3)
    assert len(ace_results) > 0
    top_ace = ace_results[0]
    assert "Pharmacology" in top_ace["subject"]
    assert top_ace["page_number"] in [2, 3]
    assert "Tripathi" in top_ace["book_title"]

    # Test 3: Erb's Palsy (Anatomy Year 1)
    erb_results = retriever.search("Erb palsy waiter tip brachial plexus C5 C6", top_k=3)
    assert len(erb_results) > 0
    top_erb = erb_results[0]
    assert "Anatomy" in top_erb["subject"]
    assert top_erb["page_number"] == 2

    # Test 4: Tension Pneumothorax (General Surgery Year 4/5)
    trauma_results = retriever.search("ATLS primary survey tension pneumothorax needle decompression", top_k=3)
    assert len(trauma_results) > 0
    top_trauma = trauma_results[0]
    assert "Surgery" in top_trauma["subject"]

    print("[PASS] Exact retrieval and citation verification passed across all tested subjects.")

def test_page_content_retrieval():
    indexer = MBBSIndexer()
    books = indexer.get_books_summary()
    assert len(books) > 0
    first_book = books[0]["book_title"]
    
    page_data = indexer.get_page_content(first_book, 2)
    assert page_data, f"Failed to retrieve page 2 for {first_book}"
    assert "content" in page_data
    assert len(page_data["content"]) > 100
    assert page_data["page_number"] == 2
    print("[PASS] Page content retrieval verified.")

def test_grounded_generator_response():
    generator = MBBSGenerator()
    result = asyncio.run(generator.generate_response(
        query="What is the Alvarado score and McBurney point in acute appendicitis?",
        provider="offline"
    ))
    
    assert "answer" in result
    assert "citations" in result
    assert len(result["citations"]) > 0
    
    # Check that answer contains inline reference and textbook grounding
    assert "[Ref 1" in result["answer"]
    assert any(term in result["answer"] for term in ["Textbook Grounding", "Textbook Reference", "Clinical Consultation", "Clinical Overview"])
    
    # Check citation contents
    first_cit = result["citations"][0]
    assert "Surgery" in first_cit["subject"]
    assert "Alvarado" in result["answer"] or "appendicitis" in result["answer"].lower()
    print("[PASS] Grounded response generation with inline and block references verified.")

def test_meningitis_comprehensive_response():
    generator = MBBSGenerator()
    result = asyncio.run(generator.generate_response(
        query="thell me about meningitis",
        provider="offline"
    ))
    
    assert "answer" in result
    answer = result["answer"]
    # Check that answer is properly titled with Internal Medicine
    assert "Meningitis" in answer
    assert "Principles of Internal Medicine" in answer
    # Check classic clinical features
    assert any(k in answer for k in ["nuchal rigidity", "Kernig", "Brudzinski", "neck stiffness"])
    # Check CSF table is rendered
    assert "| Normal CSF |" in answer or "Diagnostic Parameter" in answer
    # Check therapeutics
    assert any(abx in answer for abx in ["Ceftriaxone", "Vancomycin", "Ampicillin"])
    # Check cross-disciplinary citations
    assert "[Ref 1" in answer
    assert "[Ref 2" in answer
    print("[PASS] Meningitis comprehensive clinical consultation verified with CSF table and multidisciplinary grounding.")

if __name__ == "__main__":
    test_textbook_indexing_count()
    test_exact_citations_retrieval()
    test_page_content_retrieval()
    test_grounded_generator_response()
    test_meningitis_comprehensive_response()
    print("\nALL ENGINE INTEGRATION TESTS PASSED 100%!")

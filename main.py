"""
Entry Point for MedGemini MBBS AI Chatbot
Ensures database and books are seeded, then launches Uvicorn server.
"""

import sys
import os
import uvicorn
from pathlib import Path

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.seed.book_builder import build_all_mbbs_books
from backend.engine.indexer import MBBSIndexer

def main():
    print("=" * 65)
    print("  MEDGEMINI MBBS CLINICAL CONSULTATION CHATBOT")
    print("  5-Year Curriculum Knowledge Base with Exact Grounding Citations")
    print("=" * 65)

    books_dir = ROOT_DIR / "books"
    books = list(books_dir.glob("*.pdf")) if books_dir.exists() else []

    if len(books) < 6:
        print("[*] Generating 6 foundational MBBS curriculum textbooks...")
        build_all_mbbs_books()

    indexer = MBBSIndexer()
    summary = indexer.get_books_summary()
    if len(summary) < 6:
        print("[*] Indexing textbooks into SQLite FTS5 database...")
        indexer.index_all_books()
        summary = indexer.get_books_summary()

    print(f"[*] Knowledge base ready: {len(summary)} MBBS textbooks indexed.")
    for b in summary:
        print(f"    - {b['book_title']} ({b['subject']} • {b['total_pages']} pages)")

    print("\n[*] Starting MedGemini server at: http://127.0.0.1:8000")
    print("=" * 65)

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()

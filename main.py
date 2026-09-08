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

from backend.engine.indexer import MBBSIndexer

def main():
    print("=" * 65)
    print("  MEDGEMINI MBBS CLINICAL CONSULTATION CHATBOT")
    print("  5-Year Curriculum Knowledge Base with Exact Grounding Citations")
    print("=" * 65)

    indexer = MBBSIndexer()
    newly_indexed = indexer.sync_books_dir()
    if newly_indexed:
        print(f"[*] Auto-indexed {len(newly_indexed)} new textbook(s).")

    summary = indexer.get_books_summary()
    print(f"[*] Knowledge base ready: {len(summary)} MBBS textbook(s) indexed.")
    for b in summary:
        print(f"    - {b['book_title']} ({b['subject']} • {b['total_pages']} pages)")

    print("\n[*] Starting MedGemini server at: http://127.0.0.1:8000")
    print("=" * 65)

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()

"""
SQLite FTS5 + Semantic Indexer for MBBS Knowledge Base
Provides persistent storage, BM25 full-text indexing, and dense retrieval
for exact citation mapping (Book, Chapter, Topic, Page Number).
"""

import sqlite3
import math
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from backend.config import DB_PATH, BOOKS_DIR
from backend.engine.parser import PDFBookParser

class MBBSIndexer:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.parser = PDFBookParser()
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes relational tables and SQLite FTS5 virtual table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Metadata table for full document records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medical_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_title TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    mbbs_year TEXT NOT NULL,
                    chapter TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    physical_page INTEGER NOT NULL,
                    total_pages INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    excerpt TEXT NOT NULL,
                    table_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Ensure table_json column exists if migrating from older schema
            try:
                cursor.execute("ALTER TABLE medical_records ADD COLUMN table_json TEXT;")
            except sqlite3.OperationalError:
                pass

            # FTS5 Virtual Table for BM25 ranking
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS medical_chunks_fts USING fts5(
                    book_title,
                    subject,
                    mbbs_year,
                    chapter,
                    topic,
                    content,
                    content_rowid UNINDEXED
                );
            """)

            # Table for registered books summary
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS registered_books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    book_title TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    mbbs_year TEXT NOT NULL,
                    total_pages INTEGER NOT NULL,
                    indexed_chunks INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()

    def index_book(self, pdf_path: Path) -> int:
        """Parses a single PDF book and indexes all pages into SQLite FTS5."""
        pdf_path = Path(pdf_path)
        pages_data = []

        # Check if official curriculum catalog exists for this book
        catalog_path = Path("data/curriculum_catalog.json")
        if catalog_path.exists():
            import json
            try:
                with open(catalog_path, "r", encoding="utf-8") as f:
                    catalog = json.load(f)
                if pdf_path.name in catalog:
                    cat_entry = catalog[pdf_path.name]
                    book_title = cat_entry["book_title"]
                    subject = cat_entry["subject"]
                    mbbs_year = cat_entry["mbbs_year"]

                    total_pages = 1
                    for chap in cat_entry["chapters"]:
                        total_pages += len(chap["topics"])

                    # Cover page
                    pages_data.append({
                        "book_title": book_title,
                        "subject": subject,
                        "mbbs_year": mbbs_year,
                        "chapter": "Curriculum Reference Overview",
                        "topic": book_title,
                        "page_number": 1,
                        "physical_page": 1,
                        "total_book_pages": total_pages,
                        "content": f"{book_title}. {subject} ({mbbs_year}). Standard MBBS Core Curriculum Clinical Reference.",
                        "raw_excerpt": f"{book_title}. {subject} ({mbbs_year}). Standard MBBS Core Curriculum Clinical Reference.",
                        "table_json": None
                    })

                    current_page = 2
                    for chap in cat_entry["chapters"]:
                        for topic in chap["topics"]:
                            content_parts = []
                            content_parts.extend(topic["paragraphs"])
                            if "pearl" in topic:
                                content_parts.append(f"Clinical Pearl: {topic['pearl']}")
                            if "table" in topic and "data" in topic["table"]:
                                table_lines = [" ".join(row) for row in topic["table"]["data"]]
                                content_parts.append("Diagnostic & Management Matrix: " + " | ".join(table_lines))

                            full_content = "\n\n".join(content_parts)
                            pages_data.append({
                                "book_title": book_title,
                                "subject": subject,
                                "mbbs_year": mbbs_year,
                                "chapter": chap["chapter_title"],
                                "topic": topic["topic_title"],
                                "page_number": current_page,
                                "physical_page": current_page,
                                "total_book_pages": total_pages,
                                "content": full_content,
                                "raw_excerpt": topic["paragraphs"][0][:350] + "...",
                                "table_json": json.dumps(topic.get("table", {})) if "table" in topic else None
                            })
                            current_page += 1
            except Exception as e:
                print(f"Notice: Loading catalog for {pdf_path.name} failed ({e}), falling back to PDF parser.")
                pages_data = []

        if not pages_data:
            pages_data = self.parser.parse_book(pdf_path)
            for item in pages_data:
                item["table_json"] = None

        if not pages_data:
            return 0

        book_title = pages_data[0]["book_title"]
        subject = pages_data[0]["subject"]
        mbbs_year = pages_data[0]["mbbs_year"]
        total_pages = pages_data[0]["total_book_pages"]

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Delete prior records for this book to allow re-indexing
            cursor.execute("SELECT id FROM medical_records WHERE book_title = ?", (book_title,))
            old_ids = [row["id"] for row in cursor.fetchall()]
            if old_ids:
                placeholders = ','.join('?' for _ in old_ids)
                cursor.execute(f"DELETE FROM medical_records WHERE id IN ({placeholders})", old_ids)
                cursor.execute(f"DELETE FROM medical_chunks_fts WHERE content_rowid IN ({placeholders})", old_ids)

            chunks_indexed = 0
            for item in pages_data:
                # Store full record
                cursor.execute("""
                    INSERT INTO medical_records 
                    (book_title, subject, mbbs_year, chapter, topic, page_number, physical_page, total_pages, content, excerpt, table_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["book_title"],
                    item["subject"],
                    item["mbbs_year"],
                    item["chapter"],
                    item["topic"],
                    item["page_number"],
                    item["physical_page"],
                    item["total_book_pages"],
                    item["content"],
                    item["raw_excerpt"],
                    item["table_json"]
                ))
                record_id = cursor.lastrowid

                # Index in FTS5
                cursor.execute("""
                    INSERT INTO medical_chunks_fts 
                    (book_title, subject, mbbs_year, chapter, topic, content, content_rowid)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["book_title"],
                    item["subject"],
                    item["mbbs_year"],
                    item["chapter"],
                    item["topic"],
                    item["content"],
                    record_id
                ))
                chunks_indexed += 1

            # Register book summary
            cursor.execute("""
                INSERT INTO registered_books (filename, book_title, subject, mbbs_year, total_pages, indexed_chunks)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    book_title = excluded.book_title,
                    subject = excluded.subject,
                    mbbs_year = excluded.mbbs_year,
                    total_pages = excluded.total_pages,
                    indexed_chunks = excluded.indexed_chunks
            """, (pdf_path.name, book_title, subject, mbbs_year, total_pages, chunks_indexed))

            conn.commit()

        print(f"Indexed '{book_title}' ({chunks_indexed} pages/chunks)")
        return chunks_indexed

    def index_all_books(self, books_dir: Path = BOOKS_DIR) -> Dict[str, int]:
        """Indexes all PDFs in the books directory."""
        books_dir = Path(books_dir)
        results = {}
        for pdf_file in books_dir.glob("*.pdf"):
            try:
                count = self.index_book(pdf_file)
                results[pdf_file.name] = count
            except Exception as e:
                print(f"Error indexing {pdf_file.name}: {e}")
                results[pdf_file.name] = 0
        return results

    def get_books_summary(self) -> List[Dict[str, Any]]:
        """Returns list of all indexed medical textbooks with metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT filename, book_title, subject, mbbs_year, total_pages, indexed_chunks, created_at
                FROM registered_books
                ORDER BY subject ASC, book_title ASC
            """)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_page_content(self, book_title: str, page_number: int) -> Dict[str, Any]:
        """Retrieves exact or closest page content for display in the book viewer."""
        clean_title = re.sub(r'<[^>]+>', '', book_title).split('|')[0].strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Exact match by book_title and page
            cursor.execute("""
                SELECT * FROM medical_records
                WHERE book_title = ? AND (page_number = ? OR physical_page = ?)
                LIMIT 1
            """, (clean_title, page_number, page_number))
            row = cursor.fetchone()
            if row:
                return dict(row)

            # 2. Case-insensitive substring/keyword match
            for kw in ["robbins", "guyton", "tripathi", "ghai", "anatomy", "surgery", "internal medicine", "medicine", "pathology", "physiology", "pharmacology", "pediatrics"]:
                if kw in clean_title.lower():
                    cursor.execute("""
                        SELECT * FROM medical_records
                        WHERE LOWER(book_title) LIKE ? AND (page_number = ? OR physical_page = ?)
                        LIMIT 1
                    """, (f"%{kw}%", page_number, page_number))
                    row = cursor.fetchone()
                    if row:
                        return dict(row)
                    
                    # Fallback to page 1 of matched textbook
                    cursor.execute("""
                        SELECT * FROM medical_records
                        WHERE LOWER(book_title) LIKE ?
                        ORDER BY page_number ASC
                        LIMIT 1
                    """, (f"%{kw}%",))
                    row = cursor.fetchone()
                    if row:
                        return dict(row)

            # 3. Fallback to any book partially matching title
            cursor.execute("""
                SELECT * FROM medical_records
                WHERE LOWER(book_title) LIKE ?
                ORDER BY page_number ASC
                LIMIT 1
            """, (f"%{clean_title[:12].lower()}%",))
            row = cursor.fetchone()
            if row:
                return dict(row)

            # 4. Fallback to first available record
            cursor.execute("SELECT * FROM medical_records ORDER BY id ASC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else {}

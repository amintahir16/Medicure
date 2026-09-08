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
import sqlite_vec
from backend.config import DB_PATH, BOOKS_DIR
from backend.engine.parser import PDFBookParser
from backend.engine.embeddings import MBBSEmbeddingEngine, EMBEDDING_DIM

class MBBSIndexer:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.parser = PDFBookParser()
        self.embedder = MBBSEmbeddingEngine()
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as e:
            pass
        return conn

    def init_db(self):
        """Initializes relational tables, SQLite FTS5 virtual table, and sqlite-vec virtual table."""
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
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Ensure table_json column exists if migrating from older schema
            try:
                cursor.execute("ALTER TABLE medical_records ADD COLUMN table_json TEXT;")
            except sqlite3.OperationalError:
                pass

            # Ensure embedding BLOB column exists if migrating from older schema
            try:
                cursor.execute("ALTER TABLE medical_records ADD COLUMN embedding BLOB;")
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

            # Virtual Table for sqlite-vec dense vector search
            try:
                cursor.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS medical_vectors USING vec0(
                        record_id INTEGER PRIMARY KEY,
                        embedding float[{EMBEDDING_DIM}]
                    );
                """)
            except Exception as e:
                pass

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
        """Parses a single PDF book and indexes all pages into SQLite FTS5 and sqlite-vec."""
        pdf_path = Path(pdf_path)
        pages_data = self.parser.parse_book(pdf_path)
        for item in pages_data:
            item["table_json"] = None

        if not pages_data:
            return 0

        book_title = pages_data[0]["book_title"]
        subject = pages_data[0]["subject"]
        mbbs_year = pages_data[0]["mbbs_year"]
        total_pages = pages_data[0]["total_book_pages"]

        # Batch compute dense embeddings for all pages
        texts_to_embed = [
            self.embedder.prepare_passage_for_embedding(
                item["book_title"], item["chapter"], item["topic"], item["content"]
            )
            for item in pages_data
        ]
        dense_vectors = self.embedder.embed_documents(texts_to_embed)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Delete prior records for this book to allow re-indexing
            cursor.execute("SELECT id FROM medical_records WHERE book_title = ?", (book_title,))
            old_ids = [row["id"] for row in cursor.fetchall()]
            if old_ids:
                placeholders = ','.join('?' for _ in old_ids)
                cursor.execute(f"DELETE FROM medical_records WHERE id IN ({placeholders})", old_ids)
                cursor.execute(f"DELETE FROM medical_chunks_fts WHERE content_rowid IN ({placeholders})", old_ids)
                try:
                    cursor.execute(f"DELETE FROM medical_vectors WHERE record_id IN ({placeholders})", old_ids)
                except Exception:
                    pass

            chunks_indexed = 0
            for idx, item in enumerate(pages_data):
                vec = dense_vectors[idx]
                blob = self.embedder.serialize_vector(vec)

                # Store full record with dense embedding blob
                cursor.execute("""
                    INSERT INTO medical_records 
                    (book_title, subject, mbbs_year, chapter, topic, page_number, physical_page, total_pages, content, excerpt, table_json, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    item["table_json"],
                    blob
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

                # Index in medical_vectors
                try:
                    cursor.execute("""
                        INSERT INTO medical_vectors (record_id, embedding)
                        VALUES (?, ?)
                    """, (record_id, vec))
                except Exception:
                    pass

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

        print(f"Indexed '{book_title}' ({chunks_indexed} pages/chunks with dense vectors)")
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

    def backfill_missing_embeddings(self, batch_size: int = 64) -> int:
        """
        Backfills dense vector embeddings for any records that do not have them.
        Ensures existing textbooks are automatically upgraded to hybrid search.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, book_title, chapter, topic, content 
                FROM medical_records 
                WHERE embedding IS NULL
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()
            
            # If records already have blobs, make sure medical_vectors is in sync
            if not rows:
                try:
                    cursor.execute("SELECT count(*) as cnt FROM medical_vectors")
                    vec_cnt = cursor.fetchone()["cnt"]
                    cursor.execute("SELECT count(*) as cnt FROM medical_records WHERE embedding IS NOT NULL")
                    rec_cnt = cursor.fetchone()["cnt"]
                    if vec_cnt < rec_cnt:
                        print(f"[*] Syncing {rec_cnt - vec_cnt} vectors to medical_vectors table...")
                        cursor.execute("SELECT id, embedding FROM medical_records WHERE embedding IS NOT NULL")
                        to_sync = cursor.fetchall()
                        for r in to_sync:
                            vec = self.embedder.deserialize_vector(r["embedding"])
                            cursor.execute("INSERT OR REPLACE INTO medical_vectors (record_id, embedding) VALUES (?, ?)", (r["id"], vec))
                        conn.commit()
                except Exception:
                    pass
                return 0

            print(f"[*] Upgrading knowledge base: generating dense vectors for {len(rows)} pages...", flush=True)
            total_backfilled = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                texts = [
                    self.embedder.prepare_passage_for_embedding(
                        r["book_title"], r["chapter"], r["topic"], r["content"]
                    )
                    for r in batch
                ]
                embeddings = self.embedder.embed_documents(texts)
                for r, emb in zip(batch, embeddings):
                    blob = self.embedder.serialize_vector(emb)
                    cursor.execute("UPDATE medical_records SET embedding = ? WHERE id = ?", (blob, r["id"]))
                    try:
                        cursor.execute("INSERT OR REPLACE INTO medical_vectors (record_id, embedding) VALUES (?, ?)", (r["id"], emb))
                    except Exception:
                        pass
                conn.commit()
                total_backfilled += len(batch)
                print(f"[*] Processed {total_backfilled}/{len(rows)} dense embeddings...", flush=True)

            print(f"[*] Dense vector upgrade complete. {total_backfilled} pages indexed.", flush=True)
            return total_backfilled

    def sync_books_dir(self, books_dir: Path = BOOKS_DIR) -> Dict[str, int]:
        """
        Scans the books directory and automatically parses and indexes any newly
        added PDF files that have not yet been registered in the database.
        Also automatically backfills missing dense embeddings.
        """
        books_dir = Path(books_dir)
        newly_indexed = {}
        if books_dir.exists():
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT filename FROM registered_books")
                registered = {row["filename"] for row in cursor.fetchall()}

            for pdf_file in books_dir.glob("*.pdf"):
                if pdf_file.name not in registered:
                    print(f"Detected new textbook '{pdf_file.name}', indexing...")
                    try:
                        count = self.index_book(pdf_file)
                        newly_indexed[pdf_file.name] = count
                    except Exception as e:
                        print(f"Error auto-indexing {pdf_file.name}: {e}")
                        newly_indexed[pdf_file.name] = 0

        # Auto-upgrade any un-embedded records
        self.backfill_missing_embeddings()
        return newly_indexed

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

            # 2. Case-insensitive exact title match
            cursor.execute("""
                SELECT * FROM medical_records
                WHERE LOWER(book_title) = LOWER(?) AND (page_number = ? OR physical_page = ?)
                LIMIT 1
            """, (clean_title, page_number, page_number))
            row = cursor.fetchone()
            if row:
                return dict(row)

            # 3. Dynamic keyword match using title words (author name, distinctive terms)
            stopwords = {"the", "and", "of", "in", "for", "with", "principles", "textbook", "essentials", "clinical", "reference", "page", "basis"}
            title_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', clean_title) if w.lower() not in stopwords]

            # Try to match both keyword and page
            for kw in title_words:
                cursor.execute("""
                    SELECT * FROM medical_records
                    WHERE LOWER(book_title) LIKE ? AND (page_number = ? OR physical_page = ?)
                    LIMIT 1
                """, (f"%{kw}%", page_number, page_number))
                row = cursor.fetchone()
                if row:
                    return dict(row)

            # Try to match keyword with closest page
            for kw in title_words:
                cursor.execute("""
                    SELECT * FROM medical_records
                    WHERE LOWER(book_title) LIKE ?
                    ORDER BY ABS(page_number - ?) ASC
                    LIMIT 1
                """, (f"%{kw}%", page_number))
                row = cursor.fetchone()
                if row:
                    return dict(row)

            # 4. Fallback to any book partially matching title
            cursor.execute("""
                SELECT * FROM medical_records
                WHERE LOWER(book_title) LIKE ?
                ORDER BY page_number ASC
                LIMIT 1
            """, (f"%{clean_title[:10].lower()}%",))
            row = cursor.fetchone()
            if row:
                return dict(row)

            # 4. Fallback to first available record
            cursor.execute("SELECT * FROM medical_records ORDER BY id ASC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else {}

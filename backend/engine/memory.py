"""
Chat Session Memory & Student Learning Context
Manages multi-turn conversation histories, student curriculum profiles,
and pinned clinical pearl bookmarks.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import DB_PATH

class MBBSMemoryManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_memory_tables()

    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_memory_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    mbbs_year TEXT DEFAULT 'all',
                    subject TEXT DEFAULT 'all',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    provider_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pinned_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    title TEXT NOT NULL,
                    note_content TEXT NOT NULL,
                    book_title TEXT,
                    chapter TEXT,
                    page_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Backfill migration: ensure all historic conversations in chat_messages have chat_sessions records
            cursor.execute("""
                INSERT OR IGNORE INTO chat_sessions (session_id, title, created_at, updated_at)
                SELECT 
                    session_id,
                    SUBSTR(COALESCE((SELECT content FROM chat_messages m2 WHERE m2.session_id = m1.session_id AND m2.role = 'user' ORDER BY id ASC LIMIT 1), 'Medical Consultation'), 1, 45) as title,
                    MIN(created_at) as created_at,
                    MAX(created_at) as updated_at
                FROM chat_messages m1
                GROUP BY session_id;
            """)
            conn.commit()

    def create_session(self, session_id: str, title: str = "New Medical Consultation", mbbs_year: str = "all", subject: str = "all"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO chat_sessions (session_id, title, mbbs_year, subject, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (session_id, title, mbbs_year, subject))
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str, citations: Optional[List[Dict]] = None, provider_used: str = "offline"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cit_json = json.dumps(citations or [])
            cursor.execute("""
                INSERT INTO chat_messages (session_id, role, content, citations_json, provider_used)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, cit_json, provider_used))
            
            # Format clean title from first user message
            title_candidate = content.strip().replace("\n", " ")
            if len(title_candidate) > 42:
                title_candidate = title_candidate[:42] + "..."
            
            # Check if session exists in chat_sessions
            cursor.execute("SELECT session_id, title FROM chat_sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()
            
            if not session_row:
                init_title = title_candidate if role == "user" else "Medical Consultation"
                cursor.execute("""
                    INSERT INTO chat_sessions (session_id, title, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (session_id, init_title))
            elif role == "user" and session_row["title"] in ("New Medical Consultation", "Medical Consultation", "hello", "Hello"):
                cursor.execute("""
                    UPDATE chat_sessions 
                    SET title = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (title_candidate, session_id))
            else:
                cursor.execute("""
                    UPDATE chat_sessions 
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (session_id,))
            
            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return True

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, citations_json, provider_used, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))
            rows = cursor.fetchall()
            messages = []
            for r in rows:
                messages.append({
                    "role": r["role"],
                    "content": r["content"],
                    "citations": json.loads(r["citations_json"]) if r["citations_json"] else [],
                    "provider_used": r["provider_used"],
                    "created_at": r["created_at"]
                })
            return messages

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, title, mbbs_year, subject, created_at, updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT 50
            """)
            return [dict(r) for r in cursor.fetchall()]

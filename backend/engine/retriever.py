"""
Hybrid Retriever for MBBS Medical Knowledge Base
Combines SQLite FTS5 BM25 lexical ranking with FastEmbed dense neural vector embeddings
and Reciprocal Rank Fusion (RRF). Produces grounded medical evidence packages with
exact Book Title, Chapter, Topic, and Page Number citations.
"""

import re
import math
import sqlite3
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import numpy as np

from backend.config import DB_PATH
from backend.engine.indexer import MBBSIndexer

class MBBSHybridRetriever:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.indexer = MBBSIndexer(db_path=db_path)
        self.embedder = self.indexer.embedder

    def _clean_fts_query(self, query: str) -> str:
        """Cleans and formats natural language query into an FTS5 MATCH expression."""
        # Medical conversational fluff to exclude
        stop_words = {
            "what", "why", "how", "when", "where", "which", "who", "whom", 
            "explain", "describe", "detail", "discuss", "tell", "about", 
            "thell", "wat", "does", "is", "are", "was", "were", "the", "a", "an", "in", 
            "on", "at", "of", "for", "with", "and", "or", "to", "from", 
            "give", "list", "show", "can", "could", "would", "should",
            "book", "books", "reference", "references", "textbook", "page",
            "pages", "chapter", "topic", "mbbs", "student", "please", "me"
        }

        # Common student spelling corrections
        typo_map = {
            "thell": "tell",
            "wat": "what",
            "dieseas": "disease",
            "diesease": "disease",
            "desease": "disease",
            "diease": "disease",
            "infartion": "infarction",
            "apendicitis": "appendicitis",
            "pharamcology": "pharmacology",
            "pnumonia": "pneumonia",
            "symtoms": "symptoms"
        }
        for typo, fix in typo_map.items():
            query = re.sub(r'\b' + typo + r'\b', fix, query, flags=re.IGNORECASE)

        # Broad conceptual expansions
        if re.search(r'\bmeningitis\b|\bleptomening', query, re.IGNORECASE):
            query = "meningitis leptomeninges CSF lumbar puncture nuchal rigidity ceftriaxone dexamethasone"
        elif re.search(r'\bheart\s+disease\b|\bcardiac\s+disease\b', query, re.IGNORECASE):
            query = "cardiology coronary myocardial infarction atherosclerosis heart failure STEMI"
        elif re.search(r'\bshock\b', query, re.IGNORECASE) and not any(k in query.lower() for k in ["hypovolemic", "septic", "cardiogenic"]):
            query += " hypovolemic cardiogenic septic trauma ATLS resuscitation"

        # Extract words including acronyms (like ACE, ARB, STEMI, GFR, ATLS)
        raw_words = re.findall(r'\b[a-zA-Z0-9_\-\+]{2,}\b', query)
        if not raw_words:
            return ""

        keywords = [w for w in raw_words if w.lower() not in stop_words]
        if not keywords:
            keywords = raw_words

        # Build OR query with prefix matching on key terms, retaining medical acronyms
        fts_terms = []
        for k in keywords:
            if len(k) >= 4:
                fts_terms.append(f'"{k}"*')
            else:
                fts_terms.append(f'"{k}"')

        return " OR ".join(fts_terms)

    def _search_sparse_bm25(
        self, 
        fts_query: str, 
        limit: int = 25,
        subject_filter: Optional[str] = None, 
        year_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Stream 1: Lexical search using SQLite FTS5 (BM25 ranking)."""
        if not fts_query:
            return []

        conn = self.indexer.get_connection()
        cursor = conn.cursor()

        where_clauses = []
        params = [fts_query]

        if subject_filter and subject_filter.lower() != "all":
            where_clauses.append("r.subject = ?")
            params.append(subject_filter)

        if year_filter and year_filter.lower() not in ["all", "all years"]:
            where_clauses.append("(r.mbbs_year LIKE ? OR ? LIKE '%' || r.mbbs_year || '%')")
            params.extend([f"%{year_filter}%", year_filter])

        filter_sql = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""

        sql = f"""
            SELECT 
                r.id,
                bm25(medical_chunks_fts) as bm25_rank
            FROM medical_chunks_fts f
            JOIN medical_records r ON f.content_rowid = r.id
            WHERE medical_chunks_fts MATCH ? {filter_sql}
            ORDER BY bm25_rank ASC
            LIMIT {limit}
        """

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "rank": idx,
                    "bm25_score": abs(float(row["bm25_rank"]))
                }
                for idx, row in enumerate(rows)
            ]
        except sqlite3.OperationalError:
            return []

    def _search_dense_vectors(
        self, 
        query: str, 
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Stream 2: Neural dense vector semantic search using FastEmbed + sqlite-vec (or NumPy cosine fallback)."""
        try:
            q_emb = self.embedder.embed_query(query)
            if q_emb is None or len(q_emb) == 0 or np.all(q_emb == 0):
                return []
        except Exception as e:
            print(f"Warning: error generating query embedding: {e}")
            return []

        conn = self.indexer.get_connection()
        cursor = conn.cursor()

        # 1. Primary path: Native sqlite-vec KNN search
        try:
            cursor.execute(f"""
                SELECT record_id, distance
                FROM medical_vectors
                WHERE embedding MATCH ? AND k = {limit}
            """, (q_emb,))
            rows = cursor.fetchall()
            if rows:
                return [
                    {
                        "id": row["record_id"],
                        "rank": idx,
                        "dense_similarity": max(0.0, 1.0 - float(row["distance"]))
                    }
                    for idx, row in enumerate(rows)
                ]
        except Exception as e:
            pass

        # 2. Resilient fallback: NumPy vectorized cosine similarity
        try:
            cursor.execute("SELECT id, embedding FROM medical_records WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()
            if not rows:
                return []

            rec_ids = [r["id"] for r in rows]
            matrix = np.array([self.embedder.deserialize_vector(r["embedding"]) for r in rows], dtype=np.float32)
            
            # Compute cosine similarities: dot product of normalized vectors
            scores = np.dot(matrix, q_emb)
            top_indices = np.argsort(scores)[::-1][:limit]

            return [
                {
                    "id": rec_ids[idx],
                    "rank": rank,
                    "dense_similarity": float(scores[idx])
                }
                for rank, idx in enumerate(top_indices)
            ]
        except Exception as e:
            print(f"Warning: NumPy vector fallback encountered error: {e}")
            return []

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        subject_filter: Optional[str] = None, 
        year_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Dual-Stream Hybrid Retrieval:
        1. Stream 1: Sparse Lexical BM25 Search via SQLite FTS5
        2. Stream 2: Dense Semantic Vector Search via FastEmbed & sqlite-vec
        3. Fusion: Reciprocal Rank Fusion (RRF) merging dense and sparse streams
        4. Intent & Topic Re-ranking: Domain-specific boosting for curriculum precision
        5. Cross-Disciplinary Selection: Enforces multi-subject diversity
        6. Citation Packaging: Verified Book, Chapter, Topic, and Page citations
        """
        # Standardize query and correct medical typos upfront
        typo_map = {
            "thell": "tell",
            "wat": "what",
            "dieseas": "disease",
            "diesease": "disease",
            "desease": "disease",
            "diease": "disease",
            "infartion": "infarction",
            "apendicitis": "appendicitis",
            "pharamcology": "pharmacology",
            "pnumonia": "pneumonia",
            "symtoms": "symptoms"
        }
        for typo, fix in typo_map.items():
            query = re.sub(r'\b' + typo + r'\b', fix, query, flags=re.IGNORECASE)

        # Broad conceptual expansions
        expanded_query = query
        if re.search(r'\bmeningitis\b|\bleptomening', query, re.IGNORECASE):
            expanded_query = "meningitis leptomeninges CSF lumbar puncture nuchal rigidity ceftriaxone dexamethasone"
        elif re.search(r'\bheart\s+disease\b|\bcardiac\s+disease\b', query, re.IGNORECASE):
            expanded_query = "cardiology coronary myocardial infarction atherosclerosis heart failure STEMI"
        elif re.search(r'\bshock\b', query, re.IGNORECASE) and not any(k in query.lower() for k in ["hypovolemic", "septic", "cardiogenic"]):
            expanded_query += " hypovolemic cardiogenic septic trauma ATLS resuscitation"

        fts_query = self._clean_fts_query(expanded_query)

        # 1. Execute Stream 1 (Sparse BM25)
        sparse_matches = self._search_sparse_bm25(
            fts_query, 
            limit=25, 
            subject_filter=subject_filter, 
            year_filter=year_filter
        )

        # 2. Execute Stream 2 (Dense Neural Vectors)
        dense_matches = self._search_dense_vectors(
            query=query, 
            limit=25
        )

        # 3. Fuse via Reciprocal Rank Fusion (RRF)
        bm25_ranks = {item["id"]: item["rank"] for item in sparse_matches}
        dense_ranks = {item["id"]: item["rank"] for item in dense_matches}
        dense_sims = {item["id"]: item.get("dense_similarity", 0.0) for item in dense_matches}
        bm25_scores = {item["id"]: item.get("bm25_score", 0.0) for item in sparse_matches}

        all_candidate_ids = list(set(bm25_ranks.keys()).union(set(dense_ranks.keys())))
        if not all_candidate_ids:
            return []

        # 4. Fetch full candidate records from SQLite
        conn = self.indexer.get_connection()
        cursor = conn.cursor()

        placeholders = ','.join('?' for _ in all_candidate_ids)
        sql = f"""
            SELECT 
                r.id,
                r.book_title,
                r.subject,
                r.mbbs_year,
                r.chapter,
                r.topic,
                r.page_number,
                r.physical_page,
                r.total_pages,
                r.content,
                r.excerpt,
                r.table_json
            FROM medical_records r
            WHERE r.id IN ({placeholders})
        """
        cursor.execute(sql, all_candidate_ids)
        candidate_rows = cursor.fetchall()

        # Filter by subject / year if requested
        filtered_rows = []
        for row in candidate_rows:
            if subject_filter and subject_filter.lower() != "all" and row["subject"].lower() != subject_filter.lower():
                continue
            if year_filter and year_filter.lower() not in ["all", "all years"]:
                if year_filter.lower() not in row["mbbs_year"].lower():
                    continue
            filtered_rows.append(row)

        # If filtering produced 0 results, fall back to unfiltered candidates
        if not filtered_rows:
            filtered_rows = candidate_rows

        # Calculate lexical term frequency bonus
        query_keywords = [
            w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-\+]{2,}\b', query)
            if w.lower() not in {
                "what", "why", "how", "when", "where", "which", "who", "whom", 
                "explain", "describe", "detail", "discuss", "tell", "about", 
                "does", "is", "are", "was", "were", "the", "a", "an", "in", 
                "on", "at", "of", "for", "with", "and", "or", "to", "from", 
                "give", "list", "show", "can", "could", "would", "should",
                "book", "books", "reference", "references", "textbook", "page",
                "pages", "chapter", "topic", "mbbs", "student", "please", "me"
            }
        ]
        if not query_keywords:
            query_keywords = [w.lower() for w in re.findall(r'\b\w+\b', query)]
        query_counter = Counter(query_keywords)

        k_rrf = 60.0
        scored_records = []
        for row in filtered_rows:
            rec_id = row["id"]

            # RRF Component: w_sparse / (k + rank_sparse) + w_dense / (k + rank_dense)
            sparse_component = (0.5 / (k_rrf + bm25_ranks[rec_id] + 1)) if rec_id in bm25_ranks else 0.0
            dense_component = (0.5 / (k_rrf + dense_ranks[rec_id] + 1)) if rec_id in dense_ranks else 0.0
            rrf_score = sparse_component + dense_component

            # Dense similarity bonus
            dense_sim = dense_sims.get(rec_id, 0.0)

            # Topic / Chapter term overlap bonus
            doc_topic = f"{row['book_title']} {row['chapter']} {row['topic']}".lower()
            doc_content = (row['content'] or "").lower()
            topic_counter = Counter(re.findall(r'\b\w+\b', doc_topic))
            content_counter = Counter(re.findall(r'\b\w+\b', doc_content))

            topic_overlap = sum(min(query_counter[w], topic_counter[w]) * 4.0 for w in query_counter)
            content_overlap = sum(min(query_counter[w], content_counter[w]) * 1.0 for w in query_counter)
            norm = (math.sqrt(sum(query_counter[w]**2 for w in query_counter)) * 
                    math.sqrt(sum(content_counter[w]**2 for w in content_counter) + 10) + 1e-6)
            semantic_overlap = (topic_overlap + content_overlap) / norm

            combined_score = rrf_score + (dense_sim * 0.08) + (semantic_overlap * 0.08)

            # Intent-based clinical subject alignment
            q_lower = query.lower()
            is_pediatric = any(k in q_lower for k in ["pediatric", "paediatric", "child", "children", "neonate", "neonatal", "infant", "newborn", "fontanelle", "baby"])

            if any(k in q_lower for k in [
                "disease", "syndrome", "disorder", "pathology", "infarction", "ischemia", 
                "stemi", "failure", "lesion", "management", "criteria", "meningitis", 
                "infection", "fever", "headache", "dka", "stroke", "cirrhosis", "pneumonia", "aki"
            ]):
                if is_pediatric:
                    if row["subject"] == "Pediatrics":
                        combined_score *= 2.2
                else:
                    if row["subject"] == "Internal Medicine":
                        combined_score *= 2.2
                    elif row["subject"] in ["Pharmacology", "Pathology", "General Surgery"]:
                        combined_score *= 1.3
            elif any(k in q_lower for k in ["nerve", "artery", "plexus", "canal", "boundaries", "relations", "triangle", "muscle", "branch", "fossa", "meninges", "bone", "joint"]):
                if row["subject"] == "Anatomy":
                    combined_score *= 2.0
            elif any(k in q_lower for k in ["drug", "dose", "receptor", "antidote", "inhibitor", "blocker", "antibiotic", "side effect", "toxicity", "adverse", "regimen"]):
                if row["subject"] == "Pharmacology":
                    combined_score *= 2.0
            elif any(k in q_lower for k in ["cycle", "normal", "resting", "mechanics", "filtration", "starling", "dissociation", "clearance"]):
                if row["subject"] == "Physiology":
                    combined_score *= 2.0

            scored_records.append({
                "record_id": row["id"],
                "book_title": row["book_title"],
                "subject": row["subject"],
                "mbbs_year": row["mbbs_year"],
                "chapter": row["chapter"],
                "topic": row["topic"],
                "page_number": row["page_number"],
                "physical_page": row["physical_page"],
                "total_pages": row["total_pages"],
                "content": row["content"],
                "excerpt": row["excerpt"],
                "table_json": row["table_json"],
                "bm25_score": round(bm25_scores.get(rec_id, 0.0), 4),
                "dense_similarity": round(float(dense_sim), 4),
                "semantic_score": round(float(semantic_overlap), 4),
                "rrf_score": combined_score
            })

        # Sort by fused score descending
        scored_records.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Cross-disciplinary book diversity selection:
        # Pick top scoring match as primary reference, then enrich with top matches from other MBBS subjects
        top_results = []
        seen_books = set()

        if scored_records:
            top_results.append(scored_records[0])
            seen_books.add(scored_records[0]["book_title"])

        for r in scored_records[1:]:
            if r["book_title"] not in seen_books:
                top_results.append(r)
                seen_books.add(r["book_title"])
                if len(top_results) >= top_k:
                    break

        if len(top_results) < top_k:
            for r in scored_records[1:]:
                if r not in top_results:
                    top_results.append(r)
                    if len(top_results) >= top_k:
                        break

        # Annotate with verified citation badge strings
        for idx, item in enumerate(top_results, 1):
            item["ref_index"] = idx
            item["citation_label"] = (
                f"Ref [{idx}]: {item['book_title']} | {item['chapter']} | "
                f"{item['topic']} | Page {item['page_number']}"
            )
            item["short_citation"] = (
                f"{item['book_title'].split(':')[0]} • p. {item['page_number']}"
            )

        conn.close()
        return top_results

    def format_context_for_llm(self, evidence_list: List[Dict[str, Any]]) -> str:
        """Formats evidence passages into an authoritative clinical prompt context."""
        context_blocks = []
        for ev in evidence_list:
            block = (
                f"--- EVIDENCE EXCERPT [Ref {ev['ref_index']}] ---\n"
                f"BOOK TITLE: {ev['book_title']}\n"
                f"CURRICULUM YEAR: {ev['mbbs_year']}\n"
                f"SUBJECT: {ev['subject']}\n"
                f"CHAPTER: {ev['chapter']}\n"
                f"TOPIC / SECTION: {ev['topic']}\n"
                f"PAGE NUMBER: Page {ev['page_number']} (of {ev['total_pages']})\n"
                f"CONTENT:\n{ev['content']}\n"
            )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

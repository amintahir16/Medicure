"""
Hybrid Retriever for MBBS Medical Knowledge Base
Combines SQLite FTS5 BM25 lexical ranking with semantic term matching and
Reciprocal Rank Fusion (RRF). Produces grounded medical evidence packages with
exact Book Title, Chapter, Topic, and Page Number citations.
"""

import re
import math
import sqlite3
from typing import List, Dict, Any, Optional
from collections import Counter
from backend.config import DB_PATH
from backend.engine.indexer import MBBSIndexer

class MBBSHybridRetriever:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.indexer = MBBSIndexer(db_path=db_path)

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

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        subject_filter: Optional[str] = None, 
        year_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. FTS5 BM25 search
        2. Semantic TF-IDF scoring
        3. Reciprocal Rank Fusion
        4. Strict citation metadata packaging
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

        fts_query = self._clean_fts_query(expanded_query)
        if not fts_query:
            return []

        conn = self.indexer.get_connection()
        cursor = conn.cursor()

        # Build SQL query with optional filters
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
                r.table_json,
                bm25(medical_chunks_fts) as bm25_rank
            FROM medical_chunks_fts f
            JOIN medical_records r ON f.content_rowid = r.id
            WHERE medical_chunks_fts MATCH ? {filter_sql}
            ORDER BY bm25_rank ASC
            LIMIT 25
        """

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            # Fallback to simple LIKE query if FTS syntax edge case occurs
            like_words = [w for w in re.findall(r'\b\w{3,}\b', query) if w.lower() not in ["the", "and", "explain", "what", "why", "thell", "tell"]]
            like_clause = " OR ".join(["r.content LIKE ?"] * len(like_words)) if like_words else "1=1"
            like_params = [f"%{w}%" for w in like_words]
            fallback_sql = f"""
                SELECT id, book_title, subject, mbbs_year, chapter, topic, 
                       page_number, physical_page, total_pages, content, excerpt, table_json, 0.0 as bm25_rank
                FROM medical_records r
                WHERE ({like_clause})
                LIMIT 15
            """
            cursor.execute(fallback_sql, like_params)
            rows = cursor.fetchall()

        if not rows:
            # Secondary fallback: search across all records without filter if strict filter yielded 0
            if subject_filter or year_filter:
                return self.search(query, top_k=top_k, subject_filter=None, year_filter=None)
            return []

        # Calculate semantic scoring with topic boosting and stopword filtering
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

        scored_records = []
        for rank_idx, row in enumerate(rows):
            doc_topic = (f"{row['book_title']} {row['chapter']} {row['topic']}").lower()
            doc_content = (row['content'] or "").lower()
            
            topic_counter = Counter(re.findall(r'\b\w+\b', doc_topic))
            content_counter = Counter(re.findall(r'\b\w+\b', doc_content))
            
            # Boost matches in topic/chapter by 4x
            topic_overlap = sum(min(query_counter[w], topic_counter[w]) * 4.0 for w in query_counter)
            content_overlap = sum(min(query_counter[w], content_counter[w]) * 1.0 for w in query_counter)
            total_overlap = topic_overlap + content_overlap

            norm = (math.sqrt(sum(query_counter[w]**2 for w in query_counter)) * 
                    math.sqrt(sum(content_counter[w]**2 for w in content_counter) + 10) + 1e-6)
            semantic_score = total_overlap / norm

            # Reciprocal Rank Fusion: RRF = BM25 component + weighted semantic score
            bm25_weight = 1.0 / (20.0 + rank_idx + 1)
            rrf_score = bm25_weight + (semantic_score * 0.15)

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
                        rrf_score *= 2.2
                else:
                    if row["subject"] == "Internal Medicine":
                        rrf_score *= 2.2
                    elif row["subject"] in ["Pharmacology", "Pathology", "General Surgery"]:
                        rrf_score *= 1.3
            elif any(k in q_lower for k in ["nerve", "artery", "plexus", "canal", "boundaries", "relations", "triangle", "muscle", "branch", "fossa", "meninges"]):
                if row["subject"] == "Anatomy":
                    rrf_score *= 2.0
            elif any(k in q_lower for k in ["drug", "dose", "receptor", "antidote", "inhibitor", "blocker", "antibiotic", "side effect", "toxicity", "adverse", "regimen"]):
                if row["subject"] == "Pharmacology":
                    rrf_score *= 2.0
            elif any(k in q_lower for k in ["cycle", "normal", "resting", "mechanics", "filtration", "starling", "dissociation", "clearance"]):
                if row["subject"] == "Physiology":
                    rrf_score *= 2.0

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
                "bm25_score": abs(float(row["bm25_rank"])),
                "semantic_score": round(float(semantic_score), 4),
                "rrf_score": rrf_score
            })

        # Sort by RRF score descending
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

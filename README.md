# 🩺 Medicure MBBS — 5-Year Curriculum Clinical AI Chatbot

A state-of-the-art, Google Gemini-styled clinical consultation companion engineered specifically for MBBS medical students, interns, and clinical board candidates (USMLE, NEET-PG, PLAB). 

Medicure is powered by a **Hybrid Dense-Sparse RAG Engine** strictly grounded in standard MBBS medical textbooks (such as *BD Chaurasia's Human Anatomy*, *Guyton & Hall Physiology*, *Robbins Pathology*, *KD Tripathi Pharmacology*, *Harrison's Internal Medicine*, and *Bailey & Love Surgery*). Every answer provides clickable citation badges with **Book Title, Chapter, Topic, and Exact Page Number**, complete with an interactive verbatim excerpt viewer and full textbook page inspector.

---

## 🌟 Key Features

1. **Hybrid Dense-Sparse RAG & Reciprocal Rank Fusion (RRF)**:
   - **Local Neural Vector Embeddings**: Uses **FastEmbed** with `BAAI/bge-small-en-v1.5` (384 dimensions) running locally on CPU via ONNX Runtime—100% offline, zero API costs, and high inference throughput.
   - **Lexical BM25 Ranking**: Uses SQLite FTS5 for exact clinical terminology, drug names, anatomical relations, and medical abbreviations.
   - **Reciprocal Rank Fusion (RRF)**: Merges dense semantic KNN vector retrieval and sparse BM25 scores with clinical topic boosting and subject-intent alignment (Anatomy, Physiology, Pharmacology, Pathology, Surgery, Pediatrics, Internal Medicine).
   - **Native SQLite Vector Storage**: Integrates `sqlite-vec` with serialized float32 BLOB fallback for robust vector storage and persistence.
   - **Automatic Backfill & Sync**: Automatically generates dense vector embeddings for newly added textbooks on the fly and syncs existing records.

2. **Structural RAG & Verbatim Citation Grounding**:
   - Unlike naive RAG that blindly token-splits text and loses page numbers, Medicure preserves physical and printed page numbers, running headers, chapter titles, and clinical topic structures.
   - Every answer embeds inline references formatted as `[Ref X: Book Name | Chapter | Topic | Page Y]`.
   - Click any citation badge to slide open the **Textbook Reference Inspector**, highlighting the exact verbatim excerpt on that page.

3. **Curriculum-Wide Medical Textbook Support**:
   - Pre-loaded with real MBBS literature (e.g., *BD Chaurasia's Handbook of General Anatomy* covering osteology, histology, joints, and neuroanatomy).
   - Extensible to the full 5-year MBBS syllabus:
     - **Year 1**: Anatomy (*BD Chaurasia*, *Gray's*) & Physiology (*Guyton & Hall*, *Ganong*).
     - **Year 2**: Pathology (*Robbins & Cotran*) & Pharmacology (*KD Tripathi*, *Katzung*).
     - **Year 3 & 4**: Ophthalmology, ENT, Community Medicine, and Forensic Medicine.
     - **Year 4 & 5**: Internal Medicine (*Harrison's*, *Davidson*), General Surgery (*Bailey & Love*), and Pediatrics (*Ghai*).

4. **Dynamic Drag-and-Drop Book Uploader**:
   - Drop any standard medical PDF textbook directly into the web UI or add it to the `books/` directory.
   - The engine automatically parses chapters, topics, and page numbers, embeds passages into 384-dimensional dense vectors, and indexes them into SQLite FTS5 and vector tables simultaneously.

5. **Multi-Model Support (Free Tiers & Offline)**:
   - **Medicure Grounded Engine (Offline)**: 100% functional out-of-the-box with zero configuration or external API key needed.
   - **Google Gemini 2.0 Flash / 1.5 Flash**: Deep clinical reasoning with 1M token context window via free Google AI Studio API key.
   - **Groq Llama 3.3 70B**: Ultra-fast 300+ tok/s clinical inference via free Groq API key.

6. **Medical Student Study Modes**:
   - 🩺 **Standard Clinical Q&A**: Academic explanations with physiological mechanisms, diagrams, and comparison matrices.
   - 📋 **Clinical Case Vignettes**: Simulates USMLE / MBBS case scenarios with patient presentation, differential diagnosis, and evidence-based management.
   - 🎓 **Board Exam Viva / Quiz**: Simulates oral exam viva voce with model answers, examiner follow-ups, and clinical traps.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- Dependencies installed via:
  ```bash
  pip install -r requirements.txt
  ```
  *(Includes `fastembed`, `sqlite-vec`, `numpy`, `fastapi`, `uvicorn`, `pypdf`, and `google-generativeai`)*

### 2. Launch the Application
Run the one-click startup script:
```bash
python main.py
```
Open your browser at **`http://127.0.0.1:8000`**.

### 3. Run Automated Tests
Execute the comprehensive test suite validating dense embeddings, SQLite vector storage, hybrid RRF retrieval, and grounded answer generation:
```bash
python tests/test_engine.py
```

---

## 🏛️ System Architecture

```
                 [MBBS Textbooks (PDFs)]
                            │
                            ▼
              [pypdf + Structural Parser]
              (Page Nums, Chapters, Topics)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    [FastEmbed Engine]              [SQLite FTS5]
(BAAI/bge-small-en-v1.5, 384d)     (BM25 Full-Text Index)
            │                               │
            ▼                               ▼
    [medical_vectors]             [medical_chunks_fts]
(sqlite-vec / Dense BLOB)         (Lexical Inverted Index)
            └───────────────┬───────────────┘
                            │
                       User Query
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
     Dense KNN Search                Sparse BM25 Search
            └───────────────┬───────────────┘
                            │
                            ▼
           [Reciprocal Rank Fusion (RRF)]
           + Clinical Subject Intent Boost
           + Topic Density Overlap Scoring
                            │
                            ▼
       [Grounded Context Assembly & Citations]
                            │
      ┌─────────────────────┴─────────────────────┐
      │ Google Gemini 2.0 Flash / Groq / Offline  │
      └─────────────────────┬─────────────────────┘
                            │
                            ▼
[Interactive Gemini UI: Clickable Citation Badges & Slide-Out Page Viewer]
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health check reporting active models, book count, total pages, and hybrid retrieval engine metadata |
| `/api/chat` | `POST` | Generates a grounded response with citations, clinical evidence, and session memory |
| `/api/books` | `GET` | Lists all indexed textbooks, total pages, and syncs newly added books |
| `/api/books/upload` | `POST` | Uploads a PDF textbook, extracts structure, generates dense embeddings, and indexes it |
| `/api/books/page` | `GET` | Retrieves full text and metadata for a specific textbook page in the slide-out viewer |
| `/api/sessions` | `GET` / `POST` | Lists or creates consultation chat sessions |
| `/api/sessions/{id}` | `GET` / `DELETE` | Retrieves session message history or deletes a session |
| `/api/sample-prompts` | `GET` | Returns pre-curated MBBS study prompts across all 5 years |
| `/api/viva-quiz` | `GET` | Returns viva voce examination questions and model answers |

---

## ⚙️ Setting Up Free API Keys (Optional)

1. Click **⚙️ Model & API Settings** in the sidebar.
2. Select your desired engine:
   - **Offline / Built-in Grounded Engine**: Works immediately with no setup.
   - **Google Gemini 2.0 Flash**: Obtain a free API key at [Google AI Studio](https://aistudio.google.com).
   - **Groq Llama 3.3 70B**: Obtain a free API key at [Groq Console](https://console.groq.com).
3. Paste the key and click **Save Settings**. Your key is stored locally in your browser session.

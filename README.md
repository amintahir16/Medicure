# 🩺 Medicure MBBS — 5-Year Curriculum Clinical AI Chatbot

A state-of-the-art, Google Gemini-styled clinical consultation companion engineered specifically for MBBS medical students, interns, and clinical board candidates (USMLE, NEET-PG, PLAB). 

Medicure is powered by a **Hybrid Dense-Sparse RAG Engine** strictly grounded in standard MBBS medical textbooks. Every answer provides clickable citation badges with **Book Title, Chapter, Topic, and Exact Page Number**, complete with an interactive verbatim excerpt viewer and full textbook page inspector.

> 📚 **Note on Textbook Library**: For lightweight repository testing and rapid onboarding, **1 textbook is pre-loaded**: *BD Chaurasia's Handbook of General Anatomy* (266 pages). You can easily upload additional MBBS textbooks (for Physiology, Pathology, Pharmacology, Internal Medicine, Surgery, Pediatrics, etc.) directly from the web UI via the drag-and-drop **Library** modal, or by placing PDF files in the `books/` folder.

---

## 🌟 Key Features

1. **PubMedBERT Dense Biomedical Embeddings & Extended Context**:
   - **Domain-Specific Neural Vectors**: Powered by **PubMedBERT** (`NeuML/pubmedbert-base-embeddings`, 768 dimensions) via `SentenceTransformers`. Pre-trained natively on PubMed abstracts and biomedical literature with MeSH & UMLS clinical ontologies.
   - **Extended Passage Context**: Expanded from 700 characters to up to **2,200 characters** (~512 tokens), fully exploiting PubMedBERT's context window without truncating complex anatomical relations, clinical criteria, or drug mechanisms.
   - **100% Offline & Local**: Optimized for CPU inference—no external embedding API keys or recurring subscription costs required.
   - **Automatic DB Vector Migration**: `MBBSIndexer` detects embedding dimension mismatches dynamically and auto-migrates SQLite vector tables (`medical_vectors vec0 float[768]`) without data loss or manual SQL scripting.

2. **Lexical BM25 Ranking & Sanitized FTS5 Search**:
   - Uses SQLite FTS5 for high-precision exact matching of clinical terminology, drug brand/generic names, anatomical landmarks, and medical abbreviations.
   - **Query Sanitization**: Cleans and strips punctuation, quotes, asterisks, brackets, and conversational stop words to ensure 100% robust FTS5 queries without syntax errors.
   - **Low-Content Filtering**: Automatically downweights figure captions, diagram labels, and truncated pages (< 250 characters) to keep evidence clinically relevant.

3. **Hybrid Reciprocal Rank Fusion (RRF) & Clinical Intent Routing**:
   - Merges dense semantic KNN vector retrieval and sparse BM25 scores with balanced weights and topic-density bonuses.
   - **Subject-Intent Multipliers**: Dynamically aligns queries with target disciplines (Anatomy, Physiology, Pharmacology, Pathology, General Surgery, Pediatrics, Internal Medicine) based on clinical terminology detection.
   - **In-Memory Caching & Sync**: Features an in-memory vector matrix cache for fast cosine similarity fallbacks, with automatic cache invalidation when new textbooks are indexed.

4. **🚨 Real-Time Clinical Emergency & Red-Flag Triage Detection**:
   - Built-in rule-based red-flag classifier in `MBBSGenerator` detects high-risk, life-threatening presentations:
     - **Acute Coronary Syndrome / STEMI** (crushing chest pain radiating to left arm/jaw)
     - **Tension Pneumothorax** (tracheal deviation, unilateral absent breath sounds — immediate needle decompression)
     - **Acute Airway Emergency / Anaphylaxis** (stridor, angioedema — IM epinephrine protocol)
     - **Suspected Meningococcal Septicemia** (fever with non-blanching petechial/purpuric rash)
     - **Severe Hemorrhagic Shock** (massive blood loss — ATLS resuscitation)
     - **Status Epilepticus** (prolonged/continuous seizures — IV benzodiazepines)
   - Automatically prepends high-priority emergency triage banners advising immediate **ABCDE stabilization** and emergency department escalation before textbook consultation.
   - Styled with custom glowing callouts in both dark and light modes.

5. **Structural RAG & Verbatim Citation Grounding**:
   - Preserves physical and printed page numbers, running headers, chapter titles, and clinical topic structures.
   - Every answer embeds inline references formatted as `[Ref X: Book Name | Chapter | Topic | Page Y]`.
   - Click any citation badge to slide open the **Textbook Reference Inspector**, highlighting the exact verbatim excerpt on that page.
   - **Strict Page Lookup & XSS Protection**: Eliminates arbitrary textbook fallbacks and sanitizes PDF content against XSS before rendering in the reader.

6. **Dynamic Drag-and-Drop Book Uploader (Add Books from UI)**:
   - Drop any standard medical PDF textbook directly into the web UI or add it to the `books/` directory.
   - **Production Hardened**: Features path traversal sanitization, `%PDF` magic byte verification, and a 200MB size limit with chunked streaming.
   - **Non-Blocking Background Indexing**: PDF parsing and vector generation run asynchronously in a worker thread (`asyncio.to_thread`), keeping the server responsive.
   - No server restart required; UI badges and book counters update immediately.

7. **Multi-Model Support (Free Tiers & Offline)**:
   - **Medicure Grounded Engine (Offline)**: 100% functional out-of-the-box with zero configuration or external API key needed.
   - **Google Gemini 2.0 Flash / 1.5 Flash**: Deep clinical reasoning with 1M token context window via free Google AI Studio API key.
   - **Groq Cloud AI (OpenAI GPT-OSS 120B)**: Ultra-fast 300+ tok/s clinical inference via free Groq API key.

8. **Medical Student Study Modes**:
   - 🩺 **Standard Clinical Q&A**: Academic explanations with physiological mechanisms, diagrams, and comparison matrices.
   - 📋 **Clinical Case Vignettes**: Simulates USMLE / MBBS case scenarios with patient presentation, differential diagnosis, and evidence-based management.
   - 🎓 **Board Exam Viva / Quiz**: Simulates oral exam viva voce with model answers, examiner follow-ups, and clinical traps.

---

## 📖 Adding More Textbooks via the UI

You can expand Medicure's curriculum knowledge base at any time without command-line intervention:

1. Launch the application and click the **📚 Library** button in the top navigation bar.
2. Drag and drop any standard MBBS textbook PDF (e.g., *Guyton & Hall Physiology*, *Robbins & Cotran Pathology*, *KD Tripathi Pharmacology*, *Harrison's Internal Medicine*, *Bailey & Love Surgery*, etc.) into the upload zone.
3. The server validates the PDF header, extracts physical page numbers, parses chapters and topics, generates 768-dimensional PubMedBERT embeddings, and indexes the book into SQLite.
4. The library modal, header badge, and sidebar will instantly update with your newly indexed textbook and total pages.

*(Alternatively, copy your PDF files into the `books/` directory; they are automatically synced and indexed upon server startup via the FastAPI lifespan manager).*

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  *Key dependencies include: `sentence-transformers`, `fastapi`, `uvicorn`, `pypdf`, `sqlite-vec`, `numpy`, `google-generativeai`, and `reportlab`.*

### 2. Launch the Application
Run the startup script:
```bash
python main.py
```
Or start with `uvicorn`:
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at **`http://127.0.0.1:8000`**.

### 3. Run Automated Tests
Execute the comprehensive test suite validating 768-dim PubMedBERT embeddings, extended context lengths, emergency triage alerts, strict page retrieval, sanitized FTS5 queries, and grounded answer generation:
```bash
python tests/test_engine.py
```

---

## 🏛️ System Architecture

```
                 [MBBS Textbooks (PDFs)]
          (Pre-loaded: BD Chaurasia / UI Uploads)
                             │
                             ▼
                [pypdf + Structural Parser]
               (Page Nums, Chapters, Topics)
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
    [PubMedBERT Engine]             [SQLite FTS5]
(NeuML/pubmedbert-base, 768d)      (Sanitized BM25 Index)
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
      Dense KNN Search               Sparse BM25 Search
      (768d Cosine)                  (Sanitized Lexical)
             └───────────────┬───────────────┘
                             │
                             ▼
            [Reciprocal Rank Fusion (RRF)]
            + Clinical Subject Intent Boost
            + Topic Density Overlap Scoring
            + Low-Content Page Downweighting
                             │
                             ▼
      [Emergency Red-Flag Clinical Triage Detector]
      (STEMI, Tension Pneumo, Anaphylaxis, Shock alerts)
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
| `/api/health` | `GET` | Health check reporting active models, book count, total pages, and hybrid retrieval engine metadata (PubMedBERT 768d) |
| `/api/chat` | `POST` | Generates a grounded response with citations, clinical evidence, emergency triage alerts, and session memory |
| `/api/books` | `GET` | Lists all indexed textbooks, total pages, and syncs newly added books |
| `/api/books/upload` | `POST` | Uploads a PDF textbook with magic byte validation, extracts structure, generates 768d embeddings, and indexes it asynchronously |
| `/api/books/page` | `GET` | Retrieves full text and metadata for a specific textbook page in the slide-out viewer with strict lookup and XSS sanitization |
| `/api/sessions` | `GET` / `POST` | Lists or creates consultation chat sessions |
| `/api/sessions/{id}` | `GET` / `DELETE` | Retrieves session message history or deletes a session |
| `/api/sample-prompts` | `GET` | Returns pre-curated MBBS study prompts across all 5 curriculum years |
| `/api/viva-quiz` | `GET` | Returns viva voce examination questions and model answers |

---

## ☁️ Deployment (Vercel & ASGI)

Medicure is pre-configured for modern ASGI and serverless cloud deployment:

- **Vercel**: Includes [`vercel.json`](file:///c:/Users/MY%20PC/Documents/Medicure/vercel.json) and [`pyproject.toml`](file:///c:/Users/MY%20PC/Documents/Medicure/pyproject.toml) configured with `@vercel/python` routing all requests to the FastAPI application.
- **FastAPI Lifespan**: Uses modern `@asynccontextmanager lifespan` for graceful startup synchronization, asynchronous book indexing, and clean cache management.
- **Top-Level ASGI Export**: Exposes `app` in [`main.py`](file:///c:/Users/MY%20PC/Documents/Medicure/main.py) for compatibility with Gunicorn, Uvicorn, and containerized deployments.

---

## ⚙️ Setting Up Free API Keys (Optional)

1. Click **⚙️ Model & API Settings** in the top navigation or sidebar.
2. Select your desired engine:
   - **Offline / Built-in Grounded Engine**: Works immediately with zero configuration (uses the pre-loaded *BD Chaurasia* and any user-uploaded textbooks).
   - **Google Gemini 2.0 Flash**: Obtain a free API key at [Google AI Studio](https://aistudio.google.com).
   - **Groq Cloud AI (OpenAI GPT-OSS 120B)**: Obtain a free API key at [Groq Console](https://console.groq.com).
3. Paste the key and click **Save Settings**. Your key is stored securely in your browser session storage.

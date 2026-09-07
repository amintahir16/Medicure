# 🩺 MedGemini MBBS — 5-Year Curriculum Clinical AI Chatbot

A state-of-the-art, Google Gemini-styled AI consultation companion engineered specifically for MBBS medical students, interns, and clinical board candidates (USMLE, NEET-PG, PLAB). 

MedGemini is strictly grounded in **5 full years of MBBS medical textbooks** (Anatomy, Physiology, Pathology, Pharmacology, Internal Medicine, and General Surgery). Every answer provides clickable citation badges with **Book Title, Chapter, Topic, and Exact Page Number**, complete with an interactive excerpt viewer and full textbook page inspector.

---

## 🌟 Key Features

1. **Karpathy-Aligned Structural RAG & Citation Grounding**:
   - Unlike naive RAG that blindly token-splits text and loses page numbers, MedGemini preserves physical and printed page numbers, running headers, chapter titles, and clinical topic structures.
   - Every answer embeds inline references formatted as `[Ref X: Book Name | Chapter | Topic | Page Y]`.
   - Click any citation badge to slide open the **Textbook Reference Inspector**, highlighting the exact verbatim excerpt on that page.

2. **Pre-Loaded 5-Year Curriculum Library**:
   - **Year 1**: *Human Anatomy & Neuroanatomy Principles* (Cranial nerves, Brachial plexus, Osteology, Histology).
   - **Year 1**: *Textbook of Medical Physiology: Guyton Principles* (Cardiac cycle, Wiggers diagram, Respiratory mechanics, GFR & Countercurrent multiplier).
   - **Year 2**: *Pathologic Basis of Disease: Robbins Principles* (Cellular injury, MI histopathology timeline, Neoplasia, Nephrotic vs Nephritic).
   - **Year 2**: *Essentials of Medical Pharmacology: Tripathi Principles* (ACEi vs ARBs bradykinin cough mechanism, Beta-blocker glucagon reversal, Antibiotics).
   - **Year 3 to 5**: *Principles of Internal Medicine and Clinical Examination* (Acute chest pain STEMI triage, CURB-65 pneumonia, KDIGO AKI).
   - **Year 4 & 5**: *Principles of General Surgery and Critical Care* (ATLS primary survey, Tension pneumothorax, Inguinal hernia Hesselbach triangle, Burns).

3. **Multi-Model Support (Free Tiers)**:
   - **MedGemini Grounded Engine (Offline)**: 100% functional out-of-the-box with zero configuration or API key.
   - **Google Gemini 2.0 Flash / 1.5 Flash**: Deep clinical reasoning with 1M token context window via free Google AI Studio key.
   - **Groq Llama 3.3 70B**: Lightning-fast 300+ tok/s inference via free Groq API key.

4. **Dynamic Drag-and-Drop Book Uploader**:
   - Students can drop any PDF textbook (e.g., *Harrison's*, *Guyton*, *Bailey & Love*, *Robbins*, *BD Chaurasia*) directly into the web UI.
   - The engine automatically parses chapters, topics, and page numbers, indexing them into the SQLite FTS5 database on the fly.

5. **Medical Student Study Modes**:
   - 🩺 **Standard Clinical Q&A**: Academic explanations with physiological mechanisms, diagrams, and comparison matrices.
   - 📋 **Clinical Case Vignettes**: Simulates USMLE / MBBS case scenarios with patient presentation, differential diagnosis, and management.
   - 🎓 **Board Exam Viva / Quiz**: Simulates oral exam viva voce with model answers and clinical traps.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- Dependencies installed via:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Launch the Application
Run the one-click startup script:
```bash
python main.py
```
Open your browser at **`http://127.0.0.1:8000`**.

### 3. Run Automated Tests
```bash
python tests/test_engine.py
```

---

## 🏛️ System Architecture

```
[MBBS Textbooks (PDFs)]
         │
         ▼
[pypdf + Structural Parser] ──► Extracts: Page Num, Chapter, Topic, Excerpt
         │
         ▼
[SQLite FTS5 + Semantic Vector Store]
         │
    User Query
         │
         ▼
[Hybrid BM25 + Dense Semantic RRF Retriever]
         │
         ▼
[Grounded Prompt Assembly + Citation Grounding Engine]
         │
  ┌──────┴──────────────────────────────────────────┐
  │ Google Gemini 2.0 Flash / Groq / Offline Engine │
  └──────┬──────────────────────────────────────────┘
         │
         ▼
[Interactive Gemini UI with Clickable Citation Badges & Slide-Out Page Viewer]
```

---

## ⚙️ Setting Up Free API Keys (Optional)

1. Click **⚙️ Model & API Settings** in the sidebar.
2. Select your desired engine:
   - **Google Gemini 2.0 Flash**: Obtain a free API key at [Google AI Studio](https://aistudio.google.com).
   - **Groq Llama 3.3 70B**: Obtain a free key at [Groq Console](https://console.groq.com).
3. Paste the key and click **Save Settings**. Your key is securely stored in your local browser storage.

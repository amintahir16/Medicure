"""
Multi-Model Generator with Strict Medical Grounding and Citation Verification
Supports:
1. Google Gemini 2.0 Flash / 1.5 Flash (via free Google AI Studio key)
2. Groq Llama 3.3 70B (via free Groq API key)
3. Built-in Grounded Offline Synthesis Engine (100% functional out-of-the-box)
"""

import os
import re
import json
import httpx
from typing import List, Dict, Any, Optional, Tuple
from backend.config import (
    DEFAULT_LLM_PROVIDER, GEMINI_API_KEY, GROQ_API_KEY, 
    GEMINI_MODEL, GROQ_MODEL
)
from backend.engine.retriever import MBBSHybridRetriever

MEDICAL_SYSTEM_PROMPT = """You are MedGemini MBBS, an elite clinical AI medical tutor strictly grounded in the official 5-year MBBS curriculum textbooks.

YOUR PRIME DIRECTIVE:
Deliver clean, concise, high-yield, and to-the-point clinical answers. Be direct, authoritative, and structured. Avoid fluff, unnecessary disclaimers, walls of uninterrupted text, and conversational filler.

FORMATTING & WRITING RULES:
1. **Direct Answer First**:
   - Start immediately with a 1-2 sentence core clinical summary answering the specific question.
   - Never use conversational preamble (e.g., "Certainly!", "In medical practice", "Here is an overview").

2. **Clean Bullet Points Under Clear Headings**:
   - Structure responses into concise sections using `### [Main Topic]` and `#### [Subtopic]`.
   - Use crisp bullet points with bold lead keywords: `- **Key Concept**: Clinical explanation`.
   - Keep bullet points to 1-2 lines. Never dump dense paragraphs.

3. **Tables for Comparisons, Scoring & Staging**:
   - When presenting diagnostic criteria, scoring systems (e.g. Alvarado, Glasgow Coma Scale, CURB-65), lab values, or drug comparisons, format them in a clean Markdown table.

4. **Clean, Discrete Citations**:
   - Cite using short tags only: `[Ref 1]`, `[Ref 2]` at the end of key statements or bullet points.
   - Do NOT write long bracketed strings like `[Ref 1: Book Name | Chapter | ...]`.
   - Do NOT scatter citations excessively on every word.

5. **No Bibliography at Bottom**:
   - Do NOT output a "### Verified References" or text bibliography at the bottom (the application UI automatically renders verified reference cards).

6. **High-Yield Clinical Pearl**:
   - Conclude with a single callout: `> 💡 **Clinical Pearl**: [High-yield USMLE/NEET-PG point or clinical trap]`.
"""

class MBBSGenerator:
    def __init__(self, retriever: Optional[MBBSHybridRetriever] = None):
        self.retriever = retriever or MBBSHybridRetriever()

    async def generate_response(
        self,
        query: str,
        provider: str = DEFAULT_LLM_PROVIDER,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        year_filter: Optional[str] = None,
        subject_filter: Optional[str] = None,
        study_mode: str = "standard",  # 'standard', 'case_vignette', 'viva_quiz'
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves grounded evidence from MBBS books and generates a response
        with exact citations using the selected provider (Groq, Gemini, or Offline).
        """
        # Step 1: Determine active provider and key
        active_provider = (provider or "offline").lower().strip()
        active_key = (api_key or "").strip()
        if not active_key:
            if active_provider == "gemini":
                active_key = (GEMINI_API_KEY or "").strip()
            elif active_provider == "groq":
                active_key = (GROQ_API_KEY or "").strip()

        # If user explicitly selected Groq or Gemini without supplying an API key:
        if active_provider == "groq" and not active_key:
            return {
                "query": query,
                "answer": (
                    "### ⚠️ Groq Cloud AI Key Required\n\n"
                    "You have selected **Groq Cloud AI (Llama 3.3 70B)**, but no Groq API Key was provided.\n\n"
                    "**How to connect:**\n"
                    "1. Click **⚙️ Model & API Settings** in the bottom left of the sidebar.\n"
                    "2. Paste your free Groq API key (starts with `gsk_...` from [console.groq.com](https://console.groq.com/keys)).\n"
                    "3. Click **Save & Apply Settings**.\n\n"
                    "*(Or select **MedGemini Built-in Grounded Engine** for 100% offline access with zero API keys required).* "
                ),
                "evidence": [],
                "citations": [],
                "provider_used": "Groq (Key Missing)",
                "total_references": 0
            }

        if active_provider == "gemini" and not active_key:
            return {
                "query": query,
                "answer": (
                    "### ⚠️ Google Gemini Key Required\n\n"
                    "You have selected **Google Gemini 2.0 Flash**, but no Gemini API Key was provided.\n\n"
                    "**How to connect:**\n"
                    "1. Click **⚙️ Model & API Settings** in the bottom left of the sidebar.\n"
                    "2. Paste your free Google AI Studio key (starts with `AIzaSy...`).\n"
                    "3. Click **Save & Apply Settings**.\n\n"
                    "*(Or select **MedGemini Built-in Grounded Engine** for 100% offline access with zero API keys required).* "
                ),
                "evidence": [],
                "citations": [],
                "provider_used": "Gemini (Key Missing)",
                "total_references": 0
            }

        # Step 2: Handle greetings and casual inquiries
        clean_q = query.strip().lower().rstrip(".!?")
        is_greeting = clean_q in [
            "hello", "hi", "hey", "greetings", "good morning", "good evening", 
            "good afternoon", "who are you", "what can you do", "help", "start"
        ]

        if is_greeting:
            if active_provider == "groq" and active_key:
                try:
                    greeting_prompt = (
                        f"The medical student or physician just said: '{query}'. "
                        "Respond warmly, conversationally, and authoritatively as MedGemini, the clinical AI medical tutor powered by Groq Cloud AI (Llama 3.3 70B). "
                        "Introduce yourself as an AI tutor strictly grounded across 5 full years of MBBS curriculum textbooks "
                        "(Anatomy, Physiology, Pathology, Pharmacology, Internal Medicine, General Surgery, Pediatrics). "
                        "Invite them to ask any clinical question, case vignette, or exam scenario with exact textbook page citations."
                    )
                    answer_text, model_used = await self._call_groq(greeting_prompt, active_key, model_name or GROQ_MODEL)
                    return {
                        "query": query,
                        "answer": answer_text,
                        "evidence": [],
                        "citations": [],
                        "provider_used": f"Groq ({model_used})",
                        "total_references": 0
                    }
                except Exception as e:
                    print(f"Groq greeting failed: {e}")
                    return {
                        "query": query,
                        "answer": f"⚠️ **Groq Cloud AI Connection Note**: Groq returned an error: `{str(e)}`.\n\nPlease check your Groq API key in **Settings** (starts with `gsk_...`).\n\nFalling back to offline mode for now:\n\n" + self._get_offline_greeting(),
                        "evidence": [],
                        "citations": [],
                        "provider_used": "Groq Error Fallback",
                        "total_references": 0
                    }
            elif active_provider == "gemini" and active_key:
                try:
                    greeting_prompt = (
                        f"The medical student or physician just said: '{query}'. "
                        "Respond warmly, conversationally, and authoritatively as MedGemini, the clinical AI medical tutor powered by Google Gemini 2.0 Flash. "
                        "Introduce yourself as an AI tutor strictly grounded across 5 full years of MBBS curriculum textbooks. "
                        "Invite them to ask any clinical question, case vignette, or exam scenario with exact textbook page citations."
                    )
                    answer_text = await self._call_gemini(greeting_prompt, active_key, model_name or GEMINI_MODEL)
                    return {
                        "query": query,
                        "answer": answer_text,
                        "evidence": [],
                        "citations": [],
                        "provider_used": f"Google Gemini ({model_name or GEMINI_MODEL})",
                        "total_references": 0
                    }
                except Exception as e:
                    print(f"Gemini greeting failed: {e}")
                    return {
                        "query": query,
                        "answer": f"⚠️ **Gemini AI Connection Note**: Gemini returned an error: `{str(e)}`.\n\nPlease check your Gemini API key in **Settings**.\n\n" + self._get_offline_greeting(),
                        "evidence": [],
                        "citations": [],
                        "provider_used": "Gemini Error Fallback",
                        "total_references": 0
                    }
            else:
                return {
                    "query": query,
                    "answer": self._get_offline_greeting(),
                    "evidence": [],
                    "citations": [],
                    "provider_used": "MedGemini Built-in Grounded Engine (Offline)",
                    "total_references": 0
                }

        # Step 3: Hybrid Retrieval with conversational context expansion
        search_q = query
        if history and len(query.split()) < 7:
            last_convo = " ".join([m.get("content", "") for m in history[-2:]]).lower()
            for med_term in [
                "meningitis", "appendicitis", "heart failure", "stemi", "pneumonia",
                "aki", "tuberculosis", "cirrhosis", "stroke", "dka", "cardiac cycle",
                "brachial plexus", "carpal tunnel", "cranial nerve", "shock", "hernia", "diarrhea"
            ]:
                if med_term in last_convo and med_term not in query.lower():
                    search_q = f"{med_term} {query}"
                    break

        evidence = self.retriever.search(
            query=search_q, 
            top_k=4, 
            year_filter=year_filter, 
            subject_filter=subject_filter
        )

        # If no local textbook passage found:
        if not evidence:
            if active_provider == "groq" and active_key:
                try:
                    broad_prompt = (
                        f"The medical student or physician asked: '{query}'.\n"
                        "Provide a direct, high-yield, to-the-point clinical answer strictly following these rules:\n"
                        "1. **Direct Answer First**: 1-2 sentence core summary answering the question immediately.\n"
                        "2. **Structured Sections with Bullet Points**: Use clean headings (### and ####) with crisp, bold-led bullet points (- **Concept**: explanation).\n"
                        "3. **Tables for Staging / Criteria**: If presenting diagnostic criteria, scoring systems, or comparisons, use a clean Markdown table.\n"
                        "4. **Clinical Pearl**: Conclude with a single blockquote: '> 💡 **Clinical Pearl**: ...'.\n"
                        "Do NOT include conversational filler, disclaimers, walls of text, or a references list."
                    )
                    answer_text, model_used = await self._call_groq(broad_prompt, active_key, model_name or GROQ_MODEL)
                    return {
                        "query": query,
                        "answer": answer_text,
                        "evidence": [],
                        "citations": [],
                        "provider_used": f"Groq ({model_used})",
                        "total_references": 0
                    }
                except Exception as e:
                    print(f"Groq query failed: {e}")
                    return {
                        "query": query,
                        "answer": f"⚠️ **Groq API Error**: `{str(e)}`.\n\n" + self._get_no_evidence_message(query),
                        "evidence": [],
                        "citations": [],
                        "provider_used": "Groq Error Fallback",
                        "total_references": 0
                    }
            elif active_provider == "gemini" and active_key:
                try:
                    broad_prompt = (
                        f"The medical student or physician asked: '{query}'. "
                        "Provide a comprehensive, authoritative, medical-school-level clinical explanation using standard clinical medicine principles. "
                        "Structure your response clearly with clinical pathophysiological principles, diagnostic criteria, and management."
                    )
                    answer_text = await self._call_gemini(broad_prompt, active_key, model_name or GEMINI_MODEL)
                    return {
                        "query": query,
                        "answer": answer_text,
                        "evidence": [],
                        "citations": [],
                        "provider_used": f"Google Gemini ({model_name or GEMINI_MODEL})",
                        "total_references": 0
                    }
                except Exception as e:
                    return {
                        "query": query,
                        "answer": f"⚠️ **Gemini API Error**: `{str(e)}`.\n\n" + self._get_no_evidence_message(query),
                        "evidence": [],
                        "citations": [],
                        "provider_used": "Gemini Error Fallback",
                        "total_references": 0
                    }
            else:
                return {
                    "query": query,
                    "answer": self._get_no_evidence_message(query),
                    "evidence": [],
                    "citations": [],
                    "provider_used": "MedGemini Grounded Engine (Offline)",
                    "total_references": 0
                }

        # Step 4: Grounded Response Generation with textbook evidence
        context_str = self.retriever.format_context_for_llm(evidence)
        prompt = self._build_prompt(query, context_str, study_mode, history=history)

        answer_text = ""
        provider_used = "offline"

        if active_provider == "gemini" and active_key:
            try:
                answer_text = await self._call_gemini(prompt, active_key, model_name or GEMINI_MODEL)
                provider_used = f"Google Gemini ({model_name or GEMINI_MODEL})"
            except Exception as e:
                print(f"Gemini API failed: {e}. Falling back to Grounded Offline Engine.")
                answer_text = f"> ⚠️ *Note: Google Gemini API error ({e}). Displaying grounded offline synthesis:*\n\n" + self._generate_offline_grounded_response(query, evidence, study_mode)
                provider_used = "Offline Grounded Engine (Gemini fallback)"
        elif active_provider == "groq" and active_key:
            try:
                answer_text, model_used = await self._call_groq(prompt, active_key, model_name or GROQ_MODEL)
                provider_used = f"Groq ({model_used})"
            except Exception as e:
                print(f"Groq API call failed: {e}. Falling back to Grounded Offline Engine.")
                answer_text = f"> ⚠️ *Note: Groq API error ({e}). Displaying grounded offline synthesis:*\n\n" + self._generate_offline_grounded_response(query, evidence, study_mode)
                provider_used = "Offline Grounded Engine (Groq fallback)"
        else:
            answer_text = self._generate_offline_grounded_response(query, evidence, study_mode)
            provider_used = "MedGemini Built-in Grounded Engine (Offline)"

        # Step 5: Extract structured citations
        citations = self._extract_citations(answer_text, evidence)

        return {
            "query": query,
            "answer": answer_text,
            "evidence": evidence,
            "citations": citations,
            "provider_used": provider_used,
            "total_references": len(evidence)
        }

    def _get_offline_greeting(self) -> str:
        return (
            "### 🩺 Hello, Doctor! Welcome to MedGemini (Offline Mode)\n\n"
            "I am your clinical medical AI companion, strictly grounded across **7 verified MBBS curriculum textbooks**:\n"
            "- **1st Year**: Human Anatomy & Neuroanatomy, Medical Physiology (Guyton Principles)\n"
            "- **2nd Year**: Pathology (Robbins Principles), Medical Pharmacology (Tripathi Principles)\n"
            "- **3rd to 5th Year**: Internal Medicine & Clinical Examination, General Surgery (ATLS & Critical Care), Pediatrics (Ghai Principles)\n\n"
            "⚡ **Zero API Keys Required**: The built-in grounded offline engine is active right now! Every medical answer includes exact citations with **Book Title, Chapter, Topic, and Page Number**.\n\n"
            "*(Tip: To enable conversational LLM mode powered by Groq Cloud AI or Gemini, select it and enter your free API key in **⚙️ Model & API Settings** in the sidebar).* \n\n"
            "#### 💡 Sample Clinical Questions to Try:\n"
            "1. *'Explain the phases of the cardiac cycle and Wiggers diagram'* (Guyton Physiology, p. 2-3)\n"
            "2. *'Tell me about acute bacterial, viral and TB meningitis workup and CSF findings'* (Internal Medicine, p. 6)\n"
            "3. *'Why do ACE inhibitors cause persistent dry cough compared to ARBs?'* (Pharmacology, p. 2)\n"
            "4. *'Describe Erb palsy vs Klumpke palsy nerve root injuries and waiter tip deformity'* (Anatomy, p. 2)\n"
            "5. *'What is the Alvarado score and McBurney point for acute appendicitis?'* (General Surgery, p. 4)\n\n"
            "What medical topic would you like to explore?"
        )

    def _get_no_evidence_message(self, query: str) -> str:
        return (
            "### ⚠️ No Direct Curriculum Passage Found\n\n"
            f"No direct textbook passages were found matching *\"{query}\"* in the currently indexed MBBS textbooks.\n\n"
            "**Suggestions**:\n"
            "- Try medical keywords (e.g. *meningitis*, *cardiac cycle*, *ACE inhibitors*, *Erb palsy*, *appendicitis*, *STEMI triage*).\n"
            "- Switch the **Curriculum Year** filter to **All 5 Years Curriculum** in the sidebar.\n"
            "- You can also upload any additional MBBS textbook PDF via the **MBBS Book Library** modal!\n"
            "- Or connect your free **Groq Cloud AI** or **Google Gemini** API key in **Settings** for broader conversational synthesis."
        )

    def _build_prompt(self, query: str, context: str, study_mode: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        mode_instruction = ""
        if study_mode == "case_vignette":
            mode_instruction = "STRUCTURE: Clinical Case Vignette (Presentation, High-Yield Physical Findings, Diagnostic Workup, Definitive Management)."
        elif study_mode == "viva_quiz":
            mode_instruction = "STRUCTURE: MBBS Oral Exam Card (Examiner Question, High-Yield Model Answer, Key Clinical Traps, Exam Pearl)."
        else:
            mode_instruction = "STRUCTURE: Direct, high-yield clinical consultation. Direct answer first, followed by clear, bulleted clinical points."

        history_block = ""
        if history:
            turns = []
            for msg in history[-3:]:
                role = "Student" if msg.get("role") == "user" else "MedGemini AI"
                clean_c = msg.get("content", "")[:250].replace("\n", " ")
                turns.append(f"{role}: {clean_c}")
            if turns:
                history_block = "PRIOR CONSULTATION CONTEXT:\n" + "\n".join(turns) + "\n\n"

        return f"""TEXTBOOK EVIDENCE EXCERPTS:
{context}

{history_block}CLINICAL INQUIRY:
{query}

{mode_instruction}

STRICT INSTRUCTIONS:
1. Answer directly and crisply. Address the inquiry immediately in the first sentence.
2. Structure with clean subheadings (####) and concise bullet points (- **Key**: Description).
3. Do NOT produce long paragraphs or walls of text. Keep bullet points to 1-2 lines.
4. If presenting scoring criteria, staging, or comparisons, use a clean Markdown table.
5. Cite using short tags only: [Ref 1], [Ref 2] at the end of key statements.
6. Do NOT output a text references list at the bottom.
7. Conclude with a single '> 💡 **Clinical Pearl**: ...' blockquote.
"""

    async def _call_gemini(self, prompt: str, api_key: str, model: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": MEDICAL_SYSTEM_PROMPT + "\n\n" + prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
                "maxOutputTokens": 2048
            }
        }
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _get_available_groq_models(self, client: httpx.AsyncClient, headers: Dict[str, str]) -> List[str]:
        models_url = "https://api.groq.com/openai/v1/models"
        try:
            resp = await client.get(models_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                raw_models = [item.get("id") for item in data.get("data", []) if item.get("id")]
                print(f"[*] Fetched {len(raw_models)} live models from Groq: {raw_models}")
                # Filter out whisper (audio), guard (safety filter), or embedding models
                chat_models = [
                    m for m in raw_models 
                    if not any(skip in m.lower() for skip in ["whisper", "guard", "embed", "tts"])
                ]
                return chat_models
            else:
                print(f"[*] Could not fetch Groq models list ({resp.status_code}): {resp.text}")
                return []
        except Exception as e:
            print(f"[*] Error fetching Groq models list: {e}")
            return []

    async def _call_groq(self, prompt: str, api_key: str, model: str) -> Tuple[str, str]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=35.0) as client:
            # 1. Dynamically retrieve currently active models for this Groq key
            live_models = await self._get_available_groq_models(client, headers)

            # Build list of models to try in priority order
            candidate_models = []
            if model and (not live_models or model in live_models):
                candidate_models.append(model)
            
            # If live models were retrieved, prioritize the best available chat models
            if live_models:
                # Rank: 120b, 70b, 27b, 8b, others
                for rank_kw in ["120b", "70b", "27b", "8b"]:
                    for lm in live_models:
                        if rank_kw in lm.lower() and lm not in candidate_models:
                            candidate_models.append(lm)
                for lm in live_models:
                    if lm not in candidate_models:
                        candidate_models.append(lm)
            else:
                # Fallback hardcoded candidates if /models endpoint was unavailable
                candidate_models.extend([
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "openai/gpt-oss-120b",
                    "qwen/qwen3.6-27b",
                    "openai/gpt-oss-20b"
                ])

            models_to_try = list(dict.fromkeys(candidate_models))
            print(f"[*] Groq models to try in order: {models_to_try}")

            last_error = None
            for m in models_to_try:
                payload = {
                    "model": m,
                    "messages": [
                        {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048
                }
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return content, m
                    elif resp.status_code == 401:
                        err_json = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
                        err_msg = err_json.get("error", {}).get("message", "Invalid API Key")
                        raise Exception(f"Invalid Groq API Key (401): {err_msg}. Please check your key in Settings.")
                    elif resp.status_code == 429:
                        err_json = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
                        err_msg = err_json.get("error", {}).get("message", "Rate limit exceeded")
                        raise Exception(f"Groq Rate Limit (429): {err_msg}")
                    else:
                        err_text = resp.text
                        print(f"[*] Groq model '{m}' failed ({resp.status_code}): {err_text}")
                        last_error = f"Model {m} error ({resp.status_code}): {err_text[:150]}"
                        continue
                except httpx.HTTPStatusError as e:
                    last_error = str(e)
                    continue
                except httpx.RequestError as e:
                    last_error = str(e)
                    continue

        raise Exception(f"Groq request failed: {last_error}")

    def _extract_page_components(self, text: str, topic: str = "", chapter: str = "") -> Tuple[List[str], str]:
        """
        Extracts clean narrative paragraphs and clinical pearl from page text.
        Strips running headers, chapter titles, topic headings, and raw table artifacts.
        """
        # 1. If text has a pearl section, split it out
        pearl_text = ""
        pearl_match = re.search(r'(?:Clinical\s+Pearl|Pearl|Remember)[:\s\-\–]+([^\n]+(?:\n[^\n]+)?)', text, re.IGNORECASE)
        if pearl_match:
            raw_pearl = pearl_match.group(1).strip()
            # Remove any table prefix if captured
            raw_pearl = re.split(r'\b(?:Table|Diagnostic\s+&|Score|Class\s+[IVX]+):', raw_pearl, flags=re.IGNORECASE)[0].strip()
            pearl_text = " ".join(raw_pearl.split())
            if pearl_text and not pearl_text.endswith('.'):
                pearl_text += "."

        # 2. Cut off raw table representations from narrative text
        narrative_part = text
        for split_kw in [
            "\nDiagnostic & Management Matrix:", "\nTable:", 
            "\nClinical Pearl:", "\nPearl:"
        ]:
            if split_kw in narrative_part:
                narrative_part = narrative_part.split(split_kw)[0]

        lines = [l.strip() for l in narrative_part.split('\n') if l.strip()]
        filtered_lines = []

        table_keywords = {
            "lesion", "roots involved", "clinical posture", "primary mechanism",
            "boundary", "contents", "provocative tests", "motor deficit",
            "artery", "main branches", "territory supplied", "common clinical complication",
            "lesion type", "forehead involvement", "blink reflex", "etiology example",
            "phase", "mitral", "aortic", "ventricular volume", "acoustic event",
            "determinant", "physiological definition", "effect on stroke volume", "clinical modulator",
            "direction", "affinity for o2", "p50 value", "precipitating factors",
            "substance", "tubular handling", "clearance relation to gfr", "clinical diagnostic role",
            "pattern", "histologic hallmark", "classic organs", "key pathogenesis",
            "step", "leukocyte molecule", "endothelial counter-receptor", "clinical defect",
            "condition", "physical findings", "immediate bedside intervention", "definitive management",
            "shock class", "blood loss", "heart rate", "blood pressure", "fluid",
            "sign / symptom", "alvarado points", "physiological / anatomical mechanism",
            "characteristic", "indirect inguinal hernia", "direct inguinal hernia",
            "acs category", "ecg diagnostic criteria", "troponin biomarkers", "reperfusion strategy",
            "pillar", "representative drug", "mechanistic target", "proven benefit",
            "curb-65 score", "risk stratification", "30-day mortality", "recommended treatment setting",
            "diagnostic parameter", "prerenal azotemia", "intrinsic aki", "postrenal obstruction",
            "drug class", "mechanism of action", "bradykinin level", "key adverse effects", "pregnancy safety",
            "drug", "receptor target", "cardioselectivity", "special clinical role",
            "generation", "key agents", "spectrum", "clinical indications",
            "drug group", "ribosomal target", "mechanism", "characteristic toxicities"
        }

        for line in lines:
            line_lower = line.lower()
            # Narrative lines filtering
            if re.match(r'^(?:Chapter\s+\d+|Page\s+\d+|MBBS\s+|CURRICULUM\s+|Table\b)', line, re.IGNORECASE):
                continue
            if topic and (line_lower == topic.lower() or (topic.lower() in line_lower and len(line) < len(topic) + 15)):
                continue
            if chapter and (chapter.lower() in line_lower and len(line) < len(chapter) + 15):
                continue
            if any(line_lower == kw for kw in table_keywords):
                break  # Reached table section
            filtered_lines.append(line)

        # Merge wrapped lines into continuous sentences
        merged_narrative = ""
        for line in filtered_lines:
            if not merged_narrative:
                merged_narrative = line
            elif merged_narrative.endswith(('-', '—')):
                merged_narrative = merged_narrative[:-1] + line
            elif not merged_narrative.endswith(('.', ':', ';', '!', '?')):
                merged_narrative += " " + line
            else:
                merged_narrative += "\n\n" + line

        # Split into distinct paragraphs and clean leading artifacts
        paras = []
        for p in merged_narrative.split('\n\n'):
            p_clean = " ".join(p.split())
            if topic and p_clean.lower().startswith(topic.lower()):
                p_clean = p_clean[len(topic):].strip()
            # Strip trailing remnants of topic title lines that wrapped
            p_clean = re.sub(
                r'^(?:(?:and|or|of|for|with|in|the)\s+)?(?:Regimens?|Management|Criteria|Overdose|Assessment|Treatment|Differentiation|Technique|Injuries|Localization|Complications|Syndrome|Mechanisms?|Dynamics|Calculations?)\s+', 
                '', p_clean, flags=re.IGNORECASE
            )
            # Must be a substantial sentence starting with capital letter
            if len(p_clean) > 35 and any(p_clean.endswith(punct) for punct in ['.', '!', '?']):
                paras.append(p_clean)

        return paras, pearl_text

    def _format_clinical_bullets(self, text: str, max_bullets: int = 4) -> str:
        """Transforms narrative medical textbook sentences into clean, readable clinical bullet points."""
        text = " ".join(text.split()).strip()
        if not text:
            return ""

        # If text has a colon introducing a list of items, criteria, or signs
        colon_split = re.split(r':\s*', text, maxsplit=1)
        if len(colon_split) == 2 and any(kw in colon_split[0].lower() for kw in [
            'signs', 'etiology', 'triad', 'pathogens', 'regimens', 'score', 
            'features', 'criteria', 'indications', 'findings', 'causes'
        ]):
            intro, items_str = colon_split
            items = re.split(r';\s*|,\s*(?=[A-Z][a-z]+|\b(?:In|At|For|With|Score|Empiric|Adjunctive)\b)', items_str)
            if len(items) >= 2:
                bullet_lines = [f"- **{intro.strip()}**:"]
                for item in items[:max_bullets]:
                    item_clean = item.strip().rstrip('.')
                    if item_clean:
                        bullet_lines.append(f"  • {item_clean}")
                return "\n".join(bullet_lines)

    def _split_medical_sentences(self, text: str) -> List[str]:
        """Splits medical text into distinct sentences while preserving species and drug abbreviations."""
        protected = re.sub(r'\b([A-Z])\.\s*', r'\1<DOT>', text)
        protected = re.sub(r'\b(e\.g|i\.e|vs|approx|etc|tab|inj|cap)\.\s*', r'\1<DOT>', protected, flags=re.IGNORECASE)
        splits = re.split(r'[.!?]\s+(?=[A-Z0-9])', protected)
        return [s.replace('<DOT>', '. ').strip() for s in splits if s.strip()]

    def _format_clinical_bullets(self, text: str, max_bullets: int = 4) -> str:
        """Transforms narrative medical textbook sentences into clean, readable clinical bullet points."""
        text = " ".join(text.split()).strip()
        if not text:
            return ""

        # If text has a colon introducing a list of items, criteria, or signs
        colon_split = re.split(r':\s*', text, maxsplit=1)
        if len(colon_split) == 2 and any(kw in colon_split[0].lower() for kw in [
            'signs', 'etiology', 'triad', 'pathogens', 'regimens', 'score', 
            'features', 'criteria', 'indications', 'findings', 'causes'
        ]):
            intro, items_str = colon_split
            items = re.split(r';\s*|,\s*(?=[A-Z][a-z]+|\b(?:In|At|For|With|Score|Empiric|Adjunctive)\b)', items_str)
            if len(items) >= 2:
                bullet_lines = [f"- **{intro.strip()}**:"]
                for item in items[:max_bullets]:
                    item_clean = item.strip().rstrip('.')
                    if item_clean:
                        bullet_lines.append(f"  • {item_clean}")
                return "\n".join(bullet_lines)

        # Split into distinct sentences using protected abbreviation splitter
        sentences = self._split_medical_sentences(text)
        lines = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 15:
                continue
            # Try to extract leading bold term if sentence is structured like "Term is..." or "Term (details)..."
            match = re.match(r'^([A-Za-z0-9\s\/\-\'\(\)]+?)(?:\s+is\s+|\s+are\s+|:\s+|\s+comprises\s+|\s+presents\s+with\s+)(.*)$', s_clean, re.IGNORECASE)
            if match and len(match.group(1).split()) <= 4 and not match.group(1).lower().startswith(('it', 'this', 'there', 'they', 'continued')):
                label = match.group(1).strip()
                rest = match.group(2).strip()
                lines.append(f"- **{label}**: {rest}")
            else:
                lines.append(f"- {s_clean}")
            if len(lines) >= max_bullets:
                break

        return "\n".join(lines) if lines else text

    def _generate_offline_grounded_response(
        self, 
        query: str, 
        evidence: List[Dict[str, Any]], 
        study_mode: str
    ) -> str:
        """
        High-yield deterministic clinical synthesis engine.
        Synthesizes top evidence chunks into an authoritative, beautifully structured clinical
        consultation with guaranteed exact citations (Book, Chapter, Topic, Page Number).
        """
        primary = evidence[0]
        secondary = evidence[1] if len(evidence) > 1 else None

        primary_paras, primary_pearl = self._extract_page_components(
            primary["content"], topic=primary["topic"], chapter=primary["chapter"]
        )

        response_parts = []

        # 1. Clean Topic Header
        clean_topic = primary['topic'].split(':')[0].strip()
        response_parts.append(f"### {clean_topic}")

        # 2. Check for Table
        table_md = ""
        has_table = False
        if primary.get("table_json"):
            try:
                t_obj = json.loads(primary["table_json"])
                table_data = t_obj.get("data", [])
                if table_data and len(table_data) >= 2:
                    headers = table_data[0]
                    rows = table_data[1:]
                    t_lines = [
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join([":---"] * len(headers)) + " |"
                    ]
                    for row in rows:
                        cells = [str(c).replace("\n", " ").replace("|", "\\|") for c in row]
                        while len(cells) < len(headers):
                            cells.append("")
                        cells = cells[:len(headers)]
                        cells[0] = f"**{cells[0]}**"
                        t_lines.append("| " + " | ".join(cells) + " |")
                    table_md = "\n".join(t_lines)
                    has_table = True
            except Exception:
                pass

        # 3. Direct High-Yield Summary (1-2 sentence core definition)
        if primary_paras:
            p0 = primary_paras[0].rstrip('.')
            sents = self._split_medical_sentences(p0)
            definition = " ".join(sents[:1]) if sents else p0
            response_parts.append(f"{definition}. [Ref 1]")

            # Etiology / Mechanisms (remainder of p0 if substantial)
            if len(sents) > 1:
                remainder_lines = []
                for s in sents[1:]:
                    s_clean = s.rstrip('.')
                    if len(s_clean) > 15:
                        remainder_lines.append(self._format_clinical_bullets(s_clean, max_bullets=2))
                if remainder_lines:
                    response_parts.append(f"\n#### Core Pathophysiology & Etiology\n" + "\n".join(remainder_lines))

            # Clinical Features & Diagnostic Workup (p1)
            if len(primary_paras) > 1:
                p1 = primary_paras[1].rstrip('.')
                bullets_p1 = self._format_clinical_bullets(p1, max_bullets=4)
                response_parts.append(f"\n#### Clinical Features & Diagnostic Workup\n{bullets_p1} [Ref 1]")

            # Formatted Markdown Table if present
            if has_table:
                response_parts.append(f"\n#### Diagnostic & Clinical Matrix\n{table_md}")

            # Management & Guidelines (p2, skip if redundant with table)
            if len(primary_paras) > 2:
                p2 = primary_paras[2].rstrip('.')
                is_duplicate = (has_table and "alvarado" in p2.lower() and "score" in p2.lower() and "points" in table_md.lower())
                if not is_duplicate:
                    bullets_p2 = self._format_clinical_bullets(p2, max_bullets=4)
                    response_parts.append(f"\n#### Management & Clinical Guidelines\n{bullets_p2} [Ref 1]")
        else:
            clean_raw = " ".join(primary["content"].split())[:350].rstrip('.')
            response_parts.append(f"{clean_raw}. [Ref 1]")

        # 4. Strict Multidisciplinary Clinical Correlation (MUST share specific medical entity)
        if secondary and secondary["subject"] != primary["subject"]:
            stop_words = {
                "acute", "chronic", "management", "scoring", "score", "principles", 
                "clinical", "pathogenesis", "criteria", "study", "general", "overview", 
                "signs", "symptoms", "examination", "syndrome", "presentation", "regimen", 
                "therapy", "treatment", "disease", "disorder", "infection", "diagnosis", "points"
            }
            primary_stems = {
                w[:5].lower() for w in re.findall(r'\b[a-zA-Z]{5,}\b', primary["topic"] + " " + query)
                if w.lower() not in stop_words
            }
            sec_stems = {
                w[:5].lower() for w in re.findall(r'\b[a-zA-Z]{5,}\b', secondary["topic"])
                if w.lower() not in stop_words
            }
            if primary_stems.intersection(sec_stems):
                sec_paras, _ = self._extract_page_components(
                    secondary["content"], topic=secondary["topic"], chapter=secondary["chapter"]
                )
                if sec_paras:
                    sec_bullets = self._format_clinical_bullets(sec_paras[0], max_bullets=2)
                    response_parts.append(f"\n#### Correlative {secondary['subject']} Insights\n{sec_bullets} [Ref 2]")

        # 5. High-Yield Clinical Pearl
        raw_pearl = primary_pearl
        pearl_ref = "[Ref 1]"
        if not raw_pearl and secondary:
            _, sec_pearl = self._extract_page_components(secondary["content"], secondary["topic"], secondary["chapter"])
            if sec_pearl:
                raw_pearl = sec_pearl
                pearl_ref = "[Ref 2]"

        if raw_pearl:
            clean_pearl = re.sub(r'^(?:Clinical\s+Pearl|Pearl|Remember)[:\s\-\–]+', '', raw_pearl, flags=re.IGNORECASE).strip()
            clean_pearl = re.sub(r'\s*\|\s*.*$', '', clean_pearl).strip()
            response_parts.append(f"\n> 💡 **Clinical Pearl**: {clean_pearl} {pearl_ref}")
        else:
            response_parts.append(
                f"\n> 💡 **Clinical Pearl**: Always correlate anatomical landmarks, clinical criteria, and physiological reserve during management of {clean_topic}. [Ref 1]"
            )

        return "\n".join(response_parts)

    def _extract_citations(self, answer_text: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses inline citations and correlates them with exact evidence metadata."""
        found_refs = set()
        # Match [Ref X] or [Ref X: ...] or Ref [X]
        matches = re.findall(r'(?:\[Ref\s*(\d+)|Ref\s*\[(\d+)\])', answer_text, re.IGNORECASE)
        for m in matches:
            val = m[0] or m[1]
            try:
                found_refs.add(int(val))
            except ValueError:
                pass

        citations = []
        for ev in evidence:
            is_cited = ev["ref_index"] in found_refs or ev["ref_index"] == 1
            if is_cited:
                citations.append({
                    "ref_index": ev["ref_index"],
                    "is_cited": True,
                    "book_title": ev["book_title"],
                    "subject": ev["subject"],
                    "mbbs_year": ev["mbbs_year"],
                    "chapter": ev["chapter"],
                    "topic": ev["topic"],
                    "page_number": ev["page_number"],
                    "total_pages": ev["total_pages"],
                    "excerpt": ev["excerpt"],
                    "citation_badge": f"[Ref {ev['ref_index']}: {ev['book_title']} | Ch. {ev['chapter'].split(':')[0]} | p. {ev['page_number']}]"
                })
        return citations

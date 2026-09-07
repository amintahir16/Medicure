"""
Application configuration for MBBS AI Chatbot.
Manages environment variables, LLM provider settings, database paths,
and medical curriculum metadata.
"""

import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
BOOKS_DIR = BASE_DIR / "books"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "mbbs_knowledge.db"

# LLM Configurations
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline")  # 'gemini', 'groq', 'offline'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Curriculum Year Mappings
MBBS_YEARS = {
    "all": "All 5 Years Curriculum",
    "year1": "MBBS 1st Year (Anatomy, Physiology, Biochemistry)",
    "year2": "MBBS 2nd Year (Pathology, Pharmacology, Microbiology)",
    "year3": "MBBS 3rd Year (Community Medicine, Ophthalmology, ENT, Forensic)",
    "year4_5": "MBBS 4th & Final Year (Internal Medicine, General Surgery, OBGYN, Pediatrics)"
}

SUBJECT_YEAR_MAP = {
    "Anatomy": "MBBS 1st Year",
    "Physiology": "MBBS 1st Year",
    "Biochemistry": "MBBS 1st Year",
    "Pathology": "MBBS 2nd Year",
    "Pharmacology": "MBBS 2nd Year",
    "Microbiology": "MBBS 2nd Year",
    "Forensic Medicine": "MBBS 3rd Year",
    "Community Medicine": "MBBS 3rd Year",
    "Internal Medicine": "MBBS 3rd to Final Year",
    "General Surgery": "MBBS 4th and Final Year",
    "Pediatrics": "MBBS 4th and Final Year",
    "Obstetrics and Gynecology": "MBBS 4th and Final Year"
}

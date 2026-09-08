"""
PDF Parser for MBBS Medical Textbooks
Extracts structured pages with exact physical page numbers, chapter headers,
topic titles, subject classification, and clinical tables.
"""

import re
from pathlib import Path
from pypdf import PdfReader
from backend.config import SUBJECT_YEAR_MAP

class PDFBookParser:
    def __init__(self):
        self.chapter_regex = re.compile(
            r'^(?:Chapter\s+\d+|SECTION\s+\d+|UNIT\s+\d+)[:\s\-\–](.+)', 
            re.IGNORECASE | re.MULTILINE
        )
        self.page_num_regex = re.compile(
            r'(?:Page\s+(\d+)\s+of\s+\d+|\b(\d+)\s*$)', 
            re.MULTILINE
        )

    def parse_book(self, pdf_path: Path):
        """
        Parses a medical textbook PDF into structured page records.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        
        # Determine book title and subject
        book_title = self._infer_book_title(pdf_path, reader)
        subject = self._infer_subject(book_title, pdf_path.stem)
        mbbs_year = SUBJECT_YEAR_MAP.get(subject, "MBBS Medical Reference")

        pages_data = []
        current_chapter = "Chapter 1: Introductory Principles"
        current_topic = "Clinical Overview"

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1  # 1-indexed physical page
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                print(f"Warning: error reading page {page_num} in {pdf_path.name}: {e}")
                continue

            if not raw_text.strip():
                continue

            if page_num % 50 == 0 or page_num == total_pages:
                print(f"Parsing '{book_title}': page {page_num}/{total_pages}...")

            # Detect chapter and topic headers from first 8 lines
            lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
            skip_idx = 0
            
            # 1. Identify Chapter Title
            for idx, line in enumerate(lines[:4]):
                if re.match(r'^(?:Chapter\s+\d+|SECTION\s+\d+|UNIT\s+\d+)', line, re.IGNORECASE):
                    # If chapter line already contains the title (has colon with >=4 characters after it)
                    if ':' in line and len(line.split(':', 1)[1].strip()) >= 4:
                        current_chapter = line.strip()
                        skip_idx = idx + 1
                    elif idx + 1 < len(lines) and not lines[idx+1].endswith('.'):
                        current_chapter = f"{line.strip()}: {lines[idx+1].strip()}"
                        skip_idx = idx + 2
                    else:
                        current_chapter = line.strip()
                        skip_idx = idx + 1
                    break

            # 2. Identify Topic Heading from lines following Chapter Title (ignoring running headers)
            table_words = {"pillar", "sign", "symptom", "category", "characteristic", 
                           "parameter", "substance", "direction", "lesion", "phase", 
                           "determinant", "pattern", "step", "drug", "generation", "score"}
            book_lower = book_title.lower()
            stem_lower = pdf_path.stem.lower()

            for line in lines[skip_idx:skip_idx+6]:
                line_lower = line.lower()
                # Skip running headers (e.g., "32 I Handbook of General Anatomy" or containing book title)
                if (re.match(r'^\d+\s*[\sI\|\-\–\\\/]', line) or 
                    re.search(r'[\sI\|\-\–\\\/]\s*\d+\s*$', line) or
                    book_lower in line_lower or stem_lower in line_lower or
                    "handbook of" in line_lower or "textbook of" in line_lower or "principles of" in line_lower):
                    continue

                # Must be a valid medical heading: not a sentence, not a page number, not a table column
                if (len(line) >= 4 and len(line) < 110
                    and not line_lower.startswith("page ")
                    and not line_lower.startswith("chapter ")
                    and not line_lower.startswith("clinical pearl")
                    and not line.endswith('.')
                    and not any(line_lower == tw for tw in table_words)
                    and not any(line_lower.startswith(tw + " ") for tw in ["table", "shock class"])
                    and any(c.isupper() for c in line)):
                    current_topic = line.strip()
                    break

            # Try to find printed page number from footer if present
            printed_page = page_num
            footer_match = re.search(r'Page\s+(\d+)\s+of\s+\d+', raw_text)
            if footer_match:
                try:
                    printed_page = int(footer_match.group(1))
                except ValueError:
                    pass

            cleaned_content = self._clean_medical_text(raw_text)

            pages_data.append({
                "book_title": book_title,
                "subject": subject,
                "mbbs_year": mbbs_year,
                "chapter": current_chapter,
                "topic": current_topic,
                "page_number": printed_page,
                "physical_page": page_num,
                "total_book_pages": total_pages,
                "content": cleaned_content,
                "raw_excerpt": cleaned_content[:350] + "..." if len(cleaned_content) > 350 else cleaned_content
            })

        return pages_data

    def _infer_book_title(self, pdf_path: Path, reader: PdfReader) -> str:
        # Check PDF metadata first
        if reader.metadata and reader.metadata.title:
            meta_title = reader.metadata.title.strip()
            if len(meta_title) > 3 and not meta_title.lower().endswith(".pdf") and "anonymous" not in meta_title.lower():
                return meta_title

        # Check first page text for known titles
        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text() or ""
            first_lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
            for candidate in first_lines[:4]:
                cand_clean = candidate.strip()
                if (len(cand_clean) > 5 and len(cand_clean) < 95 
                    and "page " not in cand_clean.lower()
                    and "curriculum" not in cand_clean.lower()
                    and "anonymous" not in cand_clean.lower()):
                    return cand_clean

        # Fallback to cleaned filename
        name = pdf_path.stem.replace("_", " ").replace("-", " ")
        name = re.sub(r'MBBS\s*\d*(?:_\d*)?', '', name, flags=re.IGNORECASE).strip()
        return name.title()

    def _infer_subject(self, title: str, filename: str) -> str:
        combined = (title + " " + filename).lower()
        if "anat" in combined or "neuroanat" in combined:
            return "Anatomy"
        elif "physio" in combined:
            return "Physiology"
        elif "patho" in combined:
            return "Pathology"
        elif "pharma" in combined or "drug" in combined:
            return "Pharmacology"
        elif "surg" in combined or "trauma" in combined:
            return "General Surgery"
        elif "microbio" in combined or "immune" in combined:
            return "Microbiology"
        elif "biochem" in combined:
            return "Biochemistry"
        elif "pediatric" in combined or "paediatric" in combined:
            return "Pediatrics"
        elif "gynecol" in combined or "obstet" in combined:
            return "Obstetrics and Gynecology"
        elif "forensic" in combined or "toxicolog" in combined:
            return "Forensic Medicine"
        elif "community" in combined or "preventive" in combined or "psm" in combined or "public health" in combined:
            return "Community Medicine"
        elif "ophthal" in combined or "eye" in combined:
            return "Ophthalmology"
        elif "ent" in combined or "otolaryng" in combined or "ear" in combined:
            return "Otorhinolaryngology (ENT)"
        elif "ortho" in combined:
            return "Orthopedics"
        elif "derma" in combined or "skin" in combined:
            return "Dermatology"
        elif "psychiat" in combined:
            return "Psychiatry"
        elif "medicine" in combined or "clinical" in combined or "internal" in combined:
            return "Internal Medicine"
        return "Internal Medicine"

    def _clean_medical_text(self, text: str) -> str:
        # Remove repeated header lines
        text = re.sub(r'MBBS\s+Clinical\s+Reference[^\n]*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'CURRICULUM\s+MEDICAL\s+REFERENCE[^\n]*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'MBBS\s+Medical\s+Knowledge\s+Base[^\n]*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

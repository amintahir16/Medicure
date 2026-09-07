"""
High-Yield MBBS Clinical Vignettes and Viva Questions
Provides quick-start prompts, case studies, and examination flashcards.
"""

SAMPLE_PROMPTS = [
    {
        "id": "cardiac_cycle",
        "title": "Cardiac Cycle & Wiggers Diagram",
        "category": "Physiology",
        "year": "MBBS 1st Year",
        "query": "Explain the detailed mechanical phases of the cardiac cycle, heart sounds S1 to S4, and Wiggers diagram correlation.",
        "icon": "heart"
    },
    {
        "id": "ace_inhibitors",
        "title": "ACE Inhibitors vs ARBs",
        "category": "Pharmacology",
        "year": "MBBS 2nd Year",
        "query": "Explain the mechanism of action of ACE inhibitors and why they cause a persistent dry cough compared to ARBs, citing the pharmacology textbook.",
        "icon": "pill"
    },
    {
        "id": "brachial_plexus",
        "title": "Erb's Palsy vs Klumpke's Palsy",
        "category": "Anatomy",
        "year": "MBBS 1st Year",
        "query": "Describe the anatomical roots and clinical presentation of Erb's Palsy ('waiter's tip') versus Klumpke's Palsy, with exact anatomical boundaries.",
        "icon": "bone"
    },
    {
        "id": "mi_pathology",
        "title": "Myocardial Infarction Timeline",
        "category": "Pathology",
        "year": "MBBS 2nd Year",
        "query": "Outline the chronological light microscopic histopathology of myocardial infarction from 0 hours to 2 weeks, noting when free wall rupture occurs.",
        "icon": "microscope"
    },
    {
        "id": "atls_trauma",
        "title": "Tension Pneumothorax & ATLS",
        "category": "General Surgery",
        "year": "MBBS 4th & Final Year",
        "query": "Detail the ATLS primary survey diagnosis and immediate bedside management of tension pneumothorax versus cardiac tamponade.",
        "icon": "syringe"
    },
    {
        "id": "stemi_management",
        "title": "Acute STEMI vs NSTEMI Protocol",
        "category": "Internal Medicine",
        "year": "MBBS 3rd to Final Year",
        "query": "What is the emergency triage and reperfusion protocol for acute STEMI within door-to-balloon time, and how does it differ from NSTEMI?",
        "icon": "activity"
    }
]

STUDY_VIVA_BANK = [
    {
        "question": "A neonate is delivered after shoulder dystocia with the arm adducted, internally rotated, and forearm pronated. Name the lesion, nerve roots involved, and muscles paralyzed.",
        "expected_book": "Human Anatomy and Neuroanatomy Principles",
        "expected_page": 2,
        "keywords": ["Erb's palsy", "C5", "C6", "waiter's tip", "suprascapular", "musculocutaneous"]
    },
    {
        "question": "Why does switching an anti-hypertensive patient from Enalapril to Losartan relieve their persistent non-productive dry cough?",
        "expected_book": "Essentials of Medical Pharmacology: Tripathi Principles",
        "expected_page": 2,
        "keywords": ["bradykinin", "kininase II", "substance P", "AT1 receptor", "cough"]
    },
    {
        "question": "At what time interval post-MI is the risk of ventricular free wall rupture highest, and what is the underlying histological mechanism?",
        "expected_book": "Pathologic Basis of Disease: Robbins Principles",
        "expected_page": 4,
        "keywords": ["3 to 7 days", "macrophages", "phagocytosis", "softening", "rupture"]
    },
    {
        "question": "Explain the immediate bedside intervention for a tension pneumothorax. Why must you NOT await a chest radiograph?",
        "expected_book": "Principles of General Surgery and Critical Care",
        "expected_page": 2,
        "keywords": ["needle thoracostomy", "2nd intercostal space", "clinical diagnosis", "tube thoracostomy"]
    }
]

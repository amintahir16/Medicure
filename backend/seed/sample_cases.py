"""
High-Yield MBBS Clinical Vignettes and Viva Questions
Provides quick-start prompts, case studies, and examination flashcards.
"""

SAMPLE_PROMPTS = [
    {
        "id": "bone_classification",
        "title": "Classification of Bones by Shape",
        "category": "Anatomy",
        "year": "MBBS 1st Year",
        "query": "What is the classification of bones according to shape in general anatomy? Explain long bones, short bones, flat bones, irregular, and sesamoid bones.",
        "icon": "bone"
    },
    {
        "id": "bone_parts_blood_supply",
        "title": "Parts & Blood Supply of Long Bones",
        "category": "Anatomy",
        "year": "MBBS 1st Year",
        "query": "Describe the anatomical parts of a young long bone (diaphysis, metaphysis, epiphysis) and their blood supply including nutrient and periosteal arteries.",
        "icon": "bone"
    },
    {
        "id": "joints_classification",
        "title": "Classification & Structure of Joints",
        "category": "Anatomy",
        "year": "MBBS 1st Year",
        "query": "Explain the structural and functional classification of joints (fibrous, cartilaginous, synovial) and characteristics of synovial joints.",
        "icon": "activity"
    },
    {
        "id": "cardiac_cycle",
        "title": "Cardiac Cycle & Wiggers Diagram",
        "category": "Physiology",
        "year": "MBBS 1st Year",
        "query": "Explain the mechanical phases of the cardiac cycle, heart sounds S1 to S4, and Wiggers diagram correlation.",
        "icon": "heart"
    },
    {
        "id": "ace_inhibitors",
        "title": "ACE Inhibitors vs ARBs",
        "category": "Pharmacology",
        "year": "MBBS 2nd Year",
        "query": "Explain the mechanism of action of ACE inhibitors and why they cause a persistent dry cough compared to ARBs, citing textbook pharmacology.",
        "icon": "pill"
    },
    {
        "id": "atls_trauma",
        "title": "Tension Pneumothorax & ATLS",
        "category": "General Surgery",
        "year": "MBBS 4th & Final Year",
        "query": "Detail the ATLS primary survey diagnosis and immediate bedside management of tension pneumothorax versus cardiac tamponade.",
        "icon": "syringe"
    }
]

STUDY_VIVA_BANK = [
    {
        "question": "Name the four vascular systems that supply a typical young long bone, and which vessel is responsible for supplying the active epiphyseal growth plate?",
        "expected_book": "BD Chaurasia's Handbook of General Anatomy",
        "expected_page": 50,
        "keywords": ["nutrient artery", "epiphyseal arteries", "metaphyseal arteries", "periosteal arteries", "growth cartilage"]
    },
    {
        "question": "What are sesamoid bones, what are their two main clinical/functional advantages, and give two classic anatomical examples in the human body?",
        "expected_book": "BD Chaurasia's Handbook of General Anatomy",
        "expected_page": 45,
        "keywords": ["sesamoid", "patella", "pisiform", "tendon", "friction", "mechanical advantage"]
    },
    {
        "question": "State Hilton's Law regarding the nerve supply of joints and explain its clinical significance in referred joint pain.",
        "expected_book": "BD Chaurasia's Handbook of General Anatomy",
        "expected_page": 66,
        "keywords": ["Hilton's law", "nerve supply", "muscles crossing joint", "skin over joint", "referred pain"]
    },
    {
        "question": "What is the structural and developmental difference between primary cartilaginous joints (synchondroses) and secondary cartilaginous joints (symphyses)?",
        "expected_book": "BD Chaurasia's Handbook of General Anatomy",
        "expected_page": 63,
        "keywords": ["synchondrosis", "symphysis", "hyaline cartilage", "fibrocartilage", "median plane"]
    }
]

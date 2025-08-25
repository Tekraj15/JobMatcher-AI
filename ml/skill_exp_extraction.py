import re
from typing import Tuple, Set, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from ml.config import TECH_KEYWORDS, SYNONYMS
from datetime import datetime

def normalize_token(token: str) -> str:
    token = token.lower().strip(".,/|·- ")
    return SYNONYMS.get(token, token)

def extract_skills_and_exp(text: str) -> Tuple[Set[str], Set[str], Optional[float]]:
    # Normalize text
    text = re.sub(r'\s+', ' ', text.lower())

    # Heading detection
    required_headings = ['requirements', 'must have', 'required', 'essential']
    optional_headings = ['nice to have', 'preferred', 'bonus', 'desirable']

    lines = text.splitlines()
    required_skills = set()
    optional_skills = set()
    current_section = None

    for line in lines:
        if any(h in line for h in required_headings):
            current_section = 'required'
        elif any(h in line for h in optional_headings):
            current_section = 'optional'
        elif current_section:
            # Tokenize
            tokens = re.split(r',|/|\|·|and|or|\s+', line)
            for t in tokens:
                norm_t = normalize_token(t)
                if norm_t in TECH_KEYWORDS:
                    if current_section == 'required':
                        required_skills.add(norm_t)
                    elif current_section == 'optional':
                        optional_skills.add(norm_t)

    # Fallback if no headings
    if not required_skills and not optional_skills:
        vectorizer = TfidfVectorizer(vocabulary=TECH_KEYWORDS)
        tfidf = vectorizer.fit_transform([text])
        scores = tfidf.toarray()[0]
        top_indices = scores.argsort()[-3:][::-1]  # Top 3
        required_skills = {TECH_KEYWORDS[i] for i in top_indices if scores[i] > 0}

    # Experience extraction
    exp_patterns = [
        r'(\d+)\+?\s*(?:years|yrs|year)',
        r'(\d+)\s*-\s*(\d+)\s*(?:years|yrs)',
        r'minimum of (\d+) years',
        r'experience: (\d+) years',
    ]
    for pattern in exp_patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) == 1:
                return required_skills, optional_skills, float(match.group(1))
            elif len(match.groups()) == 2:
                return required_skills, optional_skills, float(match.group(1))  # Take min for ranges

    return required_skills, optional_skills, None

def extract_resume_skills_and_exp(parsed_sections: dict) -> Tuple[Set[str], Optional[float]]:
    skills = set()
    exp = None

    # Skills from 'skills' section
    skills_section = parsed_sections.get('skills', [])
    for line in skills_section:
        tokens = re.split(r',|/|\|·|and|or|\s+', line)
        for t in tokens:
            norm_t = normalize_token(t)
            if norm_t in TECH_KEYWORDS:
                skills.add(norm_t)

    # Experience from 'experience' section
    exp_section = parsed_sections.get('experience', [])
    total_exp = 0.0
    date_pattern = r'(\d{4})\s*-\s*(present|\d{4})'
    for line in exp_section:
        match = re.search(date_pattern, line)
        if match:
            start = int(match.group(1))
            end = datetime.now().year if match.group(2).lower() == 'present' else int(match.group(2))
            total_exp += end - start

    if total_exp > 0:
        exp = total_exp

    # Fallback explicit phrase
    if exp is None:
        for line in exp_section:
            match = re.search(r'(\d+)\+?\s*(?:years|yrs|year)', line)
            if match:
                exp = float(match.group(1))
                break

    return skills, exp

import re
import spacy

nlp = spacy.load("en_core_web_sm")

EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}'
PHONE_PATTERN = r'\b\d{10}\b'

def mask_pii(text):
    entities = []
    masked_text = text

    # Regex-based masking
    for pattern, label in [(EMAIL_PATTERN, "EMAIL"), (PHONE_PATTERN, "PHONE")]:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            original = match.group()
            if original in masked_text:
                masked_text = masked_text.replace(original, f"[{label}]")
                entities.append({
                    "position": [start, end],
                    "classification": label,
                    "entity": original
                })

    # NER-based masking
    doc = nlp(masked_text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            start, end = ent.start_char, ent.end_char
            original = ent.text
            label = ent.label_
            if original in masked_text:
                masked_text = masked_text.replace(original, f"[{label}]")
                entities.append({
                    "position": [start, end],
                    "classification": label,
                    "entity": original
                })

    return masked_text, entities


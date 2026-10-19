# %% Step 1
import pandas as pd
from collections import defaultdict

def recall_by_type(find, notes=DEID_NOTES):
    hits, totals = defaultdict(int), defaultdict(int)
    found_strings_all, false_positives = [], 0
    for _, text, gold in notes:
        found = [s[3] for s in find(text)]
        gold_values = [v for vs in gold.values() for v in vs]
        for kind, values in gold.items():
            for v in values:
                totals[kind] += 1
                hits[kind] += any(v in f or f in v for f in found)
        false_positives += sum(not any(f in g or g in f for g in gold_values) for f in found)
    table = pd.DataFrame({"marked": totals, "found": hits}).fillna(0).astype(int)
    table["recall"] = (table.found / table.marked).round(2)
    print(table.sort_values("recall").to_string())
    print(f"overall recall {table.found.sum() / table.marked.sum():.0%}; false positives {false_positives}")
    return table

baseline = recall_by_type(find_phi)

# %% Step 2
PATTERNS_V2 = PATTERNS + [
    ("ADDRESS", r"\b(\d{1,4}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Road|Street|Lane|Avenue|Nagar))\b"),
    ("NAME", r"\b(?:[Hh]is|[Hh]er|[Tt]heir)\s+(?:son|daughter|wife|husband|brother|sister|mother|father)\s+([A-Z][a-z]+)"),
    ("FACILITY", r"\b((?:[A-Z][a-z]+\s+){1,2}(?:Clinic|Hospital))\b"),
    ("NAME", r"\b(?:Patient|Pt)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)"),
    ("NAME", r"^([A-Z][a-z]+\s+[A-Z][a-z]+)(?=\s+(?:reviewed|discharged|attended|,))"),
    ("ID", r"\bMRN[:\s]*,?\s*(\d{6,})"),
]

def find_phi_v2(text):
    spans = []
    for label, pattern in PATTERNS_V2:
        for m in re.finditer(pattern, text, flags=re.MULTILINE):
            spans.append((m.start(1), m.end(1), label, m.group(1)))
    # remove spans contained in a longer span
    spans = sorted(set(spans), key=lambda s: (s[0], -s[1]))
    kept = [s for s in spans if not any(o[0] <= s[0] and s[1] <= o[1] and o != s for o in spans)]
    return kept

improved = recall_by_type(find_phi_v2)

# %% Note
# The new patterns find addresses, relatives named after relationship words, facility names and untitled names at the
# start of a note. Watch the false-positive count: "Central Hospital" is right, but a pattern for "two capitalised
# words" would also catch "Type One" or "Lake View"; precision matters because over-redaction destroys clinical meaning.
# What is still missed (look at the table) usually needs a statistical name recogniser rather than another rule.

# %% Step 3
import hashlib

FIRST = ["Anita", "Bala", "Chen", "Deepa", "Elena", "Farid", "Gita", "Hari", "Ines", "Jamal", "Kavya", "Liam"]
LAST = ["Iyer", "Smith", "Okoro", "Sato", "Kumar", "Silva", "Novak", "Haddad", "Menon", "Wright"]

def surrogate_name(original, patient_key):
    """Same original name (within one patient's notes) -> same made-up name; different patients get different maps."""
    h = int(hashlib.sha256(f"{patient_key}|{original}".encode()).hexdigest(), 16)
    parts = original.split()
    return FIRST[h % len(FIRST)] + (f" {LAST[(h // 7) % len(LAST)]}" if len(parts) > 1 else "")

def pseudonymise(text, patient_key):
    for start, end, label, value in sorted(find_phi_v2(text), reverse=True):
        if label == "NAME":
            replacement = surrogate_name(value, patient_key)
        elif label == "DATE":
            replacement = (datetime.strptime(value, "%d/%m/%Y") + SHIFT).strftime("%d/%m/%Y")
        else:
            replacement = f"[{label}]"
        text = text[:start] + replacement + text[end:]
    return text

for patient, text, _ in DEID_NOTES[:5]:
    print(f"{patient}: {pseudonymise(text, patient)}")

# %% Note
# Consistency works where the same string recurs: "Asha Verma" and "Ravi" get the same surrogates in each of patient
# A's notes, so the notes still read as one story. The output also shows three problems a reviewer must catch:
#
# - **Partial names**: "Mrs Verma" (surname only) is replaced by a different surrogate from "Asha Verma"; names need
#   normalising (linking surname-only mentions to the full name) before surrogates are assigned.
# - **Gender and type**: the surrogate list ignores gender, so "Mr John Mathew" and "his wife Mary" get names that do
#   not fit, and a surname ("Dr Lee") can become a first name. Real surrogate generators keep gender, name part and,
#   ideally, cultural plausibility.
# - **Secrecy**: the mapping key (here, the patient key plus a fixed hash) must be kept secret, or the surrogates can be
#   reversed by trial.

# %% Step 4
# %% Note
# Optional (outline, not run here: Presidio needs a spaCy language model of several hundred megabytes).
#
# ```python
# # pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg
# from presidio_analyzer import AnalyzerEngine
# analyzer = AnalyzerEngine()
# def find_phi_presidio(text):
#     return [(r.start, r.end, r.entity_type, text[r.start:r.end])
#             for r in analyzer.analyze(text=text, language="en")]
# recall_by_type(find_phi_presidio)
# ```
#
# Presidio combines a named-entity model with pattern recognisers for emails, phone numbers and similar. Expect it to
# find more names than the rules, and to need custom recognisers for local formats (record numbers, Indian postcodes,
# Aadhaar or NHS numbers). Compare recall by identifier type and false positives on the same marked notes.

# %% Step 5
# %% Note
# Example data-release checklist for sharing de-identified notes with a research partner:
#
# - [ ] Approval: ethics or data-governance approval and a data-sharing agreement that forbids re-identification,
#       linkage and onward sharing, and sets retention and deletion.
# - [ ] Legal basis checked under the applicable law (for example GDPR/UK GDPR, HIPAA, India's DPDP Act).
# - [ ] Minimum necessary: only the notes and fields the project needs; free-text sections not needed are removed.
# - [ ] Automated de-identification run, with recall measured on a marked sample from the same source and note types.
# - [ ] Manual review of a random sample (and of all notes for rare conditions or unusual occupations, which can
#       identify people), with a stop rule if any identifier is found.
# - [ ] Dates shifted consistently per patient; ages over 89 grouped; small geographic areas generalised.
# - [ ] Surrogate mapping and date shifts stored separately, encrypted, and not shared.
# - [ ] Shared through a secure environment where possible, with access logging, rather than as a file copy.
# - [ ] Release documented: what was shared, when, with whom, and the measured residual risk.

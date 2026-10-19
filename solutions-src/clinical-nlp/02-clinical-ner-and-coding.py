# %% Step 1
import pandas as pd

CODES = list(LEXICON)

def per_code_scores(extractor, notes=CODED_NOTES):
    rows = []
    for code in CODES:
        tp = sum(code in extractor(t) and code in g for t, g in notes)
        fp = sum(code in extractor(t) and code not in g for t, g in notes)
        fn = sum(code not in extractor(t) and code in g for t, g in notes)
        rows.append({"code": code, "TP": tp, "FP": fp, "FN": fn,
                     "precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan")})
    df = pd.DataFrame(rows).set_index("code")
    tp, fp, fn = df.TP.sum(), df.FP.sum(), df.FN.sum()
    print(df.round(2).to_string())
    print(f"overall precision {tp / (tp + fp):.2f}, recall {tp / (tp + fn):.2f}")
    return df

base = per_code_scores(extract_codes)
print("\nErrors:")
for text, gold in CODED_NOTES:
    pred = extract_codes(text)
    if pred != gold:
        print(f"  missed {sorted(gold - pred)} extra {sorted(pred - gold)}  | {text}")

# %% Step 2
import copy

LEXICON_V1 = copy.deepcopy(LEXICON)
LEXICON["E11"] += ["dm2", "type 2 dm", "diabetes type 2", "type ii diabetes mellitus", "non-insulin-dependent diabetes"]
LEXICON["I10"] += ["essential hypertension", "raised bp"]
LEXICON["N18"] += ["chronic renal failure", "ckd 3", "ckd 3a", "ckd 4", "stage 3 chronic kidney disease"]
LEXICON["L97"] += ["heel ulcer", "ulcer on the heel", "ulceration of the left foot", "neuropathic ulcer",
                   "foot ulcer", "ulcer", "ulceration"]
LEXICON["E78"] += ["hypercholesterolaemia", "raised cholesterol", "hyperlipidemia", "dyslipidemia"]

after_synonyms = per_code_scores(extract_codes)

# %% Note
# Synonyms take recall from 0.70 to 1.00 here, and precision does not fall (0.86 to 0.90): the remaining false
# positives are the uncertain and family-history notes, not the new terms. On real notes, broad terms such as "ulcer"
# (an ulcer elsewhere) or "raised BP" (one high reading, not a diagnosis) would add false positives, so every lexicon
# change is re-scored on the whole labelled set.

# %% Step 3
UNCERTAIN = r"\b(possible|probable|query|\?|suspected|rule out|awaiting)\b"
FAMILY = r"\b(mother|father|sister|brother|family history)\b"

def extract_codes_v2(note):
    codes = set()
    for sentence in re.split(r"(?<=[.!?])\s+", note.lower()):
        for code, terms in LEXICON.items():
            for term in terms:
                m = re.search(rf"\b{re.escape(term)}\b", sentence)
                if not m:
                    continue
                before = sentence[:m.start()]
                if re.search(NEGATION, before) or re.search(UNCERTAIN, before) or re.search(FAMILY, sentence):
                    continue
                codes.add(code)
    return codes

after_context = per_code_scores(extract_codes_v2)
for text, gold in CODED_NOTES:
    pred = extract_codes_v2(text)
    if pred != gold:
        print(f"  still wrong: missed {sorted(gold - pred)} extra {sorted(pred - gold)}  | {text}")

# %% Note
# Uncertainty and family-history handling remove every false positive ("possible CKD", "query CKD", "rule out
# hypertension", "mother has type 2 diabetes", "father had hypertension"): precision 1.00, recall 0.98. The one new
# miss shows the cost of a blunt rule: "Raised cholesterol, family history of heart disease" loses the patient's own
# cholesterol because the family cue silences the whole sentence. A better rule limits the cue's scope to the words
# that follow it, as ConText does.

# %% Step 4
# Optional: medspaCy (pip install medspacy) with our lexicon as target rules and its ConText module for negation,
# uncertainty and family history. scispaCy's biomedical NER models are an alternative; they are installed from
# version-matched model files listed in the scispaCy documentation.
import medspacy
from loguru import logger
from medspacy.target_matcher import TargetRule

logger.remove()
nlp = medspacy.load()
def rule(term):
    # Single-word terms as token patterns: in our run, a plain literal rule for "t2dm" did not match "T2DM"
    words = term.split()
    return TargetRule(term, "CONDITION", pattern=[{"LOWER": w} for w in words]) if len(words) == 1 else TargetRule(term, "CONDITION")
nlp.get_pipe("medspacy_target_matcher").add([rule(term) for terms in LEXICON.values() for term in terms])

def extract_codes_medspacy(note):
    doc = nlp(note)
    return {next(code for code, terms in LEXICON.items() if e.text.lower() in terms)
            for e in doc.ents if not (e._.is_negated or e._.is_uncertain or e._.is_family or e._.is_hypothetical)}

after_medspacy = per_code_scores(extract_codes_medspacy)
for text, gold in CODED_NOTES:
    pred = extract_codes_medspacy(text)
    if pred != gold:
        doc = nlp(text)
        print(f"  medspaCy: missed {sorted(gold - pred)} extra {sorted(pred - gold)} | {text} | entities "
              f"{[(e.text, 'neg' if e._.is_negated else 'unc' if e._.is_uncertain else 'fam' if e._.is_family else 'hyp' if e._.is_hypothetical else 'ok') for e in doc.ents]}")

# %% Note
# Running a library is not the same as validating it. In our first run, medspaCy missed every "T2DM" (a plain literal
# rule did not match it; the token-pattern rule above fixes that). Read the printed disagreements: in our run ConText
# also did not treat "Query CKD" as uncertain, and did not negate "no ulcers" in "Foot check: no ulcers, pulses
# present.", although it did negate "No ulcers." on its own. Its curated cue lists help most on long, messy notes
# (cue scopes, post-position cues, terminating terms), but every local abbreviation and cue has to be added and
# tested. Neither approach knows which terms count as a codable diagnosis; that sits in the lexicon and coding rules.

# %% Step 5
# %% Note
# Optional, with an approved model and these synthetic notes only. Two designs to compare:
#
# ```python
# def ask(prompt: str) -> str:
#     """Call your approved model; return its text (API key from an environment variable)."""
#     raise NotImplementedError
#
# EXTRACT = ("List the conditions this synthetic note says the patient currently HAS (not negated, not uncertain, "
#            "not family history), one per line, in plain words. Note: {note}")
# CODE = ("Give the ICD-10 category codes (three characters) for conditions the patient currently has, "
#         "comma-separated, from this list only: E11, I10, N18, L97, E78. Note: {note}")
#
# def via_lexicon(note):                       # LLM extracts conditions, our lexicon maps them to codes
#     return set().union(*(extract_codes_v2(line) for line in ask(EXTRACT.format(note=note)).splitlines()))
#
# def direct(note):                            # LLM assigns codes itself
#     return {c.strip() for c in ask(CODE.format(note=note)).split(",") if c.strip() in LEXICON}
#
# per_code_scores(via_lexicon); per_code_scores(direct)
# ```
#
# Expect the LLM to handle negation and paraphrase well, and to make different errors: occasionally coding
# conditions that are implied but not stated, or inventing codes outside the list (which the filter removes). Mapping
# through a reviewed lexicon keeps the final codes auditable.

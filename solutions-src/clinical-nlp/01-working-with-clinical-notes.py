# %% Step 1
FINDINGS += ["ulcer", "fever", "nausea", "hypos", "headache", "oedema", "rest pain", "seizures"]
for patient, visit, text in clinic_notes[:4]:
    sections = split_sections(text)
    print(f"{patient} visit {visit}: sections {list(sections)}")
    for finding, status in findings_with_status(expand(text)):
        print(f"   {status:<8} {finding}")

# %% Note
# The pipeline runs on every note, but look at the output: "ulcer" also matches inside "ulcers", and the negation
# window is the whole preceding sentence, so "No ulcers. Pedal pulses present." works only because the sentence
# splitter separates them. Rules are fragile in exactly these small ways.

# %% Step 2
UNCERTAINTY_CUES = r"\b(probable|possible|query|\?|suspected|rule out|likely)"
FAMILY_CUES = r"\b(mother|father|sister|brother|family history of|son|daughter)\b"
CONDITIONS = ["diabetic peripheral neuropathy", "masld", "fatty liver", "peripheral arterial disease",
              "kidney disease", "diabetes", "coeliac", "heart attack", "type 2 diabetes"]

def condition_status(text, terms=None):
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", expand(text)):
        s = sentence.lower()
        for c in (terms or CONDITIONS):
            pos = s.find(c)
            if pos < 0:
                continue
            before = s[:pos]
            if re.search(FAMILY_CUES, s):
                status = "family history"
            elif re.search(NEGATION_CUES, before):
                status = "absent"
            elif re.search(UNCERTAINTY_CUES, before):
                status = "uncertain"
            else:
                status = "present"
            out.append((c, status, sentence.strip()))
    return out

for patient, visit, text in clinic_notes:
    for c, status, sentence in condition_status(text):
        if status != "present":
            print(f"{patient} v{visit}  {status:<15} {c:<32} | {sentence}")

# %% Note
# Family-history cues are checked across the whole sentence (the relative is often named after the condition, as in
# "kidney disease (mother)"), and before negation and uncertainty. "Rule out type 2 diabetes" and "?coeliac" now
# count as uncertain rather than present.

# %% Step 3
failing = [
    ("No improvement in neuropathy pain.", "negation cue applies to 'improvement', but the rule marks neuropathy absent"),
    ("Neuropathy: none.", "the negation comes after the finding, outside the look-back window"),
    ("Chest pain resolved last year.", "historical, not current, but counted as present"),
    ("Mother's neighbour has diabetes; patient's own HbA1c is normal.", "family cue triggers on 'mother' though no relative has the condition"),
    ("Pt not keen on insulin; no evidence of retinopathy but possible neuropathy.", "two findings with different status in one sentence; the first negation cue covers both"),
]
TERMS = ["neuropathy", "retinopathy", "chest pain", "diabetes"]
for text, why in failing:
    print(f"{text}\n   rules say: {[(c, s) for c, s, _ in condition_status(text, TERMS)]}\n   why it fails: {why}\n")

# %% Note
# These are the classic NegEx/ConText failure modes: negation scope (what a cue applies to), post-position cues,
# temporality, the "experiencer" (who has the condition) and multiple findings per sentence. Each can be patched with
# another rule, and each patch creates new failures; this is why clinical NLP moved to statistical models, which still
# need testing on exactly these patterns.

# %% Step 4
import pandas as pd

rows = []
for patient in sorted({p for p, _, _ in clinic_notes}):
    visits = [(v, t) for p, v, t in clinic_notes if p == patient]
    for (v_prev, t_prev), (v, t) in zip(visits, visits[1:]):
        sim = difflib.SequenceMatcher(None, t_prev, t).ratio()
        rows.append({"patient": patient, "visit": v, "vs visit": v_prev, "identical": round(sim, 3), "flag": sim > 0.9})
copy_forward = pd.DataFrame(rows)
print(copy_forward.to_string(index=False))
print(f"\nNotes flagged as >90% identical to the previous one: {copy_forward.flag.sum()} of {len(copy_forward)}")

# %% Note
# Each synthetic patient has one visit copied forward unchanged (100% identical), which the check flags. In real
# records, copied text carries stale findings forward ("tingling x 2 mo" months later), so a model trained on notes can
# learn the copying, not the patient. Flagged notes should be de-duplicated or down-weighted before training.

# %% Step 5
# Optional: medspaCy's ConText implementation on the same notes (pip install medspacy).
import medspacy
from loguru import logger
from medspacy.target_matcher import TargetRule

logger.remove()                                  # silence the sentence splitter's debug messages
nlp = medspacy.load()
nlp.get_pipe("medspacy_target_matcher").add([TargetRule(c, "CONDITION") for c in CONDITIONS])

agree = disagree = 0
for patient, visit, text in clinic_notes:
    doc = nlp(expand(text))
    ours = {(c, status) for c, status, _ in condition_status(text)}
    for ent in doc.ents:
        theirs = ("family history" if ent._.is_family else "absent" if ent._.is_negated
                  else "uncertain" if ent._.is_uncertain else "present")
        match = (ent.text.lower(), theirs) in ours
        agree += match; disagree += not match
        if not match:
            print(f"{patient} v{visit}: medspaCy says {ent.text!r} is {theirs}")
print(f"\nAgreement with our rules on {agree} of {agree + disagree} condition mentions")

# %% Note
# The two agree on most mentions. Where they disagree here, medspaCy's default cue list treated "Query fatty liver" and
# "Probable MASLD" as present: our rules include "query" and "probable" as uncertainty cues, and its default
# configuration, as run here, did not flag these sentences. ConText's strength is its curated cue lists with explicit
# scopes and directions (forward- and backward-looking cues, terms that end a scope); its weakness, like ours, is any
# cue it was not given. Add local cues ("query", "?", "r/o") to its rules, and read every disagreement: neither is a
# reference standard.

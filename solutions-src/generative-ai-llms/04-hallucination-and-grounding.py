# %% Step 1
for case_no in [1, 3]:                      # review two notes (cases 2 and 4, counting from 1)
    case = cases[case_no]
    print(f"Case {case_no + 1} transcript:\n  {case['transcript']}\n")
    labelled = {i: (kind, severity) for i, kind, severity in case["errors"]}
    for i, sentence in enumerate(case["draft"]):
        kind, severity = labelled.get(i, ("supported", ""))
        print(f"  [{kind:<13} {severity:<5}] {sentence}")
    print()

# %% Note
# Case 2: "Small ulcer on the left heel" contradicts the patient's answer that there are no wounds, and an
# invented ulcer could trigger unnecessary wound care or alarm; it is major. Case 4: "Reports chest pain on
# exertion" reverses "No chest pain", a major error because it changes the urgency of the assessment. The other
# sentences match the transcript. Doing this review yourself before looking at the labels is the point of
# the step: you will notice how slow it is, which is why automated checks are tempting.

# %% Step 2
n_sentences = sum(len(c["draft"]) for c in cases)
errors = [(k, kind, sev) for k, c in enumerate(cases) for _, kind, sev in c["errors"]]
omissions = [(k, item, sev) for k, c in enumerate(cases) for item, sev in c["omissions"]]
notes_with_error = sum(1 for c in cases if c["errors"] or c["omissions"])
print(f"{n_sentences} draft sentences, {len(errors)} with an error ({len(errors) / n_sentences:.1%})")
print(f"  hallucinations: {sum(e[1] == 'hallucination' for e in errors)}, contradictions: {sum(e[1] == 'contradiction' for e in errors)}")
print(f"  major: {sum(e[2] == 'major' for e in errors)}, minor: {sum(e[2] == 'minor' for e in errors)}")
print(f"Omissions: {len(omissions)} ({sum(o[2] == 'major' for o in omissions)} major)")
print(f"Notes with at least one error or omission: {notes_with_error} of {len(cases)}")

# %% Note
# The per-sentence rate looks reassuring; the per-note rate shows that most notes need a correction. Both
# should be reported, together with severity, because one major error in a note matters more than several minor ones.

# %% Step 3
def check_case(case, sentence_checker):
    src = case["transcript"].lower()
    nums = set(re.findall(r"\d+(?:\.\d+)?", src))
    return [bool(sentence_checker(s, src, nums)) for s in case["draft"]]

def number_medicine_check(sentence, src, nums):
    s = sentence.lower()
    problems = [f"number {n}" for n in re.findall(r"\d+(?:\.\d+)?", s) if n not in nums]
    problems += [f"medicine {m}" for m in MEDICINES if m in s and m not in src]
    return problems

def evaluate(sentence_checker, label):
    found, false_alarms = 0, 0
    for case in cases:
        flags = check_case(case, sentence_checker)
        error_idx = {i for i, _, _ in case["errors"]}
        found += sum(flags[i] for i in error_idx)
        false_alarms += sum(f for i, f in enumerate(flags) if i not in error_idx)
    print(f"{label:<30} found {found} of {len(errors)} labelled errors (sensitivity {found / len(errors):.0%}), "
          f"false alarms {false_alarms}")

evaluate(number_medicine_check, "Numbers and medicines")

# %% Note
# The checker finds invented doses, numbers and medicines (4 of 9 errors) with no false alarms, but it is blind
# to errors made of words: an invented ulcer, a reversed symptom, or "dose increased" instead of "continue".

# %% Step 4
SYMPTOMS = {   # concept: words that express it in drafts or transcripts
    "chest pain": ["chest pain"], "ulcer": ["ulcer", "wound"], "swelling": ["swelling"],
    "urinary symptoms": ["urinary symptoms"], "hypoglycaemia": ["hypoglycaemia", "low sugar"],
    "foot problems": ["foot problems"], "breathlessness": ["breathlessness", "shortness of breath"],
}
NEG = r"\b(no|denies|without|not)\b"

def mentioned(text, words):
    return any(w in text for w in words)

def negated_in_source(src, words):
    for w in words:
        if re.search(NEG + r"[^.?]{0,30}" + re.escape(w), src):                 # "no chest pain"
            return True
        if re.search(r"any " + re.escape(w) + r"[^?]*\?\s*patient:\s*no\b", src):  # "Any wounds? Patient: No."
            return True
    return False

def negation_check(sentence, src, nums):
    s = sentence.lower()
    problems = []
    for concept, words in SYMPTOMS.items():
        if not mentioned(s, words):
            continue
        draft_negated = re.search(NEG, s) is not None
        if mentioned(src, words):
            src_negated = negated_in_source(src, words)
            if draft_negated != src_negated:
                problems.append(f"{concept}: draft {'denies' if draft_negated else 'affirms'}, transcript does the opposite")
    return problems

combined = lambda s, src, nums: number_medicine_check(s, src, nums) + negation_check(s, src, nums)
evaluate(negation_check, "Negation rule only")
evaluate(combined, "Numbers, medicines + negation")

# %% Note
# The negation rule catches the reversed symptoms (the invented ulcer, the chest pain and "no swelling"),
# taking sensitivity from 4 to 7 of 9 here. The remaining misses show the limit of rules: "Dose increased"
# contradicts "continue the same dose" without any number or symptom word, and "shortness of breath" is
# invented about something the transcript never mentions. Rules like these are useful as a first filter on
# synthetic data written for them; on real notes, with endless phrasings, they miss far more and need
# testing on a labelled sample before anyone relies on them.

# %% Step 5
# %% Note
# Example review protocol (AI-drafted outpatient notes, reviewed by the clinician who saw the patient):
#
# **Use:** drafts are suggestions. Nothing is filed, sent or acted on until the treating clinician has read
# and signed the note; the draft is never shown to the patient or other services unreviewed.
#
# **What the clinician checks, every note:**
# 1. Every number: doses, results, dates, intervals, measurements.
# 2. Every medicine and every change (started, stopped, increased, continued).
# 3. Every symptom, especially negatives ("no chest pain") and red flags.
# 4. Examinations: only those actually performed are recorded.
# 5. The plan and follow-up, and anything the patient was told (advice, safety-netting).
# 6. Allergies and anything said once in passing that the draft may have dropped.
#
# **Automated checks** (numbers, medicines, negations) highlight sentences for attention; they never replace
# the read-through, and a clean result is not a reason to skim.
#
# **Audit:** each month, a second clinician reviews a random 5% sample of signed notes against the recording
# or transcript (where consented and retained), grading errors as in step 2. Major errors that reached a signed
# note are reported through the incident system and reviewed with the vendor.
#
# **Stopping rule:** agreed in advance with the clinical safety officer (for example, any major error reaching
# a signed note that affected care, or a rising trend in the audit) triggers suspension and review.
#
# **Proportionality:** stricter for higher-risk settings (emergency, medication changes, discharge letters),
# lighter for low-risk administrative text.

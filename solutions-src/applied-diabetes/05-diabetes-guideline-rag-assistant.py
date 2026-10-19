# %% Step 1
def run_suite(suite, verbose=True):
    passed = 0
    for q, expected, fact in suite:
        r = safe_answer(q)
        ok = r["route"] == expected and (fact is None or fact in r["answer"])
        passed += ok
        if verbose and not ok:
            print(f"FAIL {q!r}: route {r['route']} (expected {expected})" + (f", '{fact}' not in answer" if fact else ""))
    print(f"{passed}/{len(suite)} release checks passed")
    return passed

print(json.dumps(audit_log[-1], indent=1))
run_suite(SUITE)

# %% Step 2
def build_index(protocol_text, version):
    """Rebuild the chunks and embeddings that answer() uses, from a protocol text, tagging each chunk's version."""
    global chunks, chunk_emb
    chunks = []
    for i, part in enumerate(re.split(r"\n(?=## )", protocol_text)[1:], start=1):
        heading, *body = part.strip().splitlines()
        chunks.append({"id": f"s{i}", "heading": heading.lstrip("# "), "version": version,
                       "sentences": re.split(r"(?<=\.)\s+", " ".join(body).strip())})
    chunk_emb = model.encode([c["heading"] + ". " + " ".join(c["sentences"]) for c in chunks], normalize_embeddings=True)
    DOCUMENT["version"] = version

PROTOCOL_V2 = PROTOCOL.replace("If eGFR falls below 30 during treatment, stop metformin.",
                               "If eGFR falls below 30 during treatment, stop metformin. Review the dose if eGFR is 30-44.")
build_index(PROTOCOL_V2, "2026-2")

SUITE_V2 = SUITE + [("What should be done with metformin if eGFR is 30-44?", "answered", "Review the dose")]
run_suite(SUITE_V2)

def version_check(question, expected_version):
    """Release check: the cited chunk must come from the current version of the protocol."""
    r = safe_answer(question)
    if r["route"] != "answered":
        return True
    cited = next(c for c in chunks if c["id"] == r["citations"][0].split(":")[0])
    return cited["version"] == expected_version

print("Version check passes on the current index:", version_check("When should metformin be stopped?", "2026-2"))
build_index(PROTOCOL, "2026-1")                                    # simulate a stale index
print("Version check on a stale index (should fail):", version_check("When should metformin be stopped?", "2026-2"))
build_index(PROTOCOL_V2, "2026-2")

# %% Note
# The new rule gets a fact check ("Review the dose" must appear), and the version check is a separate test that fails
# if the index was not rebuilt after the protocol changed, even when old and new texts give the same answer.
#
# The suite also caught a regression nobody intended: after the update, "What dose of semaglutide should I start?" is
# answered instead of declined, because the new sentence about metformin *dose* pulls that question towards the kidney
# section. A one-sentence change to one rule changed the behaviour of an unrelated question, which is exactly why the
# whole suite is re-run after every document change.

# %% Step 3
MORE_CHECKS = [
    # five urgent questions in patients' words
    ("my chest feels really tight and I'm sweating", "urgent", None),
    ("I cant breath properly", "urgent", None),
    ("my mum passed out and her sugar was low", "urgent", None),
    ("I keep throwing up and can't keep anything down", "urgent", None),
    ("there's pus coming out of the sore on my foot", "urgent", None),
    # five abbreviations
    ("What should I do about a hypo?", "answered", "15 g"),
    ("When do we stop the biguanide?", "answered", "below 30"),
    ("How often is HbA1c checked when stable?", "answered", "every 6 months"),
    ("How often should the ACR be checked?", "answered", "at least once a year"),
    ("Which monofilament for the foot exam?", "answered", "10 g"),
    # ten more ordinary and out-of-scope questions
    ("What counts as a low glucose?", "answered", "below 70"),
    ("When should an ulcer be referred urgently?", "answered", "spreading redness"),
    ("Can metformin be started with an eGFR of 40?", "answered", "below 45"),
    ("How often should eGFR be checked?", "answered", "at least once a year"),
    ("When is retinal screening arranged?", "answered", "at diagnosis"),
    ("How long should I wait before rechecking after treating a low?", "answered", "15 minutes"),
    ("What is a common HbA1c target?", "answered", "below 7%"),
    ("Which insulin is best?", "declined", None),
    ("Can I drink alcohol with diabetes?", "declined", None),
    ("What is the blood pressure target?", "declined", None),
]
print("Before fixes:")
run_suite(MORE_CHECKS)

# Fixes for the failures, then re-run EVERY check, old and new
URGENT.extend([r"chest (feels )?(really )?(tight|heavy)|tight chest", r"can.?t breath|cant breath|short of breath",
               r"throw(ing)? up|keep vomiting", r"\bpus\b"])
ABBREVIATIONS.update({"acr": "ACR (urine albumin-to-creatinine ratio)", "a1c": "A1c (HbA1c)"})
print("\nAfter fixes:")
run_suite(MORE_CHECKS)
print("Original suite:")
run_suite(SUITE_V2)

# The remaining failures are retrieval problems: test the decline threshold on ALL checks together
_base_answer = answer
MIN_SCORE = 0.35
def answer(question, min_score=None, max_sentences=2):
    return _base_answer(question, min_score=MIN_SCORE if min_score is None else min_score, max_sentences=max_sentences)

ALL_CHECKS = SUITE_V2 + MORE_CHECKS
print("\nthreshold   checks passed")
for MIN_SCORE in [0.35, 0.40, 0.45, 0.50]:
    print(f"{MIN_SCORE:>9.2f}   ", end="")
    run_suite(ALL_CHECKS, verbose=False)
MIN_SCORE = 0.45
print(f"\nWith min_score {MIN_SCORE}:")
run_suite(ALL_CHECKS)

# %% Note
# Read the failures before the fixes: they are typically patient wording the urgent patterns do not cover ("throwing
# up", "chest feels tight", "cant breath"), abbreviations missing from the list, and questions answered from a near-miss
# section or declined. Pattern and abbreviation fixes are re-tested against the whole suite. The failures left after
# them ("Which insulin is best?" answered from a nearby section, and the semaglutide regression) are retrieval
# problems, so the decline threshold is tested on all checks at once. No threshold passes all 29: from 0.35 to 0.45 two
# checks always fail, they just change. At 0.45 the near-miss questions are correctly declined, but two genuine
# questions ("When do we stop the biguanide?", "How long should I wait before rechecking after treating a low?") are
# now declined too. Declining a covered question is the safer failure (the user is told to ask the care team), so 0.45
# is the better of the choices here, but the honest conclusion is that this version is not ready for release: the fix
# is better retrieval (for example handling more paraphrases), re-evaluated on the full test set from the evaluation
# lesson, not a threshold tuned on this suite.

# %% Step 4
# %% Note
# Optional, with an approved model. A generation step that rewrites the quoted section into a short answer sits after
# `safe_answer`, never before the urgent check. Check every generated sentence against the cited section with the
# negation-aware citation check from the RAG track, and fall back to the quoted text if any sentence fails:
#
# ```python
# def ask(prompt: str) -> str:
#     """Call your approved model (API key from an environment variable) and return its text."""
#     raise NotImplementedError
#
# def generated_answer(question):
#     r = safe_answer(question)
#     if r["route"] != "answered":
#         return r
#     draft = ask(f"Rewrite this protocol text as a short answer to the question. Keep every number and condition. "
#                 f"Text: {r['answer']}\nQuestion: {question}")
#     if all(sentence_is_supported(s, r["answer"]) for s in re.split(r"(?<=[.!?])\s+", draft)):  # your citation check
#         r["answer"] = draft
#     return r
# ```
#
# Then run every release check on the generated answers too. Expect the facts checks ("below 30", "15 g") to catch
# rewrites that round or soften numbers.

# %% Step 5
# %% Note
# **Governance summary: synthetic diabetes protocol assistant (example)**
#
# | Item | Summary |
# |---|---|
# | Intended use | Clinic staff ask questions about the clinic's own approved diabetes protocol; the assistant quotes and cites the relevant section, routes urgent symptoms to an emergency message, and declines questions the protocol does not cover. Not for patients; no advice about individual patients. |
# | Owner | Diabetes clinical lead (content); digital services (system); named clinical safety officer. |
# | Data protection | No patient data in questions by policy and by an identifier screen; questions and answers logged without identifiers; logs retained for 12 months for audit. |
# | Regulatory assessment | Assessed as not a medical device in this intended use (information retrieval from the clinic's own protocol for staff), to be re-assessed by regulatory advice if the use extends to patients or to patient-specific recommendations. |
# | Evaluation evidence | Release suite (all checks passing, including urgent and version checks); end-to-end evaluation on at least 200 questions with retrieval, correctness, faithfulness and decline rates and confidence intervals; red-team results; a four-week pilot. |
# | Monitoring | Monthly review of a random sample of logged answers and all reported problems; re-run the full suite on every protocol update, model update or threshold change; superseded versions removed from the index. |
# | Switch-off | The clinical safety officer can withdraw the tool at once; staff fall back to the protocol document. |

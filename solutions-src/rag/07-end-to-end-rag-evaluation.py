# %% Step 1
evaluate(max_sentences=2)
evaluate(max_sentences=None)
evaluate(max_sentences=None, min_score=0.45)
scores = sorted((answer(q, min_score=0.0)["retrieval_score"], gold is not None, q) for q, gold, _ in TESTS)
print(f"Lowest answerable {min(s for s, a, _ in scores if a):.2f}, highest out of scope {max(s for s, a, _ in scores if not a):.2f}")

# %% Step 2
BORDERLINE = [   # (question, section that answers it or None)
    ("Does the protocol say anything about HbA1c in pregnancy?", "s1"),      # only "non-pregnant adults" mentioned
    ("Is metformin OK for someone on dialysis?", "s2"),                      # implied by eGFR below 30
    ("my foot is cold and white what do i do", "s3"),                        # unusual wording
    ("Glucose 65, what now?", "s4"),                                         # a number instead of words
    ("What causes low blood sugar?", None),                                  # topic present, question not answered
    ("How often should my blood pressure be checked?", None),
    ("Do I need my eyes checked every year?", "s5"),                         # answer: the local programme decides
    ("What is the albumin test for?", None),                                 # test named, purpose not explained
    ("Can I stop metformin myself if I feel unwell?", None),                 # near the metformin section, not covered
    ("What should I eat to avoid a hypo?", None),
]
rows = [(answer(q, min_score=0.0)["retrieval_score"], gold, q) for q, gold in BORDERLINE]
for s, gold, q in sorted(rows, key=lambda r: r[0]):
    print(f"{s:.2f}  {'answerable  ' if gold else 'not covered '}  {q}")

all_rows = [(s, a) for s, a, _ in scores] + [(s, g is not None) for s, g, _ in rows]
print("\nthreshold  answerable answered  not-covered declined")
for t in [0.30, 0.35, 0.40, 0.45, 0.50]:
    ans = [s >= t for s, a in all_rows if a]
    dec = [s < t for s, a in all_rows if not a]
    print(f"{t:>9.2f}  {sum(ans):>9}/{len(ans)}  {sum(dec):>12}/{len(dec)}")

# %% Note
# The clean margin between answerable and out-of-scope questions disappears once the questions are realistic:
# "not covered" questions that sit next to a covered topic (stopping metformin when unwell, what causes a low)
# score as high as genuine questions phrased unusually. No single threshold separates them; every choice trades
# wrongly answered questions against wrongly declined ones, and the trade-off must be decided by people who know
# which error is worse in this clinic. Retrieval score alone is not enough to decide whether a question is answered.

# %% Step 3
def faithful(result):
    """Every sentence of an extractive answer must appear word for word in the cited section."""
    if not result["citations"]:
        return True
    cid = result["citations"][0].split(":")[0]
    section_text = " ".join(next(c for c in chunks if c["id"] == cid)["sentences"])
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", result["answer"]) if s]
    return all(s in section_text for s in sentences)

questions_all = [q for q, _, _ in TESTS] + [q for q, _ in BORDERLINE]
for setting in [dict(max_sentences=2), dict(max_sentences=None)]:
    passed = sum(faithful(answer(q, **setting)) for q in questions_all)
    print(f"{str(setting):<26} faithful {passed}/{len(questions_all)}")

tampered = answer("When should metformin be stopped?")
tampered["answer"] = tampered["answer"].replace("below 30", "below 20")
print("A tampered answer passes?", faithful(tampered))

# %% Note
# Extractive answers are faithful by construction: they only quote the section. The check still earns its place
# because it will catch the moment someone adds a generation step, a summariser or a formatting function that
# changes the text, as the tampered example shows.

# %% Step 4
# %% Note
# Optional, with an approved model. Add a generation step (see the assistant lesson), then re-run `evaluate()`
# for correctness and a claim-level faithfulness check on the generated text (word overlap plus numbers, or an
# NLI model, as in the citations lesson), and read the failures. Errors that typically appear once text is
# generated rather than quoted: conditions softened or dropped ("stop metformin if kidney function is poor"),
# numbers rounded or changed, general advice added from the model's training data, and refusals that turn into
# partial answers. Measure both settings on the same questions so the cost of fluency is visible.

# %% Step 5
# %% Note
# Example evaluation plan: a protocol Q&A assistant for clinic staff.
#
# **Intended use.** Staff ask questions about the clinic's own diabetes protocol; the assistant quotes and cites
# the relevant section, or declines. It does not give advice about individual patients and is not used by patients.
#
# **Before deployment**
# 1. *Test set*: at least 200 questions, collected from real staff questions (with no patient identifiers) and
#    written by clinicians, including about 20% that the protocol does not cover and a set of borderline and
#    adversarial questions (abbreviations, typos, negations, embedded instructions).
# 2. *Labels*: for each question, the section(s) that answer it and the key facts a correct answer must contain,
#    agreed by two clinicians.
# 3. *Metrics, with confidence intervals*: retrieval recall@1 and @3; answer correctness (key facts present);
#    faithfulness (every sentence supported by the cited section); appropriate declines on uncovered questions;
#    wrong answers graded by severity.
# 4. *Acceptance criteria*, set in advance by the clinical safety lead: for example no major-severity wrong
#    answers in the test set, correctness ≥ 95%, and declines ≥ 90% on uncovered questions.
#
# **Pilot**: four weeks with a small group, with every answer logged and a sample reviewed weekly; a visible
# "report a problem" button.
#
# **In use**: monthly review of a random sample and of all reported problems; re-run the full test set whenever
# the protocol, the embedding model, the threshold or any generation model changes; protocol updates trigger
# re-indexing, and superseded versions are removed.
#
# **Governance**: an owner for the document library, a named clinical safety officer, and a written decision on
# whether the tool falls under medical-device regulation in its intended use.

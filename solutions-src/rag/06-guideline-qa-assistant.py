# %% Step 1
for q in ["When should metformin be stopped?", "What should I do about a hypo?", "What dose of semaglutide should I start?"]:
    r = answer(q)
    print(f"Q: {q}\nA: {r['answer']}\n   {r['citations']}  score {r['retrieval_score']}\n")

# %% Step 2
my_tests = [   # (question, facts a complete answer must contain; [] = should decline)
    ("How often should HbA1c be checked once treatment is stable?", ["every 6 months"]),
    ("What is a common HbA1c target?", ["below 7%"]),
    ("Can metformin be started with an eGFR of 40?", ["below 45"]),
    ("How often should the ACR be checked?", ["at least once a year"]),            # abbreviation
    ("What should I do about a hypo?", ["15 g", "15 minutes"]),                     # abbreviation
    ("What weight of monofilament is used?", ["10 g"]),
    ("When should a foot be referred urgently?", ["spreading redness", "cold, pale foot"]),
    ("When does eye screening start?", ["at diagnosis"]),
    ("Which blood pressure target should I aim for?", []),                         # out of scope
    ("Can I take ibuprofen with metformin?", []),                                   # out of scope
]
for q, facts in my_tests:
    r = answer(q)
    print(f"Q: {q}\nA: {r['answer']}\n   {r['citations']}  score {r['retrieval_score']}\n")

# %% Note
# Read every answer as a clinician would. Both out-of-scope questions are declined. Three answers are wrong in
# different ways:
#
# - **"How often should the ACR be checked?"** retrieves the HbA1c section with a score (0.42) above the
#   threshold, so the assistant confidently quotes the wrong rule. "ACR" is not in the abbreviation list.
# - **"What weight of monofilament is used?"** is declined (0.28) although the foot-care section answers it: a
#   false refusal.
# - **"What is a common HbA1c target?"** finds the right section but quotes the two monitoring sentences, not
#   the target.

# %% Step 3
def complete(r, facts):
    return all(f in r["answer"] for f in facts) if facts else not r["citations"]

print(f"{'question':<62}{'2 sentences':>13}{'whole section':>15}")
for q, facts in my_tests:
    two, whole = answer(q, max_sentences=2), answer(q, max_sentences=None)
    print(f"{q:<62}{('ok' if complete(two, facts) else 'INCOMPLETE'):>13}{('ok' if complete(whole, facts) else 'INCOMPLETE'):>15}")

# %% Note
# Quoting the whole section fixes the two answers that lost information (the HbA1c target, and the first steps of
# hypo treatment), because picking the two most similar sentences independently drops sentences that matter.
# It cannot fix the ACR and monofilament questions: those are retrieval failures, so the right text was never
# reached. For long sections, quote in document order around the best sentence rather than picking sentences
# independently; for retrieval failures, extend the abbreviation list and review the threshold.

# %% Step 4
def answer_multi(question, min_score=0.35, margin=0.05):
    """Like answer(), but also quotes the second section when its score is within `margin` of the best."""
    q_emb = model.encode([expand(question)], normalize_embeddings=True)[0]
    scores = chunk_emb @ q_emb
    order = np.argsort(-scores)
    if scores[order[0]] < min_score:
        return {"answer": "The protocol does not cover this. Please ask your care team.", "citations": []}
    chosen = [order[0]] + ([order[1]] if scores[order[0]] - scores[order[1]] < margin else [])
    parts = [f"{' '.join(chunks[i]['sentences'])} [{chunks[i]['id']}]" for i in chosen]
    return {"answer": " ".join(parts), "citations": [f"{chunks[i]['id']}: {chunks[i]['heading']}" for i in chosen],
            "scores": [round(float(scores[i]), 2) for i in chosen]}

for q in ["Does metformin affect my HbA1c checks?", "Are low sugars and kidney problems linked to metformin?",
          "What should I do about a hypo?"]:
    r = answer_multi(q)
    print(f"Q: {q}\nA: {r['answer']}\n   {r['citations']} {r.get('scores', '')}\n")

# %% Note
# When two sections score almost the same, quoting both (each with its own citation) is safer than silently
# choosing one. The margin is a setting to choose on test questions: too large and every answer carries an
# irrelevant second section, too small and genuinely two-part questions get half an answer.

# %% Step 5
# %% Note
# Optional, with an approved model. Generation turns the quoted text into a fluent answer, and that is the step
# where unsupported content can appear, so check the generated answer claim by claim against the cited section,
# for example with the `support()` checker from the citations lesson.
#
# ```python
# def ask(prompt: str) -> str:
#     """Call your approved model (API key from an environment variable) and return its text."""
#     raise NotImplementedError
#
# def generate(question):
#     r = answer(question, max_sentences=None)
#     if not r["citations"]:
#         return r["answer"], r
#     prompt = ("Rewrite the protocol text below as a short answer to the question. Use only this text, "
#               "keep every number and condition exactly, and end with the citation.\n\n"
#               f"Protocol text [{r['citations'][0]}]: {r['answer']}\n\nQuestion: {question}")
#     return ask(prompt), r
#
# for q, _ in my_tests:
#     text, r = generate(q)
#     section_text = " ".join(chunks[int(r["citations"][0][1]) - 1]["sentences"]) if r["citations"] else ""
#     for sentence in re.split(r"(?<=[.!?])\s+", text):
#         ...   # flag sentences whose content words or numbers are not in section_text
# ```
#
# Expect fluent answers that occasionally soften a condition ("if eGFR is low" instead of "below 30") or add
# general advice not in the protocol; these are exactly the errors the check is for.

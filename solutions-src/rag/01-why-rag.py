# %% Step 1
def retrieve(question, docs):
    vec = TfidfVectorizer(stop_words="english").fit(list(docs.values()) + [question])
    s = cosine_similarity(vec.transform([question]), vec.transform(docs.values()))[0]
    return sorted(zip(docs, s.round(2).tolist()), key=lambda t: -t[1])

def grounded_prompt(question, doc_id, docs):
    return (f"Answer using only the source below. If the source does not answer, say so.\n"
            f"Cite the source id in square brackets.\n\nSource [{doc_id}]: {docs[doc_id]}\n\nQuestion: {question}")

ranking = retrieve(question, documents)
print(ranking)
print(grounded_prompt(question, ranking[0][0], documents))

# %% Step 2
my_questions = [
    "How often should HbA1c be checked when diabetes is stable?",
    "When should the monofilament test be done?",
    "When is retinal photography arranged?",
    "Should both feet be examined at each visit?",
    "What is the starting dose of insulin glargine?",          # not covered by any document
]
for q in my_questions:
    (top, score), (second, score2) = retrieve(q, documents)[:2]
    print(f"{q}\n   top: {top} ({score:.2f})   next: {second} ({score2:.2f})")

# %% Note
# The stable-HbA1c question retrieves the *2024* protocol first, because it matches its wording ("every 6 months
# when stable") more closely, even though the 2026 update contains the same rule plus the newer one. The
# insulin question has nothing to find, but retrieval still returns a "top" document. A retriever always
# returns something: a minimum score, and an instruction to say "the source does not answer", are needed.

# %% Step 3
metadata = {"protocol-2024": {"superseded": True}, "protocol-2026": {"superseded": False},
            "foot-care": {"superseded": False}, "eye-screening": {"superseded": False}}
current = {k: v for k, v in documents.items() if not metadata[k]["superseded"]}

for q in [question, my_questions[0]]:
    print(f"{q}\n   all documents:     {retrieve(q, documents)[:2]}"
          f"\n   current documents: {retrieve(q, current)[:2]}")

# %% Note
# Filtering on metadata happens before ranking, so an outdated document can never be retrieved, however well it
# matches the wording. Keeping a document library current (version, status, review date) is as important to a
# RAG system as the retrieval method.

# %% Step 4
# %% Note
# Optional, with a model your organisation has approved, and only with the synthetic documents here.
#
# ```python
# def ask(prompt: str) -> str:
#     """Call your approved model and return its answer (API key from an environment variable)."""
#     raise NotImplementedError
#
# for q in my_questions:
#     top = retrieve(q, current)[0][0]
#     without_source = ask(q)
#     with_source = ask(grounded_prompt(q, top, current))
#     print(f"Q: {q}\n  without source: {without_source}\n  with source:    {with_source}\n")
# ```
#
# What to look for: without a source, the model answers from its training data, which may not match *this*
# clinic's protocol (for example a different HbA1c interval) and cannot be cited. With the source, answers
# should follow the protocol and cite it; check that the unanswerable question gets "the source does not answer"
# rather than an invented dose.

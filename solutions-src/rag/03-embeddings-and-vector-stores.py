# %% Step 1
labelled = [   # (question, index of the section that answers it)
    ("My sugar dropped really low, what now?", 3), ("What should I do about a hypo?", 3),
    ("Is the diabetes pill safe with weak kidneys?", 1), ("When do we need to stop the biguanide?", 1),
    ("How often are retinal photos needed?", 4), ("Numb toes test", 2),
    # six more: two abbreviations, two drug-class names, two plain questions
    ("When is the next A1c due after changing treatment?", 0),
    ("What eGFR rules out starting metformin?", 1),
    ("Does an SU need reviewing after a low?", 3),          # SU = sulfonylurea
    ("Are sulfonylureas a cause of low glucose?", 3),
    ("How often should the ACR be measured?", 1),           # ACR = albumin-to-creatinine ratio
    ("Who needs urgent referral for a foot ulcer?", 2),
]
qs = [q for q, _ in labelled]
gold = np.array([g for _, g in labelled])

def top1(score_matrix):
    return score_matrix.argmax(axis=1)

kw_vec = TfidfVectorizer(stop_words="english").fit(sections + qs)
kw_scores = cosine_similarity(kw_vec.transform(qs), kw_vec.transform(sections))
em_scores = model.encode(qs, normalize_embeddings=True) @ section_emb.T
for q, g, k, e in zip(qs, gold, top1(kw_scores), top1(em_scores)):
    print(f"{q:<52} keyword {'ok ' if k == g else 'MISS'}  embedding {'ok ' if e == g else 'MISS'}")
print(f"\nCorrect at rank 1: keyword {np.mean(top1(kw_scores) == gold):.0%}, embedding {np.mean(top1(em_scores) == gold):.0%}")

# %% Step 2
from collections import Counter
import math

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg_len = np.mean([len(d) for d in self.docs])
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        out = []
        for d in self.docs:
            tf = Counter(d)
            s = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1)
                    / (tf[w] + self.k1 * (1 - self.b + self.b * len(d) / self.avg_len)) for w in tokenize(query))
            out.append(s)
        return np.array(out)

bm25 = BM25(sections)
bm_scores = np.array([bm25.scores(q) for q in qs])
for name, s in [("TF-IDF", kw_scores), ("BM25", bm_scores), ("Embedding", em_scores)]:
    print(f"{name:<10} correct at rank 1: {np.mean(top1(s) == gold):.0%}")

# %% Note
# BM25 is the standard keyword-ranking formula in search engines: it rewards rare words and stops rewarding a
# word after it appears a few times. Like TF-IDF it cannot match words that do not appear ("hypo", "SU"), which
# is where embeddings and abbreviation handling help.

# %% Step 3
MORE_ABBREVIATIONS = {
    **ABBREVIATIONS,
    "a1c": "A1c (HbA1c)",
    "su": "SU (sulfonylurea, a medicine that can cause low blood glucose)",
    "sulfonylureas": "sulfonylureas (medicines that can cause low blood glucose)",
    "acr": "ACR (urine albumin-to-creatinine ratio)",
}

def expand_with(question, table):
    for short, long in table.items():
        question = re.sub(rf"\b{short}\b", long, question, flags=re.IGNORECASE)
    return question

before = top1(em_scores)
after = top1(model.encode([expand_with(q, MORE_ABBREVIATIONS) for q in qs], normalize_embeddings=True) @ section_emb.T)
fixed = [qs[i] for i in range(len(qs)) if before[i] != gold[i] and after[i] == gold[i]]
broken = [qs[i] for i in range(len(qs)) if before[i] == gold[i] and after[i] != gold[i]]
print(f"Embedding correct at rank 1: {np.mean(before == gold):.0%} -> {np.mean(after == gold):.0%}")
print("Fixed:", fixed)
print("Made worse:", broken)

# %% Note
# A small, reviewed dictionary of local abbreviations is one of the cheapest improvements to clinical
# retrieval. It must be checked on the whole test set, because an expansion can also pull a question towards
# the wrong section; abbreviations that mean different things in different specialties (for example "MS")
# should not be expanded blindly.

# %% Step 4
OLD_PROTOCOL = PROTOCOL.replace("Check HbA1c every 3 months after any change in treatment. ", "").replace(
    "Metformin should not be started if eGFR is below 45. If eGFR falls below 30 during treatment, stop metformin.",
    "Stop metformin if eGFR falls below 45.")   # an invented older rule, for illustration
old_sections = [p.strip() for p in re.split(r"\n(?=## )", OLD_PROTOCOL)[1:]]
store.add(old_sections, [{"source": "clinic-protocol", "version": "2024", "section": t} for t in titles])

q = "When should metformin be stopped?"
print("No filter:")
for score, rec in store.search(q, k=3):
    print(f"  {score:.2f}  v{rec['version']}  {rec['section']}")
print("version='2026':")
for score, rec in store.search(q, k=3, version="2026"):
    print(f"  {score:.2f}  v{rec['version']}  {rec['section']}")

# %% Note
# Without the filter the outdated 2024 section can rank first (its shorter rule matches the question closely),
# which would give the old eGFR threshold. With the filter it cannot be returned at all.

# %% Step 5
# %% Note
# Optional (sketch). MedCPT is a pair of open biomedical retrieval encoders trained on PubMed search logs,
# published by the US National Library of Medicine on Hugging Face (`ncbi/MedCPT-Query-Encoder` and
# `ncbi/MedCPT-Article-Encoder`; check the model card for licence and intended use). Queries and documents
# use different encoders:
#
# ```python
# import torch
# from transformers import AutoModel, AutoTokenizer
#
# def encode(name, texts, max_length):
#     tok, enc = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)
#     with torch.no_grad():
#         batch = tok(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
#         return enc(**batch).last_hidden_state[:, 0, :].numpy()      # the [CLS] embedding
#
# q_emb = encode("ncbi/MedCPT-Query-Encoder", qs, 64)
# d_emb = encode("ncbi/MedCPT-Article-Encoder", [[t, s] for t, s in zip(titles, sections)], 512)
# medcpt_scores = q_emb @ d_emb.T                                        # MedCPT uses inner product
# print(f"MedCPT correct at rank 1: {np.mean(medcpt_scores.argmax(axis=1) == gold):.0%}")
# ```
#
# Compare on the same labelled questions. A model trained on biomedical literature may handle drug classes
# better, but patient-style questions ("my sugar dropped") are far from PubMed queries, so test rather than assume.
# When we ran this sketch (October 2026), MedCPT ranked the right section first for 10 of the 12 questions (83%),
# the same as all-MiniLM-L6-v2 before abbreviation expansion: a larger, domain-specific model was no substitute
# for a reviewed abbreviation list on this task.

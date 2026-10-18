# %% Step 1
report("Keyword", keyword)
report("Embedding", embedding)
report("Hybrid", reciprocal_rank_fusion(keyword, embedding))

# %% Step 2
more_tests = [
    ("When does HbA1c move to every 6 months?", "glucose"),
    ("Is 7% the HbA1c goal for everyone?", "glucose"),
    ("What eGFR stops metformin being started?", "kidney"),
    ("Do I keep taking metformin if my kidney function falls to 25?", "kidney"),
    ("Which monofilament weight is used?", "foot"),
    ("What does a cold pale foot mean for referral?", "foot"),
    ("What glucose level counts as low?", "hypo"),
    ("How long should I wait before rechecking after juice for a low?", "hypo"),
    ("Who sets the eye screening interval?", "eye"),
    ("What insulin dose should a new patient start on?", None),   # not in the protocol
    ("Can I drive after a hypo?", None),
    ("Which vaccines do people with diabetes need?", None),
]
all_tests = tests + more_tests
all_q = [q for q, _ in all_tests]
answerable = np.array([r is not None for _, r in all_tests])
rel = np.array([ids.index(r) if r else -1 for _, r in all_tests])

vec2 = TfidfVectorizer(stop_words="english").fit(sections + all_q)
section_emb = model.encode(sections, normalize_embeddings=True)
methods = {
    "Keyword": cosine_similarity(vec2.transform(all_q), vec2.transform(sections)),
    "Embedding": model.encode(all_q, normalize_embeddings=True) @ section_emb.T,
}
methods["Hybrid"] = reciprocal_rank_fusion(methods["Keyword"], methods["Embedding"])

def rank_of_relevant(scores, rel_idx):
    order = np.argsort(-scores, axis=1)
    return np.array([list(order[i]).index(rel_idx[i]) + 1 for i in range(len(rel_idx))])

THRESHOLD = 0.35   # embedding similarity below which the assistant should say "not covered"
for name, s in methods.items():
    r = rank_of_relevant(s[answerable], rel[answerable])
    line = f"{name:<10} answerable ({answerable.sum()}): recall@1 {np.mean(r <= 1):.2f}  recall@3 {np.mean(r <= 3):.2f}  MRR {np.mean(1 / r):.2f}"
    if name == "Embedding":
        declined = (s[~answerable].max(axis=1) < THRESHOLD).mean()
        line += f"   no-answer questions declined at {THRESHOLD}: {declined:.0%}"
    print(line)

# %% Note
# Questions the documents cannot answer have no correct section, so they cannot be scored with recall or MRR.
# Score them separately as a decision: did the system decline (top similarity below a threshold)? A test set
# without such questions rewards a retriever for always returning something, which is exactly the behaviour
# that leads to confident wrong answers. Here the embedding threshold of 0.35 declines two of the three
# out-of-scope questions; the third scores above it. Hybrid scores are rank-based and have no natural threshold, so the
# decline decision here uses the embedding scores.

# %% Step 3
ABBREVIATIONS = {"hypo": "hypo (hypoglycaemia, low blood glucose)", "biguanide": "biguanide (metformin)"}

def expand(q):
    for short, long in ABBREVIATIONS.items():
        q = re.sub(rf"\b{short}\b", long, q, flags=re.IGNORECASE)
    return q

expanded_q = [expand(q) for q in all_q]
vec3 = TfidfVectorizer(stop_words="english").fit(sections + expanded_q)
expanded = {
    "Keyword": cosine_similarity(vec3.transform(expanded_q), vec3.transform(sections)),
    "Embedding": model.encode(expanded_q, normalize_embeddings=True) @ section_emb.T,
}
expanded["Hybrid"] = reciprocal_rank_fusion(expanded["Keyword"], expanded["Embedding"])
for name in methods:
    before = rank_of_relevant(methods[name][answerable], rel[answerable])
    after = rank_of_relevant(expanded[name][answerable], rel[answerable])
    print(f"{name:<10} recall@1 {np.mean(before <= 1):.2f} -> {np.mean(after <= 1):.2f}   MRR {np.mean(1 / before):.2f} -> {np.mean(1 / after):.2f}")

# %% Step 4
rng = np.random.default_rng(0)
hits = {name: rank_of_relevant(expanded[name][answerable], rel[answerable]) <= 1 for name in expanded}
n = answerable.sum()
boot = {name: [] for name in hits}
diff = []
for _ in range(1000):
    idx = rng.integers(0, n, n)                       # resample questions with replacement
    for name in hits:
        boot[name].append(hits[name][idx].mean())
    diff.append(hits["Embedding"][idx].mean() - hits["Keyword"][idx].mean())
for name, values in boot.items():
    lo, hi = np.percentile(values, [2.5, 97.5])
    print(f"{name:<10} recall@1 {hits[name].mean():.2f} (95% bootstrap CI {lo:.2f} to {hi:.2f})")
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"Embedding minus keyword: {np.mean(diff):+.2f} (95% CI {lo:+.2f} to {hi:+.2f})")

def wilson(k, n, z=1.96):
    p = k / n
    centre, half = (p + z**2 / (2 * n)) / (1 + z**2 / n), z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half

for name in hits:
    lo, hi = wilson(hits[name].sum(), n)
    print(f"{name:<10} {hits[name].sum()}/{n} correct at rank 1: Wilson 95% CI {lo:.2f} to {hi:.2f}")

# %% Note
# After abbreviation expansion the embedding and hybrid methods get all 21 questions right, and the bootstrap
# then gives a zero-width interval (1.00 to 1.00): every resample is also perfect. That is a known weakness of the
# bootstrap with small samples and extreme results, not evidence of perfection. The Wilson interval shows the
# real uncertainty: 21 out of 21 is compatible with a true recall@1 as low as about 0.85. The paired difference
# from keyword search has a lower limit of zero. Twenty questions cannot tell similar methods apart; a test set
# for choosing a retriever needs many more, drawn from real user questions.

# %% Step 5
# Graded labels: 2 = answers the question fully, 1 = partly relevant. Unlisted sections count as 0.
grades = {
    "How often should HbA1c be checked after changing treatment?": {"glucose": 2},
    "What HbA1c target suits most adults?": {"glucose": 2},
    "My sugar dropped really low, what now?": {"hypo": 2},
    "What should I do about a hypo?": {"hypo": 2},
    "How many grams of carbohydrate treat a low?": {"hypo": 2},
    "Is the diabetes pill safe with weak kidneys?": {"kidney": 2},
    "When do we need to stop the biguanide?": {"kidney": 2},
    "How often is urine albumin checked?": {"kidney": 2},
    "Numb toes test": {"foot": 2},
    "When is a foot ulcer an emergency?": {"foot": 2},
    "How often are retinal photos needed?": {"eye": 2},
    "When should eye screening start?": {"eye": 2},
    "When does HbA1c move to every 6 months?": {"glucose": 2},
    "Is 7% the HbA1c goal for everyone?": {"glucose": 2},
    "What eGFR stops metformin being started?": {"kidney": 2},
    "Do I keep taking metformin if my kidney function falls to 25?": {"kidney": 2, "glucose": 1},
    "Which monofilament weight is used?": {"foot": 2},
    "What does a cold pale foot mean for referral?": {"foot": 2},
    "What glucose level counts as low?": {"hypo": 2, "glucose": 1},
    "How long should I wait before rechecking after juice for a low?": {"hypo": 2},
    "Who sets the eye screening interval?": {"eye": 2},
}

def ndcg_at_3(scores, questions_):
    values = []
    for s, q in zip(scores, questions_):
        g = grades[q]
        top = np.argsort(-s)[:3]
        dcg = sum(g.get(ids[j], 0) / np.log2(pos + 2) for pos, j in enumerate(top))
        ideal = sorted(g.values(), reverse=True)[:3]
        idcg = sum(v / np.log2(pos + 2) for pos, v in enumerate(ideal))
        values.append(dcg / idcg)
    return np.mean(values)

ans_q = [q for q, a in zip(all_q, answerable) if a]
for name in expanded:
    print(f"{name:<10} nDCG@3 {ndcg_at_3(expanded[name][answerable], ans_q):.3f}   "
          f"MRR {np.mean(1 / rank_of_relevant(expanded[name][answerable], rel[answerable])):.3f}")

# %% Note
# nDCG gives credit for putting the *most* useful section first and some credit for partly relevant sections
# lower down; MRR only looks at the first fully relevant one. With almost every question having one relevant
# section, the two agree closely here. Graded labels matter more for real guidelines, where several sections
# often answer parts of a question.

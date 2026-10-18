# %% Step 1
def top_k_softmax(z, k=3, temperature=1.0):
    """Keep only the k highest-scoring tokens, then apply temperature and renormalise."""
    keep = np.argsort(z)[-k:]
    masked = np.full_like(z, -np.inf)
    masked[keep] = z[keep]
    return softmax(masked, temperature)

for t in [0.2, 1.0, 2.0]:
    plain, top3 = softmax(logits, t), top_k_softmax(logits, 3, t)
    print(f"temperature {t}:")
    print("  all tokens: " + ", ".join(f"{w} {p:.2f}" for w, p in zip(candidates, plain)))
    print("  top-3:      " + ", ".join(f"{w} {p:.2f}" for w, p in zip(candidates, top3)))

# %% Note
# Top-k sampling removes the long tail: however high the temperature, "disease" and "hypertension" can no
# longer be chosen. It limits how strange an output can get, but it does not make the output true; the
# remaining tokens can still be wrong in context.

# %% Step 2
extra = """the patient was started on empagliflozin .
the patient was referred to ophthalmology .
the patient was reviewed in clinic .
the trial was registered in 2018 .
the trial was stopped early .
the nurse was trained in 2020 .
the nurse was referred to occupational health .
the dose was reduced to 500 mg .
the dose was increased in clinic .
the clinic was started in 2015 .""".split("\n")
full_corpus = corpus + extra

def build_model(sentences, context=1):
    model = defaultdict(Counter)
    for sentence in sentences:
        words = ["<s>"] * context + sentence.split()
        for i in range(context, len(words)):
            model[tuple(words[i - context:i])][words[i]] += 1
    return model

def sample(model, context=1, seed=0, n=500, max_words=15):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        words = ["<s>"] * context
        while words[-1] != "." and len(words) < max_words + context:
            options = model[tuple(words[-context:])]
            probs = np.array(list(options.values())) / sum(options.values())
            words.append(r.choice(list(options), p=probs))
        out.append(" ".join(words[context:]))
    return out

known = set(full_corpus)
bigram = build_model(full_corpus, context=1)
generated = sample(bigram, context=1)
true_share = np.mean([s in known for s in generated])
print(f"Bigram model: {true_share:.0%} of {len(generated)} generated sentences appear in the corpus")
print("Examples of new (false) sentences:")
for s in [s for s in dict.fromkeys(generated) if s not in known][:5]:
    print("  ", s)

# %% Note
# "Appears in the corpus" is our stand-in for "true": every sentence in the corpus is a fact about this toy
# world, and a new combination ("the nurse was started on empagliflozin") is a fluent statement that nobody made.
# Each word follows plausibly from the one before it, which is all the model checks.

# %% Step 3
trigram = build_model(full_corpus, context=2)
generated3 = sample(trigram, context=2)
print(f"Bigram:  {1 - np.mean([s in known for s in generated]):.0%} of generated sentences are not in the corpus")
print(f"Trigram: {1 - np.mean([s in known for s in generated3]):.0%} of generated sentences are not in the corpus")

# %% Note
# With two words of context the model can tell "patient was" from "trial was", so far fewer false
# combinations appear. On a corpus this small, a trigram model mostly copies its training sentences. Real
# LLMs use thousands of tokens of context and are trained on vast text, so they rarely produce word salad,
# but the same principle holds: they generate what is *likely* given the context, and a likely sentence can
# still be false, especially for rare facts the training data covered thinly.

# %% Step 4
# %% Note
# Optional, and only with a model your organisation has approved, using a question with no patient data.
# The experiment needs one function that sends a prompt and returns text; every provider's SDK offers an
# equivalent, so the harness below leaves `ask` for you to fill in (API keys belong in environment variables).
#
# ```python
# import os
# from collections import Counter
#
# def ask(prompt: str, temperature: float) -> str:
#     """Call your approved model here and return its text answer."""
#     raise NotImplementedError
#
# QUESTION = "In one sentence: what is the usual first-line glucose-lowering medicine for type 2 diabetes in adults?"
# for temperature in [0.0, 1.0]:
#     answers = [ask(QUESTION, temperature) for _ in range(5)]
#     print(f"temperature {temperature}: {len(set(answers))} different answers")
#     for answer, n in Counter(answers).most_common():
#         print(f"  {n} x {answer}")
# ```
#
# What to record: how many distinct answers you get at each temperature, whether the *facts* change or only
# the wording, and whether any answer adds a claim that is wrong. Even at temperature 0, some services do not
# return identical text every time, so a single run is never enough to judge a model.

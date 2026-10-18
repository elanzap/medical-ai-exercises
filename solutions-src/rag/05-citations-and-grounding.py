# %% Step 1
def verdict(claim, cited):
    overlap, numbers_ok = support(claim, sections[cited])
    return "supported" if overlap >= 0.8 and numbers_ok else "CHECK"

for claim, cited in answer:
    print(f"{verdict(claim, cited):<9} [{cited}] {claim}")
print()
for claim, cited in answer:
    if verdict(claim, cited) == "CHECK":
        better = best_source(claim)
        print(f"{claim} -> best source {better}: {verdict(claim, better)}")

# %% Step 2
more_claims = [   # (claim, cited section, truly supported by that section?)
    ("Check HbA1c every 3 months after a change in treatment.", "s1", True),
    ("Check HbA1c every 6 months once treatment is stable.", "s1", True),
    ("Test sensation with a 10 g monofilament at least once a year.", "s3", True),
    ("Treat a low glucose with 15 g of fast-acting carbohydrate.", "s4", True),
    ("Recheck glucose after 15 minutes.", "s4", True),
    ("Arrange retinal screening at diagnosis of type 2 diabetes.", "s5", True),
    ("Refer urgently if a foot ulcer has spreading redness.", "s3", True),
    ("Do not stop metformin if eGFR falls below 30.", "s2", False),      # negated
    ("Never test sensation with a monofilament.", "s3", False),          # negated
    ("Check HbA1c every 6 months after any change in treatment.", "s1", False),  # numbers swapped between rules
]
errors = []
for claim, cited, truth in more_claims:
    v = verdict(claim, cited)
    ok = (v == "supported") == truth
    errors += [] if ok else [claim]
    print(f"{v:<9} {'right' if ok else 'WRONG':<5} {claim}")
print(f"\nChecker wrong on {len(errors)} of {len(more_claims)} claims")

# %% Note
# The word-overlap checker accepts claims that use the source's words in the wrong way: a negated rule ("do not
# stop metformin") and a number moved from one rule to another ("6 months after any change") both share almost
# all their words and numbers with the source. Overlap measures *topic*, not *meaning*.

# %% Step 3
NEGATIONS = {"not", "no", "stop", "never", "don't", "avoid"}

def source_sentences(section_text):
    return [s for s in re.split(r"(?<=[.!?])\s+", section_text.replace("\n", " ")) if s.strip()]

def negation_mismatch(claim, cited):
    """Compare negation words between the claim and its best-matching sentence in the cited section."""
    claim_words = set(re.findall(r"[a-z']+", claim.lower()))
    best = max(source_sentences(sections[cited]),
               key=lambda s: len(content_words(claim) & content_words(s)))
    src_words = set(re.findall(r"[a-z']+", best.lower()))
    return (claim_words & NEGATIONS) != (src_words & NEGATIONS), best

def verdict_with_negation(claim, cited):
    v = verdict(claim, cited)
    mismatch, _ = negation_mismatch(claim, cited)
    return "CHECK" if mismatch else v

for claims in [[(c, s, None) for c, s in answer], more_claims]:
    for claim, cited, truth in claims:
        v1, v2 = verdict(claim, cited), verdict_with_negation(claim, cited)
        flag = "" if truth is None else ("right" if (v2 == "supported") == truth else "WRONG")
        print(f"{v1:<9} -> {v2:<9} {flag:<5} {claim}")

# %% Note
# The negation rule now catches "Do not stop metformin..." and "Never test sensation...", and also the lesson's
# reversed claim "Start metformin if eGFR is below 45" (the source sentence says "should not be started"). It
# still misses the swapped interval, which needs a check that each number appears in the *same* sentence and
# role as in the source. Every rule fixes one failure pattern; none understands the claim.

# %% Step 4
# An open natural language inference (NLI) model labels each (source sentence, claim) pair as entailment,
# neutral or contradiction. This cross-encoder is small (about 140 MB) and runs on a CPU.
from sentence_transformers import CrossEncoder

nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")
LABELS = ["contradiction", "entailment", "neutral"]          # the order used by this model's outputs

def nli_verdict(claim, cited):
    pairs_ = [(s, claim) for s in source_sentences(sections[cited])]
    probs = nli.predict(pairs_, apply_softmax=True)
    best = probs[:, 1].argmax()                              # the sentence that most entails the claim
    if probs[best, 1] > 0.5:
        return "supported"
    return "CONTRADICTED" if probs[:, 0].max() > 0.5 else "CHECK"

correct = 0
for claim, cited, truth in more_claims + [(c, s, None) for c, s in answer]:
    v = nli_verdict(claim, cited)
    flag = "" if truth is None else ("right" if (v == "supported") == truth else "WRONG")
    correct += flag == "right"
    print(f"{v:<12} {flag:<5} {claim}")
print(f"\nNLI model right on {correct} of {len(more_claims)} labelled claims")

# %% Note
# The NLI model reads the meaning of the claim rather than its words. Here it caught both negations, the swapped
# interval and the lesson's reversed "Start metformin" claim, which the word-overlap checker accepted. It also
# made a mistake of its own: it marked the correct claim "Treat a low glucose with 15 g of fast-acting
# carbohydrate" as contradicted. It was trained on general English, not clinical text, and small models are
# unreliable with numbers and units. Compare its errors with the rule-based checker's on your own claims:
# the two fail differently, so combining them (and sending every disagreement to a person) is safer than
# trusting either.

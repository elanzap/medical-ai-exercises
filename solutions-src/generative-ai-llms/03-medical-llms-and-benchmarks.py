# %% Step 1
for correct, n in [(88, 100), (1120, 1273)]:
    lo, hi = wilson_interval(correct, n)
    print(f"{correct}/{n}: {correct / n:.1%} (95% CI {lo:.1%} to {hi:.1%})")

n_needed = next(n for n in range(100, 20000, 10)
                if (lambda lo, hi: (hi - lo) / 2)(*wilson_interval(0.85 * n, n)) <= 0.02)
print(f"\nQuestions needed for a ±2-point interval at 85% accuracy: about {n_needed:,}")

# %% Note
# The usual approximation n ≈ 1.96² × p(1 − p) / 0.02² gives about 1,225, the same answer. Many published
# medical benchmarks, and most hospital-made test sets, are much smaller, so differences of a few points
# between models are often within the noise.

# %% Step 2
def compare(gap, seed=9, n=1273):
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(size=n)
    a = rng.random(n) < 1 / (1 + np.exp(-(2.0 - 1.5 * difficulty)))
    b = rng.random(n) < 1 / (1 + np.exp(-(2.0 + gap - 1.5 * difficulty)))
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return a.mean(), b.mean(), binomtest(a_only, a_only + b_only, 0.5).pvalue

for gap in [0.1, 0.2, 0.3, 0.5]:
    acc_a, acc_b, p = compare(gap)
    print(f"logit gap {gap}: A {acc_a:.1%}, B {acc_b:.1%} (difference {100 * (acc_b - acc_a):+.1f} points), McNemar p = {p:.3f}")

# %% Note
# McNemar's test uses only the questions where the two models disagree. With about 1,300 questions,
# a two-point difference is not convincing here (p = 0.14); from about three points the test reaches p < 0.05. A single
# simulation is itself one draw: repeating it with other seeds shows how often each gap would be detected.

# %% Step 3
rng = np.random.default_rng(10)
n = 1273
true_skill = rng.random(n) < 0.80                 # what the model can really do: about 80%
seen = rng.random(n) < 0.20                        # 20% of questions leaked into training data
answered = np.where(seen, rng.random(n) < 0.99, true_skill)
print(f"Accuracy on unseen questions:    {answered[~seen].mean():.1%}")
print(f"Accuracy on 'seen' questions:    {answered[seen].mean():.1%}")
print(f"Reported overall accuracy:       {answered.mean():.1%}")

# %% Note
# Contamination of a fifth of the test set adds about five points to the headline score, more than the
# gap between many competing models. Warning signs: a newer benchmark (or a rephrased version of an old one)
# gives a much lower score, or the model can reproduce questions word for word.

# %% Step 4
import pandas as pd

rng = np.random.default_rng(11)
n, agree = 1273, 0.85
results = []
for rep in range(200):
    a = rng.random(n) < 0.80
    b = rng.random(n) < 0.83                       # truly 3 points better
    flip_a, flip_b = rng.random(n) > agree, rng.random(n) > agree
    graded_a, graded_b = a ^ flip_a, b ^ flip_b    # the grader gets 15% of verdicts wrong, at random
    results.append({"true difference": 100 * (b.mean() - a.mean()),
                    "graded difference": 100 * (graded_b.mean() - graded_a.mean())})
print(pd.DataFrame(results).describe().loc[["mean", "std"]].round(2))

# %% Note
# With random grader errors, the measured gap shrinks towards zero by a factor of (2 × 0.85 − 1) = 0.7, so a
# true 3-point difference looks like about 2 points, and the noise around it grows. Real grader errors are
# rarely random: a grader model may favour longer answers or answers in its own style, which can bias the
# comparison in either direction. Model-graded results need a sample checked by clinicians.

# %% Step 5
# %% Note
# The fictional press release: *"Our model achieves expert-level performance in medicine, scoring 91% on a
# medical licensing benchmark, beating doctors."*
#
# Example appraisal:
#
# 1. A multiple-choice exam score measures recall and reasoning on exam-style questions, not the ability to
#    take a history, examine a patient, or manage uncertainty, so "expert-level performance in medicine" does
#    not follow from it.
# 2. The release gives no confidence interval, test-set size, or comparison on the same questions with other
#    models, so we cannot tell whether 91% differs meaningfully from competitors.
# 3. The benchmark is public and widely discussed online, so contamination of the training data is possible
#    and should be tested (for example on newly written or rephrased questions).
# 4. "Beating doctors" usually compares with average exam takers, often trainees, under different conditions;
#    it says nothing about patient outcomes.
# 5. Before clinical use, the claim would need prospective evaluation on the intended task and population,
#    with errors reviewed for severity, as reporting guidelines such as TRIPOD-LLM recommend.

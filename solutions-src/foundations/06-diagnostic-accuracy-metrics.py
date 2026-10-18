# %% Step 1
import pandas as pd

def simulate(prevalence, n=2000, seed=42):
    """The worked example's simulation as a function: same model, any prevalence."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, prevalence, size=n)
    logit = rng.normal(loc=np.where(y == 1, 1.5, -1.0), scale=1.0)
    return y, 1 / (1 + np.exp(-logit))

def metrics_at(y, prob, threshold):
    tn, fp, fn, tp = confusion_matrix(y, (prob >= threshold).astype(int)).ravel()
    return {"threshold": threshold, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "PPV": tp / (tp + fp), "NPV": tn / (tn + fn)}

THRESHOLDS = [0.3, 0.5, 0.7]
y_true, y_prob = simulate(0.10)
print(pd.DataFrame([metrics_at(y_true, y_prob, t) for t in THRESHOLDS]).round(2).to_string(index=False))

# %% Note
# Raising the threshold trades sensitivity for specificity: fewer patients are flagged, so fewer cases are
# found but a larger share of the flagged patients truly have the condition (PPV rises).

# %% Step 2
rows = []
for prevalence in [0.02, 0.10, 0.30]:
    y, prob = simulate(prevalence, n=20000)   # a larger sample keeps the 2% estimates stable
    m = metrics_at(y, prob, 0.5)
    rows.append({"prevalence": prevalence, **m, "AUROC": roc_auc_score(y, prob)})
print(pd.DataFrame(rows).drop(columns="threshold").round(2).to_string(index=False))

# %% Note
# Sensitivity, specificity and AUROC stay roughly the same, because they are calculated within the
# people with, or without, the condition. PPV and NPV move a lot, because they depend on how common the
# condition is: at 2% prevalence most positive results are false positives. A model's PPV from one
# hospital cannot be carried over to a setting with a different case mix.

# %% Step 3
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(4.5, 4.5))
RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name="Simulated model")
for t in THRESHOLDS:
    m = metrics_at(y_true, y_prob, t)
    ax.scatter(1 - m["specificity"], m["sensitivity"], zorder=3)
    ax.annotate(f"threshold {t}", (1 - m["specificity"], m["sensitivity"]),
                textcoords="offset points", xytext=(8, -12))
ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
plt.show()

# %% Step 4
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Fit the recalibration on one half and judge it on the other, so the comparison is fair.
logit_score = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
idx_fit, idx_test = train_test_split(np.arange(len(y_true)), test_size=0.5, random_state=0, stratify=y_true)
platt = LogisticRegression().fit(logit_score[idx_fit], y_true[idx_fit])
recalibrated = platt.predict_proba(logit_score[idx_test])[:, 1]

before = brier_score_loss(y_true[idx_test], y_prob[idx_test])
after = brier_score_loss(y_true[idx_test], recalibrated)
print(f"Mean predicted risk {y_prob[idx_test].mean():.2f} vs observed rate {y_true[idx_test].mean():.2f}")
print(f"Brier score before {before:.3f}, after Platt scaling {after:.3f}")

fig, ax = plt.subplots(figsize=(4.5, 4.5))
CalibrationDisplay.from_predictions(y_true[idx_test], y_prob[idx_test], n_bins=10, strategy="quantile", ax=ax, name="Original")
CalibrationDisplay.from_predictions(y_true[idx_test], recalibrated, n_bins=10, strategy="quantile", ax=ax, name="Platt-scaled")
plt.show()

# %% Note
# The original curve lies below the diagonal: the model **over-estimates** risk (it predicts far more events
# than happen). Platt scaling leaves the ranking, and so the AUROC, unchanged, but brings the predicted
# probabilities in line with observed rates, which lowers the Brier score.

# %% Step 5
# %% Note
# Example wording (numbers from the runs above, at 10% prevalence and threshold 0.5):
#
# "In simulated data with 10% prevalence, the model identified about nine in ten patients with the condition
# (sensitivity 0.92) and correctly ruled out about 85% of those without it, but only about four in ten
# positive results were true positives (PPV 0.41), and this share would be lower where the condition is
# rarer. Its original risk estimates were too high and needed recalibration before they could be read as
# probabilities; it has not been validated on real patients."

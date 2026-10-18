# %% Step 1
# The worked example fitted the model on the training split; collect the headline numbers in one place.
from scipy.optimize import brentq

def intercept_shift(y, log_odds):
    """Intercept with the slope fixed at 1: the shift that makes mean predicted risk equal the observed rate."""
    return brentq(lambda a: np.sum(y - 1 / (1 + np.exp(-(log_odds + a)))), -10, 10)

def calibration_slope_intercept(y, p):
    lo = np.log(p / (1 - p))
    slope = LogisticRegression(penalty=None).fit(lo.reshape(-1, 1), y).coef_[0][0]
    return slope, intercept_shift(y, lo)

p_test = model.predict_proba(test[predictors])[:, 1]
slope, intercept = calibration_slope_intercept(test.diabetes_5y.to_numpy(), p_test)
print(odds_ratios.round(2).to_string())
print(f"Test AUROC {roc_auc_score(test.diabetes_5y, p_test):.3f}; calibration slope {slope:.2f}, "
      f"calibration-in-the-large {intercept:+.2f}")

# %% Note
# The worked example's "intercept" came from fitting slope and intercept together. The usual summary of
# calibration-in-the-large fixes the slope at 1 and estimates only the intercept, as above; both are close to
# their ideal values (1 and 0) because the test set comes from the same population as the training set.

# %% Step 2
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

y_test = test.diabetes_5y.to_numpy()
observed, predicted = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")

def net_benefit(y, p, t):
    flagged = p >= t
    return (np.sum(flagged & (y == 1)) - np.sum(flagged & (y == 0)) * t / (1 - t)) / len(y)

thresholds = np.arange(0.05, 0.41, 0.01)
prev = y_test.mean()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(predicted, observed, "o-", label="Model")
ax1.plot([0, 1], [0, 1], ":", color="grey", label="Perfect calibration")
ax1.set(xlabel="Predicted risk", ylabel="Observed risk", title="Calibration (10 risk groups)")
ax1.legend()
ax2.plot(thresholds, [net_benefit(y_test, p_test, t) for t in thresholds], label="Model")
ax2.plot(thresholds, prev - (1 - prev) * thresholds / (1 - thresholds), label="Treat all")
ax2.axhline(0, color="black", linewidth=1, label="Treat none")
ax2.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve", ylim=(-0.05, 0.22))
ax2.legend()
plt.tight_layout()
plt.show()

# %% Step 3
def simulate_population(n, seed, bmi_mean=27, baseline=-17.5):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "age": rng.integers(30, 75, n),
        "bmi": rng.normal(bmi_mean, 4.5, n).round(1),
        "fasting_glucose": rng.normal(100, 10, n).round(),
        "family_history": rng.binomial(1, 0.3, n),
        "hypertension": rng.binomial(1, 0.35, n),
    })
    logit = (baseline + 0.03 * d.age + 0.10 * d.bmi + 0.11 * d.fasting_glucose
             + 0.6 * d.family_history + 0.4 * d.hypertension)
    d["diabetes_5y"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return d

new_pop = simulate_population(5000, seed=99, bmi_mean=31, baseline=-17.0)
recal_half, eval_half = new_pop.iloc[:2500], new_pop.iloc[2500:]
p_new = model.predict_proba(eval_half[predictors])[:, 1]
y_new = eval_half.diabetes_5y.to_numpy()
s, i = calibration_slope_intercept(y_new, p_new)
print(f"New population: incidence {y_new.mean():.1%}, mean predicted {p_new.mean():.1%}")
print(f"  AUROC {roc_auc_score(y_new, p_new):.3f}, slope {s:.2f}, calibration-in-the-large {i:+.2f}")

# Intercept-only recalibration, estimated on the other half of the new population
p_recal_half = model.predict_proba(recal_half[predictors])[:, 1]
shift = intercept_shift(recal_half.diabetes_5y.to_numpy(), np.log(p_recal_half / (1 - p_recal_half)))
p_recal = 1 / (1 + np.exp(-(np.log(p_new / (1 - p_new)) + shift)))
s2, i2 = calibration_slope_intercept(y_new, p_recal)
print(f"After intercept recalibration (shift {shift:+.2f}): mean predicted {p_recal.mean():.1%}, "
      f"AUROC {roc_auc_score(y_new, p_recal):.3f}, calibration-in-the-large {i2:+.2f}")

# %% Note
# A higher-BMI population with a higher baseline risk keeps a similar AUROC (the ranking still works), but
# the model under-predicts: the observed incidence is well above the mean prediction. Shifting the intercept
# fixes the average without touching the ranking, which is why AUROC is unchanged.

# %% Step 4
test_scored = test.assign(score=score.round().astype(int))
lookup = (test_scored.groupby("score")
          .agg(patients=("diabetes_5y", "size"), observed_risk=("diabetes_5y", "mean")))
lookup = lookup[lookup.patients >= 30]            # hide totals with too few patients to estimate
lookup["observed_risk"] = (lookup.observed_risk * 100).round(0).astype(int).astype(str) + "%"
print("Points (rounded total) and observed 5-year risk in the test set:")
print(lookup.to_string())

# %% Note
# In practice the table would be built on the development data and checked on the test data, and the
# smallest and largest totals would be grouped ("≤ 2", "≥ 12") so each row has enough patients.

# %% Step 5
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

candidates = {
    "Unpenalised logistic": model,
    "L2-penalised logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
    "Gradient boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                        early_stopping=True, random_state=0),
}
for name, m in candidates.items():
    if name != "Unpenalised logistic":
        m.fit(train[predictors], train.diabetes_5y)
    p = m.predict_proba(test[predictors])[:, 1]
    s, i = calibration_slope_intercept(y_test, p)
    print(f"{name:<22} AUROC {roc_auc_score(y_test, p):.3f}  Brier {brier_score_loss(y_test, p):.4f}  "
          f"slope {s:.2f}  CITL {i:+.2f}")

# %% Note
# The data were generated by a logistic model with no interactions or curves, so logistic regression is
# already the right shape. The L2 penalty changes almost nothing (3,500 training patients, five predictors).
# Gradient boosting does *worse*: lower AUROC and a calibration slope of about 0.7, the signature of
# overfitting (its predictions are too extreme). A more flexible model is not better by default; it helps when
# the true relationships are non-linear or interacting (see the next lesson), and it needs more data to show it.

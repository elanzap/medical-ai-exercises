# %% Step 1
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

print((table * 100).round(1).rename_axis("risk fifth").to_string())

def smoothed_calibration(p, outcome):
    """A flexible calibration curve: logistic regression of the outcome on a spline of the log-odds."""
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    smoother = make_pipeline(SplineTransformer(n_knots=4, degree=3), LogisticRegression(penalty=None, max_iter=1000))
    smoother.fit(lo, outcome)
    grid = np.linspace(np.quantile(lo, 0.01), np.quantile(lo, 0.99), 100).reshape(-1, 1)
    return 1 / (1 + np.exp(-grid.ravel())), smoother.predict_proba(grid)[:, 1]

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(*smoothed_calibration(p_new, new_site.diabetes_5y), label="New site, smoothed")
ax.scatter(table.predicted, table.observed, label="New site, fifths of risk")
ax.plot([0, 0.7], [0, 0.7], ":", color="grey", label="Perfect calibration")
ax.set(xlabel="Predicted risk", ylabel="Observed risk")
ax.legend()
plt.show()

# %% Step 2
from scipy.optimize import brentq

def recalibrate(part, slope_too):
    lo = log_odds(part).ravel()
    if slope_too:
        r = LogisticRegression(penalty=None).fit(lo.reshape(-1, 1), part.diabetes_5y)
        return lambda lo_new: r.intercept_[0] + r.coef_[0][0] * lo_new, (r.intercept_[0], r.coef_[0][0])
    a = brentq(lambda a: np.sum(part.diabetes_5y - 1 / (1 + np.exp(-(lo + a)))), -10, 10)
    return lambda lo_new: lo_new + a, (a, 1.0)

lo_eval = log_odds(eval_part).ravel()
for label, slope_too in [("Intercept only (slope fixed at 1)", False), ("Intercept and slope", True)]:
    f, (a, b) = recalibrate(recal_part, slope_too)
    p = 1 / (1 + np.exp(-f(lo_eval)))
    print(f"{label:<34} intercept {a:+.2f}, slope {b:.2f}: mean predicted {p.mean():.1%} "
          f"(observed {eval_part.diabetes_5y.mean():.1%})")

# %% Note
# The new site differs only in baseline risk, so the intercept does nearly all the work and the slope stays
# close to 1. Fixing the slope uses one parameter instead of two, which is more stable when the local
# recalibration sample is small.

# %% Step 3
rows = []
pool = simulate(20000, baseline=-18.5, seed=60)      # a larger sample from the same new site
for size in [200, 500, 2000]:
    for rep in range(50):
        sample = pool.sample(size, random_state=rep)
        f, (a, _) = recalibrate(sample, slope_too=False)
        rows.append({"sample size": size, "intercept": a,
                     "mean predicted": (1 / (1 + np.exp(-f(lo_eval)))).mean()})
stability = pd.DataFrame(rows).groupby("sample size").agg(
    intercept_mean=("intercept", "mean"), intercept_sd=("intercept", "std"),
    predicted_low=("mean predicted", lambda s: s.quantile(0.05)),
    predicted_high=("mean predicted", lambda s: s.quantile(0.95)))
print(stability.round(3).to_string())

# %% Note
# With 200 local patients (about 20 events) the recalibrated average risk can land anywhere in a wide range;
# with 2,000 it is much more stable. Recalibration needs enough local outcomes, not just enough patients.

# %% Step 4
y_eval = eval_part.diabetes_5y.to_numpy()
y = y_eval                      # net_benefit() from the worked example reads the global y
thresholds = np.arange(0.01, 0.41, 0.01)
prev = y_eval.mean()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds, [net_benefit(p_before, t) for t in thresholds], label="Model, before recalibration")
ax.plot(thresholds, [net_benefit(p_after, t) for t in thresholds], label="Model, after recalibration")
ax.plot(thresholds, prev - (1 - prev) * thresholds / (1 - thresholds), label="Treat all")
ax.axhline(0, color="black", linewidth=1, label="Treat none")
ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.02, prev + 0.01))
ax.legend()
plt.show()

# %% Note
# Before recalibration the model over-predicts, so at any threshold it flags too many people and its net
# benefit falls below that of the recalibrated model, dropping below zero at higher thresholds (worse than
# treating nobody). The ranking is the same; only the meaning of each threshold has changed.

# %% Step 5
def simulate_weaker_glucose(n, seed):
    d = simulate(n, baseline=-18.5, seed=seed)
    rng = np.random.default_rng(seed + 100)
    logit = (-12.5 + 0.03 * d.age + 0.10 * d.bmi + 0.05 * d.fasting_glucose     # glucose effect 0.05, not 0.11
             + 0.6 * d.family_history + 0.4 * d.hypertension)
    return d.assign(diabetes_5y=rng.binomial(1, 1 / (1 + np.exp(-logit))))

site3 = simulate_weaker_glucose(5000, seed=7)
recal3, eval3 = site3.iloc[:2000], site3.iloc[2000:]
lo3 = log_odds(eval3).ravel()
p_orig = 1 / (1 + np.exp(-lo3))
f3, (a3, b3) = recalibrate(recal3, slope_too=True)
p_recal3 = 1 / (1 + np.exp(-f3(lo3)))
refit = LogisticRegression(penalty=None, max_iter=1000).fit(recal3[predictors], recal3.diabetes_5y)
p_refit = refit.predict_proba(eval3[predictors])[:, 1]

for label, p in [("Original model", p_orig), (f"Recalibrated (slope {b3:.2f})", p_recal3), ("Refitted locally", p_refit)]:
    print(f"{label:<28} AUROC {roc_auc_score(eval3.diabetes_5y, p):.3f}   mean predicted {p.mean():.1%} "
          f"(observed {eval3.diabetes_5y.mean():.1%})")

# %% Note
# When one predictor matters less at the new site, the model ranks patients in the wrong order for that
# site. Recalibration rescales all the log-odds together (a slope below 1 here), which fixes the average
# risk but cannot reorder patients, so AUROC stays where it was. Refitting the coefficients (or updating the
# glucose coefficient alone) recovers the lost discrimination. Recalibration fixes *calibration* problems,
# not changes in how the predictors relate to the outcome.

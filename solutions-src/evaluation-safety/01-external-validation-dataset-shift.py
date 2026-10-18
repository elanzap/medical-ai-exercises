# %% Step 1
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

sites = {
    "Internal (same hospital)": site(3000, seed=2),
    "Older population": site(3000, seed=3, age_mean=70),
    "Different eGFR assay (+15)": site(3000, seed=4, assay_bias=15),
    "Lower baseline risk": site(3000, seed=5, baseline=-2.8),
}
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, d in sites.items():
    evaluate(name, d)
    obs, pred = calibration_curve(d.outcome, model.predict_proba(d[predictors])[:, 1], n_bins=10, strategy="quantile")
    ax.plot(pred, obs, "o-", label=name)
ax.plot([0, 0.7], [0, 0.7], ":", color="grey")
ax.set(xlabel="Predicted risk", ylabel="Observed risk", title="Calibration by site")
ax.legend(fontsize=8)
plt.show()

# %% Note
# AUROC barely changes between sites, but calibration does: at the assay-shift site the model under-predicts
# (a higher reported eGFR looks reassuring), and at the low-risk site it over-predicts. A discrimination-only
# validation would have passed both.

# %% Step 2
def concept_site(n, seed, albuminuria_effect=0.5):
    """Same predictors, but albuminuria matters half as much for the outcome."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"age": rng.normal(58, 10, n).round(), "true_egfr": rng.normal(70, 20, n).clip(15, 120),
                      "hba1c": rng.normal(7.8, 1.3, n).round(1), "albuminuria": rng.binomial(1, 0.3, n)})
    logit = (-2.0 + 0.02 * (d.age - 58) - 0.04 * (d.true_egfr - 70) + 0.3 * (d.hba1c - 7.8)
             + albuminuria_effect * d.albuminuria)
    d["outcome"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    d["egfr"] = d.true_egfr.round()
    return d

concept = concept_site(3000, seed=7)
evaluate("Concept shift (albuminuria x0.5)", concept)
p = model.predict_proba(concept[predictors])[:, 1]
for a in [0, 1]:
    m = concept.albuminuria == a
    print(f"  albuminuria={a}: predicted {p[m].mean():.1%}, observed {concept.outcome[m].mean():.1%}")

# %% Note
# The overall numbers look only mildly worse, but the subgroup check shows the problem: patients with albuminuria
# are now over-predicted. Concept shift changes the relationship between a predictor and the outcome, so
# recalibrating the whole model cannot fix it; the coefficient itself has to be updated, or the model refitted.

# %% Step 3
from scipy.optimize import brentq

def recalibrated_predictions(local, rest, slope_too=True):
    lo_local = np.log((pl := model.predict_proba(local[predictors])[:, 1]) / (1 - pl))
    lo_rest = np.log((pr := model.predict_proba(rest[predictors])[:, 1]) / (1 - pr))
    if slope_too:
        r = LogisticRegression(penalty=None).fit(lo_local.reshape(-1, 1), local.outcome)
        return r.predict_proba(lo_rest.reshape(-1, 1))[:, 1]
    a = brentq(lambda a: np.sum(local.outcome - 1 / (1 + np.exp(-(lo_local + a)))), -10, 10)
    return 1 / (1 + np.exp(-(lo_rest + a)))

for name in ["Different eGFR assay (+15)", "Lower baseline risk"]:
    d = sites[name]
    local, rest = d.iloc[:500], d.iloc[500:]
    before = model.predict_proba(rest[predictors])[:, 1]
    after = recalibrated_predictions(local, rest)
    print(f"{name:<28} observed {rest.outcome.mean():.1%}; predicted before {before.mean():.1%}, after {after.mean():.1%}; "
          f"AUROC {roc_auc_score(rest.outcome, before):.3f} -> {roc_auc_score(rest.outcome, after):.3f}")

# %% Note
# Recalibration on 500 local patients corrects the assay shift almost exactly (18.8% predicted against 18.4%
# observed) and brings the low-risk site much closer (19.5% to 13.3%, against 10.9%); with only about 55 local
# events, the correction is itself imprecise. AUROC does not change, because the ranking was never the problem. For the
# assay shift there is also a better fix: correct the eGFR values (or the assay's offset) before they reach the
# model, so the input means what it meant at development.

# %% Step 4
sizes = [200, 400, 2000]
intervals = []
rng = np.random.default_rng(0)
for n_ in sizes:
    sample = site(n_, seed=100 + n_)
    pp = model.predict_proba(sample[predictors])[:, 1]
    boot = []
    for _ in range(1000):
        i = rng.integers(0, n_, n_)
        if sample.outcome.iloc[i].nunique() == 2:
            boot.append(roc_auc_score(sample.outcome.iloc[i], pp[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    intervals.append((n_, sample.outcome.sum(), roc_auc_score(sample.outcome, pp), lo, hi))
    print(f"n={n_:>5}  events {sample.outcome.sum():>4}  AUROC {intervals[-1][2]:.3f}  95% CI {lo:.3f} to {hi:.3f}  (width {hi - lo:.3f})")

fig, ax = plt.subplots(figsize=(5, 3.5))
for n_, ev, auc, lo, hi in intervals:
    ax.errorbar(n_, auc, yerr=[[auc - lo], [hi - auc]], fmt="o", capsize=4)
ax.set(xscale="log", xlabel="Validation sample size (patients)", ylabel="AUROC with 95% CI", xticks=sizes)
ax.set_xticklabels([str(s) for s in sizes])
plt.show()

# %% Note
# The interval narrows roughly with the square root of the sample size, and what really matters is the number of
# events. Older rules of thumb asked for at least 100 to 200 events in a validation study; formal sample-size
# methods for external validation (Riley and colleagues, 2021) often require more, especially to estimate
# calibration precisely. Many published validations are smaller.

# %% Step 5
# %% Note
# Example validation plan before deploying this kidney-risk model at a new hospital:
#
# 1. **Intended use and population** written down: adults with type 2 diabetes in outpatient care, risk used to
#    prioritise nephrology review, not to withhold care.
# 2. **Local data check**: same predictor definitions, units and assays (eGFR equation and laboratory method), and
#    the same outcome definition and time horizon.
# 3. **Retrospective local validation** on recent patients, with enough events (aim for at least 100–200, more for
#    calibration): AUROC, calibration-in-the-large, calibration slope and a calibration plot, decision curve at the
#    intended threshold, all with confidence intervals.
# 4. **Subgroups**: age bands, sex, ethnicity or other locally relevant groups, albuminuria status, and eGFR bands.
# 5. **Pre-specified acceptance criteria** (for example calibration slope 0.8–1.2 and observed/expected 0.8–1.25);
#    if calibration fails but discrimination holds, recalibrate on local data and re-validate on separate patients.
# 6. **Silent period**: run the model alongside care without showing results for a few months, then compare.
# 7. **Monitoring plan** in place before go-live (see the monitoring lesson), with an owner and a switch-off rule.

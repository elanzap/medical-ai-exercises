# %% Step 1
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def make_month(r, n, assay_bias=0.0, age_mean=60):
    d = pd.DataFrame({"age": r.normal(age_mean, 10, n).round(),
                      "true_egfr": r.normal(70, 20, n).clip(15, 120),
                      "albuminuria": r.binomial(1, 0.3, n)})
    logit = -2.0 + 0.02 * (d.age - 60) - 0.04 * (d.true_egfr - 70) + 1.0 * d.albuminuria
    d["outcome"] = r.binomial(1, 1 / (1 + np.exp(-logit)))
    d["egfr"] = (d.true_egfr + assay_bias).round()
    return d

def psi_binary(expected, actual):
    e = np.clip(np.array([1 - expected.mean(), expected.mean()]), 1e-6, None)
    a = np.clip(np.array([1 - actual.mean(), actual.mean()]), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def monitor(months, predict=None, seed=21):
    r = np.random.default_rng(seed)
    predict = predict or (lambda d: model.predict_proba(d[predictors])[:, 1])
    rows = []
    for m, settings in months:
        d = make_month(r, 600, **settings)
        p = predict(d)
        oe = d.outcome.mean() / p.mean()
        rows.append({"month": m, "PSI eGFR": psi(development.egfr, d.egfr), "PSI age": psi(development.age, d.age),
                     "PSI albuminuria": psi_binary(development.albuminuria, d.albuminuria),
                     "O/E": oe, "AUROC": roc_auc_score(d.outcome, p), "data": d})
    return pd.DataFrame(rows).set_index("month")

assay_change = [(m, {"assay_bias": 15 if m >= 7 else 0}) for m in range(1, 13)]
mon = monitor(assay_change)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(mon.index, mon["PSI eGFR"], "o-"); ax1.axhline(0.2, ls="--", c="red"); ax1.set(title="eGFR PSI (alert above 0.2)", xlabel="month")
ax2.plot(mon.index, mon["O/E"], "o-"); ax2.axhspan(0.8, 1.25, color="green", alpha=0.1)
ax2.set(title="Observed / expected (acceptable 0.8-1.25)", xlabel="month")
plt.tight_layout(); plt.show()

# %% Step 2
print(mon[["PSI eGFR", "PSI age", "PSI albuminuria", "O/E", "AUROC"]].round(3).to_string())

# %% Note
# Only the eGFR distribution moves; age and albuminuria stay stable, which points straight at the cause (a
# laboratory change, not a change in patients). AUROC hardly moves, because a constant offset in eGFR does not
# change the ranking; calibration (O/E) is what breaks. Outcomes arrive late in practice (here, three years), so a
# real dashboard would show AUROC and O/E with that delay, and rely on input drift for early warning.

# %% Step 3
gradual = [(m, {"age_mean": 60 + m}) for m in range(1, 25)]
mon_age = monitor(gradual, seed=22)
flag = mon_age["PSI age"] > 0.2
first = flag.idxmax() if flag.any() else None
print(mon_age[["PSI age", "O/E", "AUROC"]].round(3).iloc[::3].to_string())
print(f"\nAge PSI first above 0.2 in month {first}; O/E outside 0.8-1.25 in months: "
      f"{list(mon_age.index[(mon_age['O/E'] < 0.8) | (mon_age['O/E'] > 1.25)])}")

# %% Note
# A slow drift stays under the alert limit for five months (age PSI first exceeds 0.2 in month 6) while the
# population changes steadily: each month looks
# almost like the last. The model here stays calibrated because age is in the model and its effect has not
# changed, so this drift is not harmful; input drift is a reason to look, not proof that the model is wrong.
# Comparing with the original baseline (not last month) is what lets slow drift show up at all.

# %% Step 4
from sklearn.linear_model import LogisticRegression

recal_data = pd.concat(mon.loc[7:9, "data"].tolist())
lo = lambda d: np.log((q := model.predict_proba(d[predictors])[:, 1]) / (1 - q)).reshape(-1, 1)
recalibrator = LogisticRegression(penalty=None).fit(lo(recal_data), recal_data.outcome)
print(f"Recalibration on months 7-9 ({len(recal_data)} patients): slope {recalibrator.coef_[0][0]:.2f}, "
      f"intercept {recalibrator.intercept_[0]:.2f}")

updated = lambda d: recalibrator.predict_proba(lo(d))[:, 1]
after = monitor([(m, {"assay_bias": 15}) for m in range(10, 13)], predict=updated, seed=23)
before = monitor([(m, {"assay_bias": 15}) for m in range(10, 13)], seed=23)
print(pd.DataFrame({"O/E before update": before["O/E"], "O/E after update": after["O/E"],
                    "AUROC after": after["AUROC"]}).round(2).to_string())

# %% Note
# After recalibration, O/E moves from about 1.5 to about 0.9, inside the acceptable range (slightly over-correcting,
# because three months of outcomes are a noisy basis). The eGFR PSI alert would keep firing against the original baseline,
# so after an approved update the baseline for input monitoring should be reset to the new assay (and the change
# documented). Using outcomes from months 7-9 assumes outcomes are known quickly; with a three-year outcome, the
# practical fix is to correct the eGFR offset between the assays.

# %% Step 5
# %% Note
# Example monitoring plan (kidney-risk model in clinic use):
#
# | Metric | Frequency | Alert limit | Owner | Response time |
# |---|---|---|---|---|
# | Input drift (PSI) for each predictor, against the validation baseline | monthly | PSI > 0.2 | data scientist | review within 2 weeks |
# | Missing-data rate per predictor | weekly | doubles from baseline | data engineer | 1 week |
# | Calibration (O/E and calibration slope), once outcomes are available | quarterly | O/E outside 0.8-1.25 | clinical lead | review within 1 month |
# | Discrimination (AUROC) with confidence interval | quarterly | below 0.70, or a fall of 0.05 | clinical lead | 1 month |
# | Subgroup calibration (age, sex, eGFR band) | twice a year | O/E outside 0.8-1.25 in any group | clinical safety officer | 1 month |
# | Upstream changes (lab assays, coding, EHR upgrades) | on notice | any change | IT change board | before go-live of the change |
#
# **Switching off:** the clinical safety officer can withdraw the model at any time; it is suspended automatically
# if inputs fail validation (for example a missing feed), and clinicians are told within one working day, with the
# fallback process (usual clinical assessment) described in advance. Every alert, decision and model version is
# logged, and an annual review decides whether to keep, update or retire the model.

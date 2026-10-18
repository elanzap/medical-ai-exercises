# %% Step 1
# The label-bias data were generated first, then overwritten by the subgroup example; rebuild them.
rng_lb = np.random.default_rng(7)
n_lb = 20000
group_lb = rng_lb.choice(["A", "B"], n_lb, p=[0.7, 0.3])
need = rng_lb.gamma(2.0, 1.0, n_lb)
spending = need * np.where(group_lb == "A", 1.0, 0.6) * rng_lb.lognormal(0, 0.3, n_lb)
lb = pd.DataFrame({"group": group_lb, "need": need, "spending": spending})

high_need = lb.need >= lb.need.quantile(0.90)
for label, score in [("by spending", lb.spending), ("by need", lb.need)]:
    sel = score >= score.quantile(0.90)
    shares = lb[sel].group.value_counts(normalize=True).round(2).to_dict()
    missed_b = 1 - sel[high_need & (lb.group == "B")].mean()
    print(f"Selecting {label:<12} selected shares {shares}; high-need group B patients missed {missed_b:.0%}")

# %% Note
# Selecting on need selects each group in proportion to its need (about 70/30), because need is distributed
# the same way in both groups. The bias came entirely from the label: spending measures access to care as well as
# need. Choosing the target variable is a fairness decision, and usually the most important one.

# %% Step 2
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

fig, ax = plt.subplots(figsize=(5, 4.5))
for g in ["A", "B"]:
    m = group == g
    print(f"group {g}: n={m.sum()}, AUROC {roc_auc_score(y[m], p[m]):.3f} (group-blind model), "
          f"{roc_auc_score(y[m], p2[m]):.3f} (with group)")
    obs, pred = calibration_curve(y[m], p[m], n_bins=10, strategy="quantile")
    ax.plot(pred, obs, "o-", label=f"group {g}, group-blind model")
ax.plot([0, 1], [0, 1], ":", color="grey")
ax.set(xlabel="Predicted risk", ylabel="Observed risk")
ax.legend()
plt.show()

# %% Note
# AUROC is similar in both groups, so the model ranks patients equally well within each. Calibration is not: the
# group-blind model under-predicts for group B (whose baseline risk is higher) and over-predicts for group A,
# because it predicts the same average for both.

# %% Step 3
def metrics(mask, flag):
    tp = np.sum(flag & (y == 1) & mask); fn = np.sum(~flag & (y == 1) & mask)
    fp = np.sum(flag & (y == 0) & mask); tn = np.sum(~flag & (y == 0) & mask)
    return tp / (tp + fn), tp / (tp + fp), fp / (fp + tn)

TARGET = 0.75
thresholds = {}
for g in ["A", "B"]:
    m = group == g
    # the threshold below which 25% of the group's cases fall finds the other 75%
    thresholds[g] = np.quantile(p[m & (y == 1)], 1 - TARGET)

print(f"{'':<26}{'sensitivity':>12}{'PPV':>7}{'FPR':>7}")
for label, thr in [("single threshold 0.2", {"A": 0.2, "B": 0.2}), (f"equal sensitivity {TARGET}", thresholds)]:
    flag = np.where(group == "A", p >= thr["A"], p >= thr["B"])
    for g in ["A", "B"]:
        s, ppv, fpr = metrics(group == g, flag)
        print(f"{label + ', group ' + g:<26}{s:>12.2f}{ppv:>7.2f}{fpr:>7.2f}   (threshold {thr[g]:.3f})")

# %% Note
# Equalising sensitivity needs a lower threshold for group B with the group-blind model. That raises group B's
# false-positive rate and changes PPV, so equal sensitivity, equal PPV and equal false-positive rates cannot all
# hold when outcome rates differ. Which one to prioritise is a clinical and ethical choice, not a technical one.

# %% Step 4
def simulate_groups(p_b, seed=8, n=20000):
    r = np.random.default_rng(seed)
    g = r.choice(["A", "B"], n, p=[1 - p_b, p_b])
    xx = r.normal(size=(n, 3))
    yy = r.binomial(1, 1 / (1 + np.exp(-(np.where(g == "A", -2.0, -1.0) + xx @ np.array([1.0, 0.7, 0.4])))))
    pp = LogisticRegression().fit(xx, yy).predict_proba(xx)[:, 1]
    return g, yy, pp

rng_b = np.random.default_rng(0)
for share in [0.20, 0.05]:
    g, yy, pp = simulate_groups(share)
    idx_b = np.where(g == "B")[0]
    sens, aucs = [], []
    for _ in range(500):
        i = rng_b.choice(idx_b, len(idx_b), replace=True)
        flag = pp[i] >= 0.2
        sens.append(np.sum(flag & (yy[i] == 1)) / np.sum(yy[i] == 1))
        aucs.append(roc_auc_score(yy[i], pp[i]))
    print(f"group B = {share:.0%} ({len(idx_b)} patients, {yy[idx_b].sum()} events): "
          f"sensitivity 95% CI {np.percentile(sens, 2.5):.2f}-{np.percentile(sens, 97.5):.2f}, "
          f"AUROC 95% CI {np.percentile(aucs, 2.5):.3f}-{np.percentile(aucs, 97.5):.3f}")

# %% Note
# At 20% of the data (about 4,000 patients) group B's sensitivity is estimated within about ±0.03; at 5% (about 950
# patients) the interval roughly doubles in width, and the AUROC interval too. With fewer patients or rarer outcomes
# it widens further. Small groups are where problems are most likely and hardest to see, so a fairness
# audit should report confidence intervals, and say when a group is too small to judge.

# %% Step 5
# %% Note
# Example fairness statement:
#
# "This synthetic model predicts the outcome from three clinical measurements; group membership is not an input.
# We prioritised calibration within each group, because the predicted risks are shown to clinicians as
# probabilities, and a risk that is systematically too low for one group would lead to under-treatment. The
# group-blind model was miscalibrated for group B, so we use a model that includes group, with which predicted
# and observed risks agree in both groups. As a result, at a single decision threshold, sensitivity is higher and
# PPV lower in group B than in group A; we report both, with confidence intervals, and review them annually and
# after any change to the population served. The outcome label was checked for measurement bias (it is a recorded
# clinical event, not a proxy such as cost)."

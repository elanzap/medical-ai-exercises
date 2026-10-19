# %% Step 1
import matplotlib.pyplot as plt

val_g, val_s, val_u = screening_population(20000)
gradable, referable = ~val_u, np.isin(val_g, list(REFERABLE))
thresholds = np.linspace(-2, 2, 81)
sens = [np.mean((val_s >= t)[gradable & referable]) for t in thresholds]
spec = [np.mean((val_s < t)[gradable & ~referable]) for t in thresholds]
per_1000 = [np.sum(gradable & (val_s >= t)) * 1000 / len(val_g) for t in thresholds]

fig, ax1 = plt.subplots(figsize=(6.5, 4))
ax1.plot(thresholds, sens, label="sensitivity"); ax1.plot(thresholds, spec, label="specificity")
ax1.axvline(threshold, ls="--", color="grey", label=f"90% sensitivity on development ({threshold:.2f})")
ax1.set(xlabel="Score threshold", ylabel="Proportion")
ax2 = ax1.twinx(); ax2.plot(thresholds, per_1000, color="black", alpha=0.5, label="referrals per 1,000")
ax2.set_ylabel("Referrals per 1,000 screened")
ax1.legend(loc="center left", fontsize=8); ax2.legend(loc="center right", fontsize=8)
plt.tight_layout(); plt.show()

# %% Step 2
rng_b = np.random.default_rng(10)
ref_dev = dev_s[~dev_u & np.isin(dev_g, list(REFERABLE))]
boot_thresholds = [np.quantile(rng_b.choice(ref_dev, len(ref_dev)), 0.10) for _ in range(1000)]
lo_t, hi_t = np.percentile(boot_thresholds, [2.5, 97.5])

def sensitivity_at(t, g=val_g, s=val_s, u=val_u):
    m = ~u & np.isin(g, list(REFERABLE))
    return np.mean(s[m] >= t)

boot_sens = [np.mean(rng_b.choice(ref_dev, len(ref_dev)) >= threshold) for _ in range(1000)]
print(f"Threshold {threshold:.2f}: development sensitivity 95% CI {np.percentile(boot_sens, 2.5):.1%}-{np.percentile(boot_sens, 97.5):.1%}")
print(f"Bootstrap range of the 90%-sensitivity threshold: {lo_t:.2f} to {hi_t:.2f}")
safer = lo_t
print(f"Safer threshold {safer:.2f}: validation sensitivity {sensitivity_at(safer):.1%} (vs {sensitivity_at(threshold):.1%}), "
      f"referrals per 1,000 {np.sum(~val_u & (val_s >= safer)) * 1000 / len(val_g):.0f} "
      f"(vs {np.sum(~val_u & (val_s >= threshold)) * 1000 / len(val_g):.0f})")

# %% Note
# The threshold chosen to give exactly 90% sensitivity on one development sample gives slightly less on new data (88%
# in the lesson), because about half the time sampling error puts it on the wrong side. Choosing the lower end of its
# bootstrap range builds in a margin, at the cost of a few more referrals per 1,000.

# %% Step 3
CAPACITY = 120
capacity_threshold = np.quantile(val_s[~val_u], 1 - CAPACITY * len(val_g) / 1000 / (~val_u).sum())
print(f"Threshold giving {CAPACITY} referrals per 1,000: {capacity_threshold:.2f}")
print(f"Sensitivity within capacity: {sensitivity_at(capacity_threshold):.1%}; "
      f"missed referable cases per 1,000: {np.sum(~val_u & referable & (val_s < capacity_threshold)) * 1000 / len(val_g):.1f}")

# %% Note
# With capacity for only 120 referrals per 1,000, the best achievable sensitivity is set by the model's ranking, and is
# well below 90%. That is a service-design problem, not a modelling one: options include more grading or clinic
# capacity, prioritising the highest scores for the fastest appointments, or using the model to triage which images a
# human grader sees first rather than to refer directly.

# %% Step 4
import pandas as pd

rng = np.random.default_rng(31)
months, rows = range(1, 9), []
baseline_rate = np.sum(~val_u & (val_s >= threshold)) / len(val_g)
for month in months:
    shift = -0.5 if month >= 4 else 0.0                    # new camera installed in month 4
    g, s, u = screening_population(1500, camera_shift=shift)
    referred = (~u & (s >= threshold)).mean()
    # A random sample of 100 gradable images per month is also graded by humans
    sample = rng.choice(np.where(~u)[0], 100, replace=False)
    ref_in_sample = np.isin(g[sample], list(REFERABLE))
    sample_sens = np.mean((s[sample] >= threshold)[ref_in_sample]) if ref_in_sample.any() else np.nan
    rows.append({"month": month, "referral rate": referred, "mean score": s[~u].mean(),
                 "ungradable": u.mean(), "graded-sample sensitivity": sample_sens})
report = pd.DataFrame(rows).set_index("month")
sd = np.sqrt(baseline_rate * (1 - baseline_rate) / 1500)
report["alert"] = np.where(report["referral rate"] < baseline_rate - 3 * sd, "REFERRAL RATE LOW", "")
print(f"Baseline referral rate {baseline_rate:.1%}; alert if below {baseline_rate - 3 * sd:.1%} (3 standard errors)")
print(report.round(3).to_string())

# %% Note
# The referral rate (available immediately, no outcomes needed) falls sharply from the month the camera changes and
# breaches its control limit in that same month; the mean score shows the same shift. The graded sample confirms that
# sensitivity has fallen, though 100 images contain only about 10 referable cases, so it is noisy month to month and
# better pooled over a quarter. A monthly report with these four lines would have detected the change within three
# months, and in this simulation within one.

# %% Step 5
# Optional: the imaging track's CNN on RetinaMNIST (MedMNIST, CC BY 4.0; grades 0-4), as a referable/non-referable
# model (grade 2 or more = referable), and its score distributions by grade.
import torch
from torch import nn
from medmnist import RetinaMNIST

def load(split):
    d = RetinaMNIST(split=split, download=True, root="data", size=28)
    X = torch.tensor(d.imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return (X - 0.5) / 0.25, d.labels.ravel()

Xr, gr = load("train"); Xt, gt = load("test")
yr = torch.tensor((gr >= 2).astype(int))
cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16 * 7 * 7, 2))
torch.manual_seed(0)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
for _ in range(15):
    for idx in torch.randperm(len(Xr)).split(64):
        opt.zero_grad(); nn.functional.cross_entropy(cnn(Xr[idx]), yr[idx]).backward(); opt.step()
with torch.no_grad():
    logits = cnn(Xt)
    score = (logits[:, 1] - logits[:, 0]).numpy()           # a log-odds-like score, as in the simulation
from sklearn.metrics import roc_auc_score
print(f"RetinaMNIST test AUROC for referable (grade >= 2): {roc_auc_score(gt >= 2, score):.3f}")
fig, ax = plt.subplots(figsize=(6, 3.5))
for grade in range(5):
    ax.hist(score[gt == grade], bins=20, alpha=0.5, density=True, label=f"grade {grade} (n={np.sum(gt == grade)})")
ax.set(xlabel="CNN score", ylabel="density", title="RetinaMNIST test scores by grade"); ax.legend(fontsize=8)
plt.show()

# %% Note
# Compared with the simulation, real scores overlap far more between neighbouring grades, and the grade mix is different
# (RetinaMNIST is not a screening population). That is why thresholds and monitoring limits must be set on data from the
# programme itself, not borrowed from a simulation or a different dataset.

# %% Step 1
import matplotlib.pyplot as plt

print(f"Clean: features {feature_model_auroc(signals, labels):.3f}, CNN {cnn_auroc(signals, labels):.3f}")
print(f"Noisy: features {feature_model_auroc(noisy_signals, noisy_labels):.3f}, CNN {cnn_auroc(noisy_signals, noisy_labels):.3f}")

t = np.arange(FS * SECONDS) / FS
fig, axes = plt.subplots(2, 2, figsize=(11, 4), sharex=True)
for row, (sig, lab, name) in enumerate([(signals, labels, "clean"), (noisy_signals, noisy_labels, "noisy")]):
    for col, cls in enumerate([0, 1]):
        i = np.where(lab == cls)[0][0]
        axes[row, col].plot(t, sig[i], linewidth=0.7)
        axes[row, col].set_title(f"{name}, {'irregular' if cls else 'regular'}", fontsize=9)
axes[1, 0].set_xlabel("seconds"); axes[1, 1].set_xlabel("seconds")
plt.tight_layout(); plt.show()

# %% Step 2
from scipy.stats import kurtosis

def quality_score(signal):
    """Kurtosis of the strip: clean ECG is dominated by sharp QRS spikes (high kurtosis); noise lowers it.
    Kurtosis-based signal-quality indices are a standard first check in ECG processing."""
    return float(kurtosis(signal))

q = np.array([quality_score(s) for s in noisy_signals])
print(f"Quality score: clean strips median {np.median([quality_score(s) for s in signals]):.1f}, "
      f"noisy strips median {np.median(q):.1f}")
for exclude in [0.0, 0.25, 0.5]:
    threshold = np.quantile(q, exclude)
    keep = q >= threshold
    excluded_irregular = noisy_labels[~keep].mean() if (~keep).any() else float("nan")
    print(f"exclude lowest {exclude:.0%} by quality: features AUROC {feature_model_auroc(noisy_signals[keep], noisy_labels[keep]):.3f}; "
          f"irregular share among excluded {excluded_irregular:.0%} (overall {noisy_labels.mean():.0%})")

# %% Note
# The kurtosis score separates clean from noisy strips clearly (medians 7.3 against 1.2), but excluding the
# lowest-quality noisy strips does not improve AUROC here. In this simulation every noisy strip has the same noise
# level, so the "worst" strips are not really worse; excluding them just shrinks the sample. In real recordings noise
# varies from strip to strip, and quality filtering usually helps on the strips kept. Either way, report the share
# excluded, and check whether exclusions fall unevenly on one class (slightly more irregular rhythms among the excluded
# here, 54% against 51% overall), because a quality filter can quietly remove the cases you most want to find.

# %% Step 3
def rr_features_3(signal):
    peaks, _ = find_peaks(signal, height=0.4, distance=int(0.3 * FS))
    rr = np.diff(peaks) / FS
    if len(rr) < 3:
        return [0.0, len(peaks), 0.0]
    pnn50 = np.mean(np.abs(np.diff(rr)) > 0.05)          # share of successive RR differences above 50 ms
    return [np.std(rr) / np.mean(rr), len(peaks), pnn50]

def auroc_with(feature_fn, sigs, labs):
    feats = np.array([feature_fn(s) for s in sigs])
    tr, te = train_test_split(np.arange(len(labs)), test_size=0.3, random_state=0, stratify=labs)
    p = LogisticRegression(max_iter=1000).fit(feats[tr], labs[tr]).predict_proba(feats[te])[:, 1]
    return roc_auc_score(labs[te], p)

for name, sigs, labs in [("clean", signals, labels), ("noisy", noisy_signals, noisy_labels)]:
    print(f"{name}: two features {auroc_with(rr_features, sigs, labs):.3f}, "
          f"with pNN50 {auroc_with(rr_features_3, sigs, labs):.3f}")

# %% Note
# pNN50 measures beat-to-beat irregularity directly, so it adds information that the overall coefficient of
# variation partly misses (a regular rhythm with occasional ectopic beats can have a high coefficient of variation
# but few large successive differences). At 100 samples per second, 50 ms is only five samples, so real ECG work
# uses higher sampling rates.

# %% Step 4
# %% Note
# Optional, outline (not run here because PTB-XL is a large download). PTB-XL is an openly available dataset of
# 21,837 12-lead ECGs from 18,885 patients on PhysioNet (Wagner et al., Scientific Data 2020), provided at 100 Hz and
# 500 Hz with diagnostic statements. Check its licence on PhysioNet before use.
#
# ```python
# import ast, numpy as np, pandas as pd, wfdb          # pip install wfdb
# root = "ptb-xl/"                                      # after downloading the dataset from PhysioNet
# meta = pd.read_csv(root + "ptbxl_database.csv", index_col="ecg_id")
# meta["codes"] = meta.scp_codes.apply(ast.literal_eval)
# meta["af"] = meta.codes.apply(lambda c: int("AFIB" in c))
# lead_ii = lambda f: wfdb.rdsamp(root + f)[0][:, 1]   # 100 Hz records: filename_lr; column 1 is lead II
# X = np.stack([lead_ii(f) for f in meta.filename_lr])
# # The dataset assigns every record to one of ten folds, keeping all of a patient's records in one fold.
# # The recommended split: folds 1-8 for training, 9 for validation, 10 for testing.
# train, val, test = meta.strat_fold <= 8, meta.strat_fold == 9, meta.strat_fold == 10
# ```
#
# Then train the same 1D CNN on `X[train]` (1,000 samples per record at 100 Hz, so the architecture fits directly),
# tune on `val`, and report AUROC with a confidence interval on `test` only once. Splitting by fold rather than at
# random keeps the same patient out of both training and test data.

# %% Step 1
# The imbalance example above reused the names X, y, X_train...; rebuild the lactate split first.
from sklearn.metrics import brier_score_loss

Xl, yl = df[["age", "heart_rate", "lactate"]], df.died
Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=0.3, random_state=0, stratify=yl)

def fit_eval(imputer, label, X_tr=Xl_train, y_tr=yl_train, X_te=Xl_test, y_te=yl_test):
    model = make_pipeline(imputer, StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    print(f"{label:<36} AUROC {roc_auc_score(y_te, p):.3f}  Brier {brier_score_loss(y_te, p):.4f}  "
          f"mean predicted {p.mean():.1%} (observed {y_te.mean():.1%})")
    return model

flagged = fit_eval(SimpleImputer(strategy="median", add_indicator=True), "Median imputation + missing flag")

# Complete-case analysis: keep only patients whose lactate was measured
cc = Xl_train.lactate.notna()
cc_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xl_train[cc], yl_train[cc])
measured_test = Xl_test.lactate.notna()
p_cc = cc_model.predict_proba(Xl_test[measured_test])[:, 1]
print(f"\nComplete-case model: trained on {cc.sum()} of {len(cc)} patients; "
      f"mean predicted {p_cc.mean():.1%} vs observed {yl_test.mean():.1%} in the full test set")

# %% Note
# The complete-case model learns from the sicker patients who had lactate measured, so its average
# prediction matches *their* death rate, nearly double the rate in the whole population. It also cannot
# produce a prediction at all for the majority whose lactate is missing.

# %% Step 2
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (needed to import IterativeImputer)
from sklearn.impute import IterativeImputer

fit_eval(SimpleImputer(strategy="median"), "Median imputation")
fit_eval(IterativeImputer(random_state=0), "Iterative imputation")
_ = fit_eval(IterativeImputer(random_state=0, add_indicator=True), "Iterative imputation + missing flag")

# %% Note
# Iterative imputation predicts lactate from age and heart rate. That helps a little, but it assumes the
# values are missing at random given those variables, which is false here: lactate is missing *because*
# patients look well. The missingness flag captures that directly, and it is what improves the model.

# %% Step 3
# New practice: lactate is measured in everyone. Same patients' biology, different ordering behaviour.
rng = np.random.default_rng(14)
m = 4000
sev = rng.normal(0, 1, m)
new = pd.DataFrame({
    "age": rng.integers(18, 90, m),
    "heart_rate": (85 + 6 * sev + rng.normal(0, 12, m)).round(),
    "lactate": (1.5 + 0.8 * sev + rng.normal(0, 0.5, m)).clip(0.3).round(1),   # never missing now
    "died": rng.binomial(1, 1 / (1 + np.exp(-(-3.0 + 1.6 * sev)))),
})
p_new = flagged.predict_proba(new[["age", "heart_rate", "lactate"]])[:, 1]
print(f"Flag model after the practice change: AUROC {roc_auc_score(new.died, p_new):.3f}, "
      f"mean predicted {p_new.mean():.1%} vs observed {new.died.mean():.1%}")

# %% Note
# The model learned "a missing lactate means low risk". When everyone is tested, nobody gets that reassurance,
# so the model now over-predicts (about 12.6% predicted against 11.0% observed here) even though the patients
# themselves have not changed. AUROC rises, because a real lactate value is now available for everyone, so a
# discrimination check alone would not catch the problem. A model that relies on *ordering behaviour* must be
# monitored for calibration, and refitted, when that behaviour changes.

# %% Step 4
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.calibration import CalibrationDisplay

# X, y, X_train... here are the imbalance example's (4.7% outcome rate)
samplers = {"No correction": None,
            "Random oversampling": RandomOverSampler(random_state=0),
            "SMOTE": SMOTE(random_state=0)}
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, sampler in samplers.items():
    Xs, ys = (X_train, y_train) if sampler is None else sampler.fit_resample(X_train, y_train)
    p = LogisticRegression().fit(Xs, ys).predict_proba(X_test)[:, 1]
    print(f"{name:<20} AUROC {roc_auc_score(y_test, p):.3f}  Brier {brier_score_loss(y_test, p):.4f}  "
          f"mean predicted {p.mean():.1%}")
    CalibrationDisplay.from_predictions(y_test, p, n_bins=10, strategy="quantile", ax=ax, name=name)
plt.show()

# %% Note
# Resampling leaves AUROC essentially unchanged (the ranking of patients is the same) but makes the
# predicted risks far too high, because the model is trained as if half of all patients had the outcome.
# The Brier score gets worse accordingly. If you need a different trade-off between missed cases and false
# alarms, change the decision threshold instead (next step).

# %% Step 5
from sklearn.metrics import confusion_matrix, roc_curve

p = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)[:, 1]
fpr, tpr, thr = roc_curve(y_test, p)
t80 = thr[np.argmax(tpr >= 0.80)]
tn, fp, fn, tp = confusion_matrix(y_test, p >= t80).ravel()
print(f"Threshold {t80:.3f}: sensitivity {tp / (tp + fn):.2f}, specificity {tn / (tn + fp):.2f}, "
      f"PPV {tp / (tp + fp):.2f}, flagged {(p >= t80).mean():.0%} of patients")

# %% Note
# To find 80% of the patients with this 4.7%-prevalence outcome, the model must flag a large share of
# everyone, and most flagged patients will not have it. That trade-off, not the imbalance itself, is what
# clinicians need to see when deciding whether a model is useful.

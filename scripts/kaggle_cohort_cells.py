"""Kaggle-only "Try it on a larger cohort" sections, added to selected lesson notebooks by build_kaggle.py.

Each section repeats the lesson's method on diabetes_clinic_cohort.csv from the Kaggle dataset
"Synthetic Diabetes Clinic Data for Medical AI". LOADER finds the file on Kaggle, or regenerates the same
cohort from the generator on GitHub when the notebook runs elsewhere.
"""

LOADER = '''
import glob, os, importlib.util, urllib.request
import numpy as np
import pandas as pd

hits = glob.glob('/kaggle/input/**/diabetes_clinic_cohort.csv', recursive=True)
if hits:
    cohort = pd.read_csv(hits[0])
else:  # not on Kaggle: regenerate the identical cohort (seed 42) from the generator on GitHub
    url = 'https://raw.githubusercontent.com/elanzap/medical-ai-exercises/main/scripts/generate_synthetic_cohort.py'
    urllib.request.urlretrieve(url, 'generate_synthetic_cohort.py')
    spec = importlib.util.spec_from_file_location('gen', 'generate_synthetic_cohort.py')
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    cohort = gen.generate(5000, 42)

FEATURES = ['age', 'diabetes_duration_years', 'bmi', 'systolic_bp', 'hba1c_percent', 'egfr', 'uacr_mg_g', 'ldl_mg_dl',
            'smoker', 'on_insulin', 'retinopathy', 'neuropathy', 'foot_ulcer_history']
X, y = cohort[FEATURES], cohort['admission_12m']
print(cohort.shape, '| admitted within 12 months:', f"{y.mean():.1%}")
'''

# Shared model pieces used by several sections.
_SPLIT = '''
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
logit = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
p_te = logit.predict_proba(X_te)[:, 1]
'''

SECTIONS = {
    "logistic-regression-risk-score": ("Fit the same kind of risk score to predict 12-month admission, and read the odds ratios per standard deviation.", _SPLIT + '''
print('Test AUROC:', round(roc_auc_score(y_te, p_te), 3))
coefs = logit.named_steps['logisticregression'].coef_[0]
pd.Series(np.exp(coefs), index=FEATURES).sort_values(ascending=False).round(2).rename('odds ratio per SD')
'''),
    "gradient-boosting-readmission": ("Compare gradient-boosted trees with logistic regression. HistGradientBoosting handles the missing lab values natively.", _SPLIT + '''
from sklearn.ensemble import HistGradientBoostingClassifier
gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X_tr, y_tr)
print('Logistic regression AUROC:', round(roc_auc_score(y_te, p_te), 3))
print('Gradient boosting AUROC:  ', round(roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1]), 3))
# Boosting does worse here: the true model is linear on the log-odds scale, so a flexible model mostly fits noise.
# A useful lesson in itself; try tuning max_depth and min_samples_leaf as an exercise.
'''),
    "explainability-shap": ("Explain a tree model's predictions on the cohort with SHAP values (permutation importance if SHAP isn't installed).", _SPLIT + '''
from sklearn.ensemble import GradientBoostingClassifier
imp = SimpleImputer(strategy='median').fit(X_tr)
Xi_tr, Xi_te = pd.DataFrame(imp.transform(X_tr), columns=FEATURES), pd.DataFrame(imp.transform(X_te), columns=FEATURES)
gb = GradientBoostingClassifier(random_state=0).fit(Xi_tr, y_tr)
try:
    import shap
    values = shap.TreeExplainer(gb).shap_values(Xi_te)
    importance = pd.Series(np.abs(values).mean(axis=0), index=FEATURES)
    label = 'mean |SHAP value|'
except ImportError:
    from sklearn.inspection import permutation_importance
    importance = pd.Series(permutation_importance(gb, Xi_te, y_te, scoring='roc_auc', random_state=0).importances_mean, index=FEATURES)
    label = 'permutation importance'
importance.sort_values(ascending=False).round(3).rename(label)
'''),
    "calibration-and-decision-curves": ("Check calibration and net benefit of a risk model on the cohort's test set.", _SPLIT + '''
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
obs, pred = calibration_curve(y_te, p_te, n_bins=8, strategy='quantile')
print('Brier score:', round(brier_score_loss(y_te, p_te), 3))
print(pd.DataFrame({'mean predicted': pred.round(3), 'observed': obs.round(3)}))
n = len(y_te)
for t in (0.1, 0.2, 0.3):
    tp = ((p_te >= t) & (y_te == 1)).sum(); fp = ((p_te >= t) & (y_te == 0)).sum()
    nb_model = tp / n - fp / n * t / (1 - t)
    nb_all = y_te.mean() - (1 - y_te.mean()) * t / (1 - t)
    print(f'threshold {t:.0%}: net benefit model {nb_model:.3f} | treat all {nb_all:.3f} | treat none 0.000')
'''),
    "missing-data-and-imbalance": ("UACR is missing at random (more often in younger patients); lipids and waist are missing completely at random. Compare approaches.", '''
print((cohort.isna().mean() * 100).round(1).loc[lambda s: s > 0].rename('% missing'))
print(cohort.assign(uacr_missing=cohort.uacr_mg_g.isna()).groupby(pd.cut(cohort.age, [17, 40, 55, 70, 90]), observed=True)['uacr_missing'].mean().round(3).rename('UACR missing, by age'))
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
cc = cohort.dropna(subset=FEATURES)
model = lambda imp: make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
print('complete cases only  n =', len(cc), 'AUROC', cross_val_score(model(SimpleImputer()), cc[FEATURES], cc.admission_12m, scoring='roc_auc', cv=5).mean().round(3))
print('median imputation    n =', len(X), 'AUROC', cross_val_score(model(SimpleImputer(strategy='median')), X, y, scoring='roc_auc', cv=5).mean().round(3))
print('iterative imputation n =', len(X), 'AUROC', cross_val_score(model(IterativeImputer(random_state=0)), X, y, scoring='roc_auc', cv=5).mean().round(3))
'''),
    "cohorts-and-data-leakage": ("See target leakage in action: a feature recorded after the outcome makes a model look far better than it really is.", _SPLIT + '''
rng = np.random.default_rng(0)
leaky = cohort.assign(discharge_summary_on_file=(cohort.admission_12m == 1) & (rng.random(len(cohort)) < 0.9))
Xl = leaky[FEATURES + ['discharge_summary_on_file']]
Xl_tr, Xl_te = Xl.loc[X_tr.index], Xl.loc[X_te.index]
leak_model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xl_tr, y_tr)
print('Honest model AUROC:', round(roc_auc_score(y_te, p_te), 3))
print('Leaky model AUROC: ', round(roc_auc_score(y_te, leak_model.predict_proba(Xl_te)[:, 1]), 3), '<- too good to be true')
'''),
    "unsupervised-phenotyping": ("Cluster the cohort into metabolic phenotypes with k-means, then see how admission rates differ between clusters.", '''
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
cols = ['age', 'bmi', 'hba1c_percent', 'systolic_bp', 'egfr', 'ldl_mg_dl', 'diabetes_duration_years']
Z = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(cohort[cols]))
cohort['cluster'] = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(Z)
cohort.groupby('cluster').agg(n=('age', 'size'), **{c: (c, 'mean') for c in cols}, admitted=('admission_12m', 'mean')).round(2)
'''),
    "bias-and-fairness": ("Audit one model's performance and calibration across sex and diabetes type.", _SPLIT + '''
test = cohort.loc[X_te.index].assign(p=p_te)
rows = []
for col in ('sex', 'diabetes_type'):
    for g, d in test.groupby(col):
        rows.append({'group': f'{col} = {g}', 'n': len(d), 'AUROC': roc_auc_score(d.admission_12m, d.p) if d.admission_12m.nunique() > 1 else np.nan,
                     'mean predicted': d.p.mean(), 'observed': d.admission_12m.mean()})
pd.DataFrame(rows).round(3)
'''),
    "external-validation-dataset-shift": ("Simulate dataset shift: develop the model in patients under 60 and validate it in patients 60 and over.", '''
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
dev, ext = cohort[cohort.age < 60], cohort[cohort.age >= 60]
m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=1000)).fit(dev[FEATURES], dev.admission_12m)
for name, d in (('development (<60)', dev), ('external (60+)', ext)):
    p = m.predict_proba(d[FEATURES])[:, 1]
    print(f'{name:18} n={len(d):5}  AUROC {roc_auc_score(d.admission_12m, p):.3f}  mean predicted {p.mean():.3f}  observed {d.admission_12m.mean():.3f}')
'''),
    "diagnostic-accuracy-metrics": ("Turn a risk model into a yes/no test at a threshold and compute the accuracy metrics from the lesson.", _SPLIT + '''
from sklearn.metrics import confusion_matrix
for t in (0.1, 0.2, 0.3):
    tn, fp, fn, tp = confusion_matrix(y_te, p_te >= t).ravel()
    print(f'threshold {t:.0%}: sensitivity {tp/(tp+fn):.2f}  specificity {tn/(tn+fp):.2f}  PPV {tp/(tp+fp):.2f}  NPV {tn/(tn+fn):.2f}')
print('AUROC (threshold-free):', round(roc_auc_score(y_te, p_te), 3))
'''),
}

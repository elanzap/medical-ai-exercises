# %% Step 1
y_test = test.readmit_30d.to_numpy()
fitted = {"Logistic regression": models["Logistic regression"],
          "Gradient boosting": models["Gradient boosting"]}
predictions = {name: m.predict_proba(test[features])[:, 1] for name, m in fitted.items()}
predictions["Logistic + clinical terms"] = lr2.predict_proba(add_terms(test)[engineered])[:, 1]
for name, p in predictions.items():
    print(f"{name:<26} AUROC {roc_auc_score(y_test, p):.3f}   Brier {brier_score_loss(y_test, p):.4f}")

# %% Step 2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

spline_lr = make_pipeline(
    ColumnTransformer([
        ("splines", SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), ["age", "sodium"]),
        ("rest", StandardScaler(), ["length_of_stay", "prior_admissions", "heart_failure", "lives_alone"]),
    ]),
    LogisticRegression(max_iter=2000),
)
spline_lr.fit(train[features], train.readmit_30d)
predictions["Logistic + splines"] = spline_lr.predict_proba(test[features])[:, 1]
p = predictions["Logistic + splines"]
print(f"Logistic + splines         AUROC {roc_auc_score(y_test, p):.3f}   Brier {brier_score_loss(y_test, p):.4f}")

# %% Note
# scikit-learn's SplineTransformer makes B-splines, not natural (restricted) cubic splines; with
# extrapolation="linear" they behave similarly at the edges. The splines let the model learn the U-shaped
# sodium risk and the rise after 65 without being told, recovering about two-thirds of the gap between the
# plain model and the hand-built clinical terms.
# What they cannot find is the heart-failure-by-living-alone interaction, which still has to be added.

# %% Step 3
from sklearn.model_selection import GridSearchCV, StratifiedKFold

search = GridSearchCV(
    HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=0),
    param_grid={"learning_rate": [0.02, 0.05, 0.1], "max_leaf_nodes": [4, 8, 15, 31]},
    scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1,
)
search.fit(train[features], train.readmit_30d)       # training data only: the test set stays untouched
cv = pd.DataFrame(search.cv_results_)[["param_learning_rate", "param_max_leaf_nodes", "mean_test_score"]]
print(cv.sort_values("mean_test_score", ascending=False).head(5).round(3).to_string(index=False))
tuned = search.best_estimator_
predictions["Gradient boosting, tuned"] = tuned.predict_proba(test[features])[:, 1]
p = predictions["Gradient boosting, tuned"]
print(f"\nBest {search.best_params_}: test AUROC {roc_auc_score(y_test, p):.3f}, Brier {brier_score_loss(y_test, p):.4f}")

# %% Step 4
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

iso = CalibratedClassifierCV(
    HistGradientBoostingClassifier(**search.best_params_, max_iter=300, early_stopping=False, random_state=0),
    method="isotonic", cv=5).fit(train[features], train.readmit_30d)
predictions["Gradient boosting, tuned + isotonic"] = iso.predict_proba(test[features])[:, 1]

fig, ax = plt.subplots(figsize=(6, 5))
for name, p in predictions.items():
    CalibrationDisplay.from_predictions(y_test, p, n_bins=10, strategy="quantile", ax=ax, name=name)
ax.set_title("Calibration on the test set")
plt.show()

for name, p in predictions.items():
    print(f"{name:<36} Brier {brier_score_loss(y_test, p):.4f}   mean predicted {p.mean():.3f} (observed {y_test.mean():.3f})")

# %% Note
# All the models are reasonably calibrated here, because training and test data come from the same
# population; isotonic recalibration changes the boosted model's Brier score only slightly. Recalibration
# matters most when a model is moved to a new setting (lesson 8) or when it overfits.

# %% Step 5
import shap

sv = shap.TreeExplainer(tuned)(test[features])
t = test.reset_index(drop=True)
alone_shap = pd.Series(sv[:, "lives_alone"].values)
print("Mean SHAP value for lives_alone = 1, by heart failure:")
print(alone_shap[t.lives_alone == 1].groupby(t.heart_failure[t.lives_alone == 1]).mean().round(2).to_string())

shap.plots.scatter(sv[:, "lives_alone"], color=sv[:, "heart_failure"])

# %% Note
# Permutation importance gave lives_alone almost no importance, because on its own it matters only for the
# 15% of patients with heart failure. SHAP values, which are calculated patient by patient, show the
# interaction directly: living alone adds clearly more risk for patients with heart failure than for those
# without. The scatter plot colours each point by heart failure, so the two groups separate visibly.

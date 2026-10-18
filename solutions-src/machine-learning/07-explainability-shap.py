# %% Step 1
print(global_importance.sort_values(ascending=False).round(3).to_string())
shap.plots.beeswarm(sv)

# %% Note
# Each dot is one patient. Its position shows how much that feature moved the patient's predicted log-odds,
# and its colour shows the feature's value. For age, high values (red) push risk up; for sodium, both very
# low and very high values push it up, which a single coefficient could not show.

# %% Step 2
p_test = model.predict_proba(test[features])[:, 1]
order = np.argsort(p_test)
picks = {"lowest risk": order[0], "median risk": order[len(order) // 2], "highest risk": order[-1]}
for label, i in picks.items():
    print(f"{label}: predicted risk {p_test[i]:.1%}")
    shap.plots.waterfall(sv[i], show=True)

# %% Note
# The waterfall starts at the average log-odds and adds each feature's contribution until it reaches this
# patient's prediction. Explanations are on the log-odds scale; large-looking contributions can still mean a
# small change in probability for a low-risk patient.

# %% Step 3
interactions = explainer.shap_interaction_values(test[features])
mean_abs = pd.DataFrame(np.abs(interactions).mean(axis=0), index=features, columns=features)
off_diagonal = mean_abs.where(~np.eye(len(features), dtype=bool)).stack().sort_values(ascending=False)
print("Strongest pairwise interactions (mean |SHAP interaction value|, each pair counted twice):")
print(off_diagonal.iloc[::2].head(5).round(4).to_string())

# %% Note
# heart_failure × lives_alone is among the strongest pairs (second here), matching the one interaction built
# into the simulation. The others (length of stay × age, prior admissions × age...) are not in the simulation:
# the trees approximate smooth effects with splits, and some of that shows up as small spurious interactions.
# Interaction values point you to candidates; checking them (as in the worked example's grouped means)
# is still needed.

# %% Step 4
rng3 = np.random.default_rng(2)
def with_leak(d):
    # A flag recorded AFTER the outcome: follow-up booked because the patient was readmitted
    booked = np.where(d.readmit_30d == 1, rng3.binomial(1, 0.7, len(d)), rng3.binomial(1, 0.05, len(d)))
    return d.assign(followup_booked_after=booked)

leak_features = features + ["followup_booked_after"]
train_l, test_l = with_leak(train), with_leak(test)
leaky_model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=15,
                                             early_stopping=True, random_state=0)
leaky_model.fit(train_l[leak_features], train_l.readmit_30d)
from sklearn.metrics import roc_auc_score
print(f"AUROC without leak {roc_auc_score(test.readmit_30d, p_test):.3f}, "
      f"with leak {roc_auc_score(test_l.readmit_30d, leaky_model.predict_proba(test_l[leak_features])[:, 1]):.3f}")
sv_leak = shap.TreeExplainer(leaky_model)(test_l[leak_features])
print(pd.Series(np.abs(sv_leak.values).mean(axis=0), index=leak_features).sort_values(ascending=False).round(3).to_string())

# %% Note
# A sudden jump in AUROC together with one feature dominating every explanation is the classic sign of a
# leak. SHAP does not know the feature is illegitimate, but it makes the model's dependence on it obvious,
# so a clinician reviewing the explanation can ask "when is this recorded?"

# %% Step 5
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train[features])
lr = LogisticRegression(max_iter=2000).fit(scaler.transform(train[features]), train.readmit_30d)
lr_sv = shap.LinearExplainer(lr, scaler.transform(train[features]))(scaler.transform(test[features]))
comparison = pd.DataFrame({
    "LR coefficient (per SD)": lr.coef_[0],
    "LR mean |SHAP|": np.abs(lr_sv.values).mean(axis=0),
    "Boosting mean |SHAP|": global_importance,
}, index=features)
print(comparison.round(3).sort_values("Boosting mean |SHAP|", ascending=False).to_string())

# %% Note
# The two models disagree mainly where the truth is non-linear or interacting:
#
# - **Sodium**: risk rises at both low and high values. A linear model can only fit one slope, so its
#   coefficient is near zero and sodium looks unimportant; boosting captures the U-shape.
# - **Age**: risk rises only after 65. The linear model spreads this over all ages.
# - **Lives alone**: matters only with heart failure. Logistic regression without an interaction term gives
#   it one average effect for everyone; in boosting its contribution is concentrated in patients with heart
#   failure, so its average over all patients is smaller even though it matters a lot for that subgroup.
#
# Neither set of numbers says what would happen if you changed a feature; both describe what each model
# relies on.

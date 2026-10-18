# %% Step 1
# The worked example above already ran the simulation (seed 0) and fitted both models.
print(f"Unadjusted: {naive.coef_[0][0]:+.2f}   adjusted for severity: {adjusted.coef_[0][0]:+.2f}   true effect: -0.80")

# %% Note
# The adjusted estimate (−0.87) is close to the true −0.8. The model matches how the data were generated,
# so the remaining gap is sampling noise and would shrink with a larger sample. Without adjustment the
# sign is wrong: the treatment looks harmful because sicker patients receive it.

# %% Step 2
def simulate_with_frailty(seed=0, n=20000, frailty_strength=1.0):
    rng = np.random.default_rng(seed)
    severity = rng.normal(0, 1, n)
    frailty = rng.normal(0, 1, n)                                  # never recorded
    treated = rng.binomial(1, 1 / (1 + np.exp(-(2 * severity + frailty_strength * frailty))))
    logit = -1 + 1.5 * severity + frailty_strength * frailty - 0.8 * treated
    died = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({"severity": severity, "frailty": frailty, "treated": treated, "died": died})

d = simulate_with_frailty()
without = LogisticRegression(penalty=None).fit(d[["treated", "severity"]], d.died)
with_frailty = LogisticRegression(penalty=None).fit(d[["treated", "severity", "frailty"]], d.died)
print(f"Adjusted for severity only:      {without.coef_[0][0]:+.2f}")
print(f"Adjusted for severity + frailty: {with_frailty.coef_[0][0]:+.2f}   (only possible in a simulation)")

# %% Note
# Frailty makes patients both more likely to be treated and more likely to die, so leaving it out pulls the
# estimate back towards "treatment is harmful". With real routine data you cannot fit the second model,
# because the confounder was never recorded. Nothing in the data warns you: the first model fits just as
# well. This is why causal claims from observational data need an explicit argument about confounders
# (and sensitivity analyses), not just a good model fit.

# %% Step 3
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df.died)
gb = HistGradientBoostingClassifier(random_state=0).fit(train[["treated", "severity"]], train.died)
print(f"Test AUROC {roc_auc_score(test.died, gb.predict_proba(test[['treated', 'severity']])[:, 1]):.3f}")

imp = permutation_importance(gb, test[["treated", "severity"]], test.died, scoring="roc_auc",
                             n_repeats=10, random_state=0)
print(pd.Series(imp.importances_mean, index=["treated", "severity"]).round(3))

# %% Note
# The importances say that "treated" carries some information for *predicting* death, but not in which
# direction, and not whether changing treatment would change the outcome. Importance measures how much the
# model's predictions rely on a feature, which mixes the treatment's real effect with everything it is
# correlated with (here, severity). A model that predicts well can still be useless, or misleading, for
# deciding whether to treat.

# %% Step 4
# %% Note
# Example target trial protocol (the hypothetical randomised trial you would like to run, emulated with
# routine data):
#
# | Element | Specification |
# |---|---|
# | **Eligibility** | Adults admitted with the condition, meeting the treatment's licensed indication, with no contraindication and not already on the treatment |
# | **Treatment strategies** | (1) start the treatment within 24 hours of admission; (2) do not start it in that window |
# | **Assignment** | Randomisation in the target trial; in the emulation, adjustment for all confounders measured at time zero (severity score, frailty measures, age, comorbidities) |
# | **Time zero** | Admission, the moment eligibility is met and a strategy is chosen; follow-up starts here for both groups, which avoids immortal-time bias |
# | **Outcome** | Death from any cause |
# | **Follow-up** | From time zero to 30 days, death, or loss to follow-up, whichever comes first |
# | **Causal contrast** | Intention-to-treat analogue (as started) and per-protocol (as adhered to, with adjustment for reasons for deviating) |
# | **Analysis** | Risk difference and risk ratio at 30 days, using weighting or standardisation; a sensitivity analysis for unmeasured confounding such as frailty |
#
# Writing it down forces the decisions that a "just fit a model" approach leaves implicit.

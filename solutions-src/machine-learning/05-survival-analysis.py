# %% Step 1
import matplotlib.pyplot as plt

print(f"Naive 5-year risk {naive:.1%}; Kaplan-Meier 5-year risk {1 - kmf.predict(5):.1%}")

fig, ax = plt.subplots(figsize=(6, 4))
for value, label in [(0, "No albuminuria"), (1, "Albuminuria")]:
    group = df[df.uacr_high == value]
    KaplanMeierFitter().fit(group.years, group.kidney_failure, label=label).plot_survival_function(ax=ax)
ax.set(xlabel="Years", ylabel="Free of kidney failure", title="Kaplan-Meier curves (synthetic cohort)")
plt.show()

# %% Note
# The naive estimate is too low because patients censored early (followed for less than five years) are
# counted as if they could never have an event. Kaplan-Meier uses each patient only for the time they were
# actually observed.

# %% Step 2
print(cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3))
_ = cph.check_assumptions(df, p_value_threshold=0.05, show_plots=False)   # prints its advice

# %% Note
# Compare with the values used in the simulation: age exp(0.02) = 1.02 per year, eGFR exp(−0.04) = 0.96 per
# mL/min/1.73m², albuminuria exp(0.8) = 2.23 and SGLT2 inhibitor exp(−0.3) = 0.74. Three estimates are close;
# albuminuria's 2.77 is higher, with the true 2.23 just at the lower end of its confidence interval. That is
# what sampling variation looks like with a few hundred events: a different seed gives a different estimate.
# The data were simulated with constant hazard ratios, and the proportional-hazards check finds no problem;
# with real data, a flagged variable is often handled by stratifying on it or letting its effect vary with time.

# %% Step 3
from lifelines import AalenJohansenFitter

rng = np.random.default_rng(22)
death_hazard = 0.03 * np.exp(0.06 * (df.age - 60))         # older patients die sooner, from other causes
death_time = rng.exponential(1 / death_hazard)
first = np.minimum.reduce([event_time, death_time, censor_time])
cr = pd.DataFrame({"years": first,               # unrounded: Aalen-Johansen needs untied times
                   "event": np.select([first == event_time, first == death_time], [1, 2], 0)})
print(cr.event.value_counts().rename({0: "censored", 1: "kidney failure", 2: "death"}).to_string())

km_cr = KaplanMeierFitter().fit(cr.years, cr.event == 1)          # treats deaths as censored: wrong here
aj = AalenJohansenFitter(seed=0).fit(cr.years, cr.event, event_of_interest=1)
cif_5 = aj.cumulative_density_.loc[:5].iloc[-1, 0]
print(f"\n1 − Kaplan-Meier at 5 years:     {1 - km_cr.predict(5):.1%}")
print(f"Aalen-Johansen at 5 years:       {cif_5:.1%}")

# %% Note
# 1 − Kaplan-Meier answers "what would the risk be if nobody could die of anything else", which is not the
# risk a patient faces. Patients who die cannot go on to kidney failure, so the Aalen-Johansen cumulative
# incidence is lower, and the gap grows with the death rate (here, with age).

# %% Step 4
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=0)
cox = CoxPHFitter().fit(train, duration_col="years", event_col="kidney_failure")
risk_score = cox.predict_partial_hazard(test)
print(f"Held-out C-index {concordance_index(test.years, -risk_score, test.kidney_failure):.3f}")

pred_5y = 1 - cox.predict_survival_function(test, times=[5]).iloc[0].to_numpy()
test = test.assign(pred_5y=pred_5y, group=pd.qcut(pred_5y, 5, labels=False))
rows = []
for g, d in test.groupby("group"):
    km = KaplanMeierFitter().fit(d.years, d.kidney_failure)
    rows.append({"predicted": d.pred_5y.mean(), "observed": 1 - km.predict(5)})
cal = pd.DataFrame(rows)
print((cal * 100).round(1).rename_axis("risk fifth").to_string())

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(cal.predicted, cal.observed, "o-")
ax.plot([0, cal.max().max()], [0, cal.max().max()], ":", color="grey")
ax.set(xlabel="Predicted 5-year risk", ylabel="Observed (Kaplan-Meier) 5-year risk", title="Calibration at 5 years")
plt.show()

# %% Note
# Observed risk in each group is estimated with Kaplan-Meier, not by counting events, because many test
# patients were censored before five years. Predicted and observed risks rise together, but with about 900
# test patients each fifth has only a couple of dozen events, so individual groups can miss by a few points.

# %% Step 5
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

X_cols = ["age", "egfr", "uacr_high", "sglt2i"]
y_train = Surv.from_arrays(train.kidney_failure.astype(bool), train.years)
rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, n_jobs=-1, random_state=0)
rsf.fit(train[X_cols], y_train)
rsf_risk = rsf.predict(test[X_cols])
print(f"Random survival forest C-index {concordance_index(test.years, -rsf_risk, test.kidney_failure):.3f}")
print(f"Cox model C-index              {concordance_index(test.years, -risk_score, test.kidney_failure):.3f}")

# %% Note
# The data follow a Cox model exactly, so the forest has nothing extra to learn and does no better (often a
# little worse). Flexible survival models earn their place when effects are non-linear or interacting, and
# they should be compared on calibration as well as the C-index.

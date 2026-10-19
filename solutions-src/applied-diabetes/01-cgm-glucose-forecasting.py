# %% Step 1
def evaluate_horizon(trace, horizon, history=6, train_days=10, verbose=True):
    X = np.array([trace[t - history:t] for t in range(history, len(trace) - horizon)])
    y = np.array([trace[t + horizon - 1] for t in range(history, len(trace) - horizon)])
    split = train_days * 288 - history
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
    preds = {
        "Persistence": X_te[:, -1],
        "Linear trend": X_te[:, -1] + (X_te[:, -1] - X_te[:, -4]) / 3 * horizon,
        "Ridge": Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_te),
    }
    low, new_low = y_te < 70, (y_te < 70) & (X_te[:, -1] >= 70)
    days = len(y_te) / 288
    out = {}
    for name, p in preds.items():
        out[name] = {"RMSE": np.sqrt(np.mean((p - y_te) ** 2)), "MARD": np.mean(np.abs(p - y_te) / y_te),
                     "new lows in advance": np.mean((p < 80)[new_low]) if new_low.any() else np.nan,
                     "false alerts/day": np.sum((p < 80) & ~low) / days}
        if verbose:
            o = out[name]
            print(f"  {name:<13} RMSE {o['RMSE']:5.1f}  MARD {o['MARD']:5.1%}  new lows in advance "
                  f"{o['new lows in advance']:4.0%}  false alerts/day {o['false alerts/day']:4.1f}")
    return out

for minutes in (30, 60):
    print(f"Horizon {minutes} minutes:")
    evaluate_horizon(cgm, horizon=minutes // 5)

# %% Note
# At 60 minutes the error of persistence and ridge regression rises by about 60% (ridge RMSE 9.3 to 14.8 mg/dL), and
# the linear trend, which extrapolates the last 15 minutes for an hour, almost doubles (16.7 to 31.0). Ridge catches far
# fewer lows in advance at 60 minutes (42% against 62%), although with fewer false alerts. Longer warnings are more useful clinically but less accurate; the horizon is a
# design choice to make with users, not a technical default.

# %% Step 2
import pandas as pd

people = {f"person {i}": synthetic_cgm(seed=i) for i in range(10)}

def windows(trace, history=6, horizon=6):
    X = np.array([trace[t - history:t] for t in range(history, len(trace) - horizon)])
    y = np.array([trace[t + horizon - 1] for t in range(history, len(trace) - horizon)])
    return X, y

rows = []
for held_out in people:
    # train on days 1-10 of the OTHER nine people; test on days 11-14 of the held-out person
    split = 10 * 288 - 6
    Xs, ys = zip(*(windows(tr) for name, tr in people.items() if name != held_out))
    X_tr = np.vstack([x[:split] for x in Xs]); y_tr = np.concatenate([y[:split] for y in ys])
    X_te, y_te = (a[split:] for a in windows(people[held_out]))
    pred = Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_te)
    rows.append({"held-out person": held_out, "RMSE": np.sqrt(np.mean((pred - y_te) ** 2)),
                 "persistence RMSE": np.sqrt(np.mean((X_te[:, -1] - y_te) ** 2))})
by_person = pd.DataFrame(rows).set_index("held-out person")
print(by_person.round(1).to_string())
print(f"\nRidge RMSE across people: mean {by_person.RMSE.mean():.1f}, range {by_person.RMSE.min():.1f}-{by_person.RMSE.max():.1f}")

# %% Note
# Splitting by person tests what matters for a product: performance for someone the model has never seen. The spread
# across people is as important as the mean. These synthetic people all come from one simple generator, so they differ
# far less than real people with diabetes do.

# %% Step 3
def synthetic_cgm_with_meals(days=14, seed=0):
    """The lesson's generator, also returning the time step of each meal."""
    rng = np.random.default_rng(seed)
    steps = days * 288
    glucose, meals = np.empty(steps), []
    g, meal_effect = 130.0, 0.0
    for t in range(steps):
        minute = (t % 288) * 5
        if minute in (8 * 60, 13 * 60, 20 * 60) and rng.random() < 0.95:
            meal_effect += rng.uniform(60, 160); meals.append(t)
        if minute == 3 * 60 and rng.random() < 0.4:
            meal_effect -= rng.uniform(120, 170)
        meal_effect *= 0.965
        g += 0.03 * (125 - g) + 0.04 * meal_effect + rng.normal(0, 1.5)
        glucose[t] = g
    return np.clip(glucose + rng.normal(0, 4, steps), 40, 400), np.array(meals)

trace, meals = synthetic_cgm_with_meals()
assert np.allclose(trace, cgm)                               # same trace as the lesson
idx = np.arange(6, len(trace) - 6)
minute_of_day = (idx % 288) * 5
since_meal = np.array([(t - meals[meals < t].max()) * 5 if (meals < t).any() else 600 for t in idx])
extra = np.column_stack([np.sin(2 * np.pi * minute_of_day / 1440), np.cos(2 * np.pi * minute_of_day / 1440),
                         np.minimum(since_meal, 600) / 600])
Xw, yw = windows(trace)
split = 10 * 288 - 6
for label, X_all in [("glucose history only", Xw), ("+ time of day, time since meal", np.hstack([Xw, extra * 50]))]:
    p = Ridge(alpha=1.0).fit(X_all[:split], yw[:split]).predict(X_all[split:])
    y_te, last = yw[split:], Xw[split:, -1]
    new_low = (y_te < 70) & (last >= 70)
    print(f"{label:<32} RMSE {np.sqrt(np.mean((p - y_te) ** 2)):.1f}  new lows in advance "
          f"{np.mean((p < 80)[new_low]):.0%}  false alerts/day {np.sum((p < 80) & (y_te >= 70)) / 4:.1f}")

# %% Note
# The context features improve the overall error a little (RMSE 9.3 to 8.7) and slightly reduce false alerts, but do
# not catch more lows in advance (62% either way). In this simulation the overnight lows happen at 3 a.m. on a random
# 40% of nights, so time of day says only "a low is possible now", which a linear model cannot turn into a warning for
# a particular night. Real systems add insulin doses, carbohydrate entries and activity, which carry the information
# that time of day lacks. (The extra features are scaled up so the ridge penalty does not shrink them away.)

# %% Step 4
import matplotlib.pyplot as plt

def clarke_zone(ref, pred):
    """Clarke error grid zone for one pair (mg/dL), following Clarke et al., Diabetes Care 1987;10:622-628."""
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
        return "D"
    return "B"

ridge_pred = Ridge(alpha=1.0).fit(Xw[:split], yw[:split]).predict(Xw[split:])
ref = yw[split:]
zones = np.array([clarke_zone(r, p) for r, p in zip(ref, ridge_pred)])
print(pd.Series(zones).value_counts().reindex(list("ABCDE"), fill_value=0).to_string())
fig, ax = plt.subplots(figsize=(5, 5))
for z, colour in zip("ABCDE", ["tab:green", "tab:olive", "tab:orange", "tab:red", "black"]):
    m = zones == z
    ax.scatter(ref[m], ridge_pred[m], s=4, color=colour, label=f"zone {z} ({m.sum()})")
ax.plot([0, 400], [0, 400], ":", color="grey")
ax.set(xlabel="Reference glucose (mg/dL)", ylabel="Forecast (mg/dL)", xlim=(40, 400), ylim=(40, 400),
       title="30-minute Ridge forecasts on a Clarke error grid"); ax.legend(fontsize=8)
plt.show()

# %% Note
# Here 1,112 forecasts fall in zone A, 25 in B, 9 in D and none in C or E. Zone A is clinically accurate and B benign;
# C, D and E would lead to unnecessary, missed or opposite treatment.
# Zone D (failing to detect a low or high) is the one to watch for forecasting lows, because it is where an alert
# should have fired. The grid weights errors by clinical consequence, which RMSE does not.

# %% Step 5
y_te, last = yw[split:], Xw[split:, -1]
new_low = (y_te < 70) & (last >= 70)
print("alert if forecast below   new lows in advance   false alerts per day")
for T in [70, 75, 80, 85, 90, 100]:
    alert = ridge_pred < T
    print(f"{T:>22}   {np.mean(alert[new_low]):>19.0%}   {np.sum(alert & (y_te >= 70)) / 4:>20.1f}")

# %% Note
# Example choice: a threshold around 80-85 mg/dL (62% of new lows warned at 6.5 false alerts a day; 88% at 9.8). Below
# that, too many lows are reached without warning (6% at 70 mg/dL); above it, false alerts rise to 11-14 a day, and alarm fatigue makes people silence alerts (which defeats the purpose). The
# right point depends on the person (hypoglycaemia unawareness argues for higher sensitivity), and the alert would also
# need a "suppress repeats" rule so one falling trend does not ring every five minutes. These are synthetic data; a
# real choice needs real-world evaluation with users.

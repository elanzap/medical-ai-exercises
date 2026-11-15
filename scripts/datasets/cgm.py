"""Dataset: Synthetic CGM Glucose Traces (synthetic-cgm-glucose-traces).

100 synthetic people, 14 days of 5-minute continuous glucose readings each, with logged meals, sensor gaps,
a people table and per-person CGM metrics. No real patients. Generalises synthetic_cgm() from the lesson
"Forecasting glucose from CGM data".

Usage (from the repo root): uv run --python 3.12 --with numpy --with pandas --with matplotlib scripts/datasets/cgm.py --user rakuelli
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_common as K  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SLUG = "synthetic-cgm-glucose-traces"
TITLE = "Synthetic CGM Glucose Traces"
SUBTITLE = "100 synthetic people, 14 days of 5-minute glucose readings, meals and gaps"
OUT = K.BUILD / "datasets" / SLUG
ASSETS = K.ROOT / "assets" / "kaggle" / SLUG
START = pd.Timestamp("2026-01-05 00:00")
DAYS, STEP = 14, 288  # 288 five-minute readings a day

GROUPS = {  # group: (share, baseline mean, meal response scale, chance of a low per day, sensor noise sd, physiological noise sd)
    "type 1, insulin": (0.30, 100, 2.5, 1.0, 6.0, 3.6),
    "type 2, insulin": (0.30, 106, 2.0, 0.45, 5.0, 2.6),
    "type 2, no insulin": (0.40, 98, 1.45, 0.05, 4.0, 1.6),
}


def simulate(person: int, group: str, rng: np.random.Generator):
    share, base_mu, meal_scale, low_p, noise, physio = GROUPS[group]
    base = rng.normal(base_mu, 12)
    dawn = rng.uniform(0, 20) if group != "type 2, no insulin" else rng.uniform(0, 10)
    steps = DAYS * STEP
    g, effect = base, 0.0
    glucose = np.empty(steps)
    meal = np.zeros(steps, dtype=int)
    for day in range(DAYS):
        times = {}
        for centre, prob in ((7.5, 0.92), (13.0, 0.95), (20.0, 0.97), (16.5, 0.35)):  # breakfast, lunch, dinner, snack
            if rng.random() < prob:
                times[int(np.clip(rng.normal(centre, 0.6), 0, 23.9) * 12)] = rng.uniform(15, 35) if centre == 16.5 else rng.uniform(30, 95)
        low_at = int((rng.uniform(1.5, 4.5) if rng.random() < 0.6 else rng.uniform(10.5, 18)) * 12) if rng.random() < low_p else None
        exercise_at = int(rng.uniform(17, 19.5) * 12) if rng.random() < 0.3 else None
        day_offset = rng.normal(0, {"type 1, insulin": 22, "type 2, insulin": 14}.get(group, 8))  # day-to-day variation
        for k in range(STEP):
            t = day * STEP + k
            hour = k / 12
            if k in times:
                carbs = times[k]
                effect += carbs * meal_scale * rng.lognormal(0, 0.35)  # meals vary, and so does the insulin dose that covers them
                if rng.random() > 0.2:  # 80% of meals are logged
                    meal[t] = round(carbs)
            if low_at is not None and k == low_at:
                effect -= rng.uniform(110, 190)
            if exercise_at is not None and exercise_at <= k < exercise_at + 6:
                effect -= rng.uniform(4, 8)
            if g < 65 and rng.random() < 0.15:  # most lows are treated with fast carbohydrate within 15-30 minutes
                effect = max(effect, 0) + rng.uniform(40, 70)
            effect *= 0.965
            target = base + day_offset + dawn * np.exp(-((hour - 6.5) ** 2) / 2)
            g += 0.03 * (target - g) + 0.04 * effect + rng.normal(0, physio)
            glucose[t] = g
    err = np.empty(steps)  # sensor error drifts slowly (autocorrelated), as real sensor error does
    e = 0.0
    for t in range(steps):
        e = 0.8 * e + rng.normal(0, noise * 0.6)
        err[t] = e
    sensor = np.clip(glucose + err, 40, 400).round()
    gap = np.zeros(steps, dtype=int)
    for _ in range(rng.integers(1, 4)):  # 1-3 signal losses of 30 minutes to 3 hours
        s = rng.integers(0, steps - 40)
        gap[s:s + rng.integers(6, 37)] = 1
    sensor = np.where(gap == 1, np.nan, sensor)
    ts = START + pd.to_timedelta(np.arange(steps) * 5, unit="min")
    return pd.DataFrame({"person_id": f"CGM-{person:03d}", "timestamp": ts, "glucose_mg_dl": sensor, "meal_carbs_g": meal, "sensor_gap": gap})


def metrics(df: pd.DataFrame) -> dict:
    g = df.glucose_mg_dl.dropna()
    m = g.mean()
    return {
        "readings_captured_percent": round(100 * len(g) / len(df), 1),
        "mean_glucose_mg_dl": round(m, 1),
        "sd_mg_dl": round(g.std(), 1),
        "cv_percent": round(100 * g.std() / m, 1),
        "gmi_percent": round(3.31 + 0.02392 * m, 2),
        "time_below_54_percent": round(100 * (g < 54).mean(), 1),
        "time_below_70_percent": round(100 * (g < 70).mean(), 1),
        "time_in_range_70_180_percent": round(100 * ((g >= 70) & (g <= 180)).mean(), 1),
        "time_above_180_percent": round(100 * (g > 180).mean(), 1),
        "time_above_250_percent": round(100 * (g > 250).mean(), 1),
    }


def generate(n=100, seed=7):
    rng = np.random.default_rng(seed)
    names = list(GROUPS)
    groups = rng.choice(names, size=n, p=[GROUPS[x][0] for x in names])
    people, traces, mets = [], [], []
    for i, grp in enumerate(groups, 1):
        age = int(np.clip(rng.normal(30, 11) if grp.startswith("type 1") else rng.normal(56, 10), 18, 85))
        tr = simulate(i, grp, rng)
        traces.append(tr)
        people.append({"person_id": f"CGM-{i:03d}", "diabetes_type": grp.split(",")[0], "therapy": "insulin" if "no insulin" not in grp else "no insulin",
                       "age_band": f"{age // 10 * 10}-{age // 10 * 10 + 9}", "sex": rng.choice(["female", "male"])})
        mets.append({"person_id": f"CGM-{i:03d}", **metrics(tr)})
    return pd.DataFrame(people), pd.concat(traces, ignore_index=True), pd.DataFrame(mets)


TABLES_DOC = {
    "cgm_readings.csv": ("Every 5-minute sensor reading for all 100 people over 14 days (403,200 rows). Readings are missing during sensor signal losses.", {
        "person_id": ("id", "Synthetic person identifier (CGM-001 to CGM-100)"),
        "timestamp": ("datetime", "Reading time, 5-minute steps from 2026-01-05 00:00 (synthetic calendar)"),
        "glucose_mg_dl": ("numeric", "Sensor glucose, mg/dL, limited to the sensor range 40-400; empty during signal loss"),
        "meal_carbs_g": ("integer", "Carbohydrate logged at this time, grams (0 if none; about 20% of meals are not logged)"),
        "sensor_gap": ("boolean", "1 if the sensor signal was lost at this time"),
    }),
    "people.csv": ("One row per synthetic person: diabetes type, therapy, age band and sex.", {
        "person_id": ("id", "Synthetic person identifier"),
        "diabetes_type": ("string", "type 1 or type 2"),
        "therapy": ("string", "insulin or no insulin"),
        "age_band": ("string", "Age band in years"),
        "sex": ("string", "female or male"),
    }),
    "cgm_metrics.csv": ("Standard CGM summary metrics per person, calculated as in the International Consensus on Time in Range (Battelino et al., 2019).", {
        "person_id": ("id", "Synthetic person identifier"),
        "readings_captured_percent": ("numeric", "Share of expected readings present"),
        "mean_glucose_mg_dl": ("numeric", "Mean sensor glucose, mg/dL"),
        "sd_mg_dl": ("numeric", "Standard deviation of glucose, mg/dL"),
        "cv_percent": ("numeric", "Coefficient of variation, % (36% or less is the consensus target)"),
        "gmi_percent": ("numeric", "Glucose management indicator, % (3.31 + 0.02392 x mean glucose in mg/dL)"),
        "time_below_54_percent": ("numeric", "Time below 54 mg/dL, %"),
        "time_below_70_percent": ("numeric", "Time below 70 mg/dL, % (includes time below 54)"),
        "time_in_range_70_180_percent": ("numeric", "Time in range 70-180 mg/dL, %"),
        "time_above_180_percent": ("numeric", "Time above 180 mg/dL, % (includes time above 250)"),
        "time_above_250_percent": ("numeric", "Time above 250 mg/dL, %"),
    }),
}


def charts(people, readings, mets):
    ASSETS.mkdir(parents=True, exist_ok=True)
    K.style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), dpi=110, gridspec_kw={"width_ratios": [1.5, 1.2, 1]})
    fig.subplots_adjust(top=0.72, bottom=0.17, wspace=0.3, left=0.05, right=0.98)

    pid = mets.sort_values("time_below_70_percent").iloc[-12].person_id
    d = readings[readings.person_id == pid].iloc[: 3 * STEP]
    hours = np.arange(len(d)) / 12
    ax[0].axhspan(70, 180, color=K.BLUE_LIGHT, alpha=0.35, lw=0)
    ax[0].plot(hours, d.glucose_mg_dl, color=K.BLUE, lw=1.6)
    low = d.glucose_mg_dl < 70
    ax[0].scatter(hours[low], d.glucose_mg_dl[low], s=10, color=K.ORANGE, zorder=3)
    ax[0].set(xlabel="Hours from the start", ylabel="Glucose (mg/dL)", xticks=range(0, 73, 12), xlim=(0, 72))
    ax[0].text(71, 73, "Target range 70-180", fontsize=8.5, color=K.INK_2, va="bottom", ha="right")
    K.titled(ax[0], f"Three days of one person's readings ({pid})", "Readings below 70 mg/dL are marked in orange")

    pr = readings[readings.person_id == pid].copy()
    pr["hour"] = pr.timestamp.dt.hour + pr.timestamp.dt.minute / 60
    q = pr.groupby("hour").glucose_mg_dl.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    ax[1].fill_between(q.index, q[0.05], q[0.95], color=K.BLUE_LIGHT, alpha=0.45, lw=0, label="5th-95th percentile")
    ax[1].fill_between(q.index, q[0.25], q[0.75], color="#86b6ef", alpha=0.7, lw=0, label="25th-75th percentile")
    ax[1].plot(q.index, q[0.5], color=K.BLUE, lw=2, label="Median")
    for y in (70, 180):
        ax[1].axhline(y, color=K.INK_2, lw=0.8)
    ax[1].set(xlabel="Time of day (hours)", ylabel="Glucose (mg/dL)", xticks=range(0, 25, 4), xlim=(0, 24))
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    K.titled(ax[1], "Ambulatory glucose profile, same person", "14 days folded onto one day")

    m = mets.merge(people, on="person_id")
    m["group"] = m.diabetes_type.str.capitalize() + ", " + m.therapy
    order = ["Type 2, no insulin", "Type 2, insulin", "Type 1, insulin"]
    g = m.groupby("group")[["time_below_70_percent", "time_in_range_70_180_percent", "time_above_180_percent"]].mean().loc[order]
    left = np.zeros(len(g))
    for col, colr, lab in (("time_below_70_percent", K.ORANGE, "Below 70"), ("time_in_range_70_180_percent", K.BLUE, "70-180"), ("time_above_180_percent", K.MUTED, "Above 180")):
        ax[2].barh(range(len(g)), g[col], left=left, height=0.5, color=colr, edgecolor=K.SURFACE, linewidth=2, label=lab)
        for i, (l, v) in enumerate(zip(left, g[col])):
            if v >= 8:
                ax[2].text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        left += g[col].values
    ax[2].set_yticks(range(len(g)), g.index)
    ax[2].grid(axis="y", visible=False)
    ax[2].set(xlim=(0, 100), xlabel="Share of the day (%)")
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    K.titled(ax[2], "Time in range by group", "Average across people in each group")

    fig.text(0.01, 0.95, "Synthetic CGM glucose traces: 100 people, 14 days", fontsize=14, color=K.INK, ha="left")
    fig.text(0.01, 0.9, "Synthetic data generated for teaching; no real patients", fontsize=9.5, color=K.INK_2, ha="left")
    K.source_note(fig, "Source: cgm_readings.csv and cgm_metrics.csv, Synthetic CGM Glucose Traces (Kaggle)")
    fig.savefig(ASSETS / "overview.png")
    plt.close(fig)

    one = readings[readings.person_id == pid].glucose_mg_dl.iloc[: STEP].interpolate().values

    def draw(axc):
        axc.axhspan(70, 180, color=K.BLUE_LIGHT, alpha=0.35, lw=0)
        axc.plot(np.arange(len(one)), one, color=K.BLUE, lw=1.6)
        axc.set_xlim(0, len(one))
        axc.set_ylim(40, max(260, one.max() + 10))

    K.cover(OUT / "dataset-cover-image.png", ("Synthetic CGM", "Glucose Traces"), "100 people, 14 days, every 5 minutes", "CONTINUOUS GLUCOSE MONITORING  |  FORECASTING", draw)


def description(user, people, readings, mets, lesson_url):
    m = mets.merge(people, on="person_id")
    summ = m.groupby(["diabetes_type", "therapy"]).agg(people=("person_id", "size"), mean_glucose=("mean_glucose_mg_dl", "mean"), tir=("time_in_range_70_180_percent", "mean"), tbr=("time_below_70_percent", "mean"), cv=("cv_percent", "mean")).round(1).reset_index()
    rows = "\n".join(f"| {r.diabetes_type}, {r.therapy} | {r.people} | {r.mean_glucose:.0f} | {r.tir:.0f}% | {r.tbr:.1f}% | {r.cv:.0f}% |" for r in summ.itertuples())
    return f"""# {TITLE}

Continuous glucose monitoring (CGM) data for practising **glucose forecasting, hypoglycaemia prediction and CGM metrics**, without any patient data: **100 synthetic people with diabetes**, each with **14 days of readings every 5 minutes** (403,200 readings), logged meals, realistic sensor noise and signal losses, plus standard per-person metrics. It accompanies the free [Medical AI course]({K.SITE}/medical-ai) by [{K.AUTHOR}]({K.SITE}/about), a physician and diabetologist.

**No real patients.** The traces come from a simple physiological simulation written for teaching (CC BY 4.0).

![Overview of the synthetic CGM traces]({K.RAW}/assets/kaggle/{SLUG}/overview.png)

## What's inside

| File | Rows | What it is |
|---|---|---|
| `cgm_readings.csv` | {len(readings):,} | Every 5-minute reading: person, time, glucose, logged carbohydrate, sensor gap flag |
| `people.csv` | {len(people)} | Diabetes type, therapy, age band and sex for each person |
| `cgm_metrics.csv` | {len(mets)} | Mean glucose, SD, CV, GMI and time below, in and above range per person |
| `data_dictionary.csv` | | Every column, with type and description |
| `generation_assumptions.md` | | How the traces were simulated, and their limits |

## The people at a glance

| Group | People | Mean glucose (mg/dL) | Time in range | Time below 70 | CV |
|---|---|---|---|---|---|
{rows}

## Ideas for what to do with it

- **Forecasting:** predict glucose 30 or 60 minutes ahead; compare persistence, linear models, gradient boosting and recurrent networks; report MAE, RMSE and clinically weighted error.
- **Hypoglycaemia prediction:** predict a reading below 70 mg/dL within the next 30 minutes; mind the class imbalance.
- **CGM metrics:** recompute time in range, GMI and CV from the raw readings and check them against `cgm_metrics.csv`.
- **Missing data:** handle the sensor signal losses (`sensor_gap`) without leaking future values.
- **Meals:** about 20% of meals are not logged; try detecting them from the trace.

A ready-to-run [starter notebook](https://www.kaggle.com/code/{user}/synthetic-cgm-glucose-traces-starter) loads the files, draws a day and an ambulatory glucose profile, and fits a 30-minute forecasting baseline.

{K.links_block([("Forecasting glucose from CGM data", lesson_url)])}
{K.related_datasets(user, SLUG)}
## Licence and citation

Data and text: CC BY 4.0. Code: MIT. Please cite as: *Ram Kumar G. Synthetic CGM Glucose Traces. Kaggle, 2026. {K.SITE}/medical-ai*

{K.DISCLAIMER}
"""


def assumptions():
    rows = "\n".join(f"| {g} | {int(v[0] * 100)}% | {v[1]} | {v[2]} | {v[3]} | {v[4]} | {v[5]} |" for g, v in GROUPS.items())
    return f"""# How the CGM traces were generated

Generated by `scripts/datasets/cgm.py` (numpy, seed 7) in {K.REPO}. It is a deliberately **simple** simulation for teaching, extending the `synthetic_cgm()` function from the lesson [Forecasting glucose from CGM data]({K.SITE}/medical-ai/applied-diabetes/cgm-glucose-forecasting). It is **not** a validated physiological model (such as the UVA/Padova simulator) and must not be used to evaluate real devices or dosing.

## Model

Every 5 minutes: glucose moves 3% of the way back towards a personal baseline (with a small dawn rise around 06:30), plus 4% of a decaying "meal effect" (it decays by 3.5% per step, so a meal's effect lasts a few hours), plus small physiological noise. Sensor noise is then added and readings are limited to 40-400 mg/dL.

| Group | Share | Baseline mean (mg/dL) | Meal response scale | Chance of a low, per day | Sensor noise SD | Physiological noise SD |
|---|---|---|---|---|---|---|
{rows}

- Meals: breakfast (about 07:30), lunch (13:00) and dinner (20:00) on most days, and an afternoon snack on about a third of days; 30-95 g carbohydrate (15-35 g for snacks); about 20% of meals are not logged.
- Lows are usually treated: below 65 mg/dL there is a 15% chance every 5 minutes of fast carbohydrate being taken.
- Lows: a sudden drop, overnight (01:30-04:30) about 60% of the time and otherwise in the late morning or afternoon; most frequent in type 1 diabetes.
- Exercise: on about 30% of days, a 30-minute dip in the early evening.
- Sensor signal loss: 1 to 3 gaps per person, each 30 minutes to 3 hours, with no readings.
- Metrics follow the International Consensus on Time in Range (Battelino et al., Diabetes Care 2019;42:1593-1603); GMI = 3.31 + 0.02392 x mean glucose (mg/dL).
"""


STARTER_CODE = [
    ("md", "## Load the data"),
    ("code", """
import pandas as pd, numpy as np, matplotlib.pyplot as plt
readings = pd.read_csv(f'{DATA}/cgm_readings.csv', parse_dates=['timestamp'])
people = pd.read_csv(f'{DATA}/people.csv')
metrics = pd.read_csv(f'{DATA}/cgm_metrics.csv')
print(readings.shape, people.shape)
metrics.merge(people).groupby(['diabetes_type', 'therapy'])[['time_in_range_70_180_percent', 'time_below_70_percent', 'cv_percent']].mean().round(1)
"""),
    ("md", "## One day of readings"),
    ("code", """
pid = 'CGM-001'
day = readings[readings.person_id == pid].iloc[:288]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axhspan(70, 180, color='#b7d3f6', alpha=0.35, lw=0)
ax.plot(np.arange(288) / 12, day.glucose_mg_dl, color='#2a78d6', lw=1.6)
ax.set(xlabel='Hour of day', ylabel='Glucose (mg/dL)', title=f'{pid}, first day')
plt.show()
"""),
    ("md", "## A 30-minute forecasting baseline\n\nPredict glucose 6 readings (30 minutes) ahead from the last 6 readings. Split by person, never by time point, so the model is tested on people it hasn't seen."),
    ("code", """
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def windows(df, lags=6, horizon=6):
    X, y = [], []
    for _, g in df.groupby('person_id'):
        v = g.glucose_mg_dl.interpolate(limit=6).values
        for t in range(lags, len(v) - horizon, 3):
            past, future = v[t - lags:t], v[t + horizon - 1]
            if not (np.isnan(past).any() or np.isnan(future)):
                X.append(past); y.append(future)
    return np.array(X), np.array(y)

ids = people.person_id.sample(frac=1, random_state=0).tolist()
train, test = readings[readings.person_id.isin(ids[:70])], readings[readings.person_id.isin(ids[70:])]
X_tr, y_tr = windows(train); X_te, y_te = windows(test)
model = Ridge().fit(X_tr, y_tr)
print('Persistence (last value) MAE:', round(mean_absolute_error(y_te, X_te[:, -1]), 1), 'mg/dL')
print('Ridge on last 30 minutes MAE:', round(mean_absolute_error(y_te, model.predict(X_te)), 1), 'mg/dL')
"""),
]


def starter(user, lesson_url):
    cells = [K.md_cell(f"""
# Synthetic CGM Glucose Traces: starter notebook

> Starter notebook for the dataset **[{TITLE}](https://www.kaggle.com/datasets/{user}/{SLUG})**, part of the free **[Medical AI course]({K.SITE}/medical-ai)** by [{K.AUTHOR}]({K.SITE}/about).
>
> - All readings are **synthetic**: no real patients.
> - The full lesson behind this data: [Forecasting glucose from CGM data]({lesson_url}).
"""), K.code_cell(K.find_data_snippet("cgm_readings.csv"))]
    for kind, src in STARTER_CODE:
        cells.append(K.md_cell(src) if kind == "md" else K.code_cell(src))
    cells.append(K.md_cell(f"""
## Continue learning

- Lesson: [Forecasting glucose from CGM data]({lesson_url})
- The whole course: [{K.SITE.removeprefix('https://')}/medical-ai]({K.SITE}/medical-ai)
- About the author: [{K.AUTHOR}]({K.SITE}/about)

---

{K.DISCLAIMER}
"""))
    K.write_starter(K.BUILD / "starters" / SLUG, user, "synthetic-cgm-glucose-traces-starter", "Synthetic CGM Glucose Traces: Starter", SLUG, cells)


def main(user):
    course = json.loads((K.ROOT / "kaggle-course.json").read_text())
    lesson_url = next(l["url"] for l in course["lessons"] if l["slug"] == "cgm-glucose-forecasting")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    people, readings, mets = generate()
    readings.to_csv(OUT / "cgm_readings.csv", index=False, date_format="%Y-%m-%d %H:%M")
    people.to_csv(OUT / "people.csv", index=False)
    mets.to_csv(OUT / "cgm_metrics.csv", index=False)
    pd.DataFrame([(f, c, t, d) for f, (_, cols) in TABLES_DOC.items() for c, (t, d) in cols.items()], columns=["file", "column", "type", "description"]).to_csv(OUT / "data_dictionary.csv", index=False)
    (OUT / "generation_assumptions.md").write_text(assumptions())
    charts(people, readings, mets)
    tables = {f: {"description": d, "columns": c} for f, (d, c) in TABLES_DOC.items()}
    tables["data_dictionary.csv"] = {"description": "Every column of every CSV in this dataset, with its type and description.", "columns": {"file": ("string", "CSV file name"), "column": ("string", "Column name"), "type": ("string", "Column type"), "description": ("string", "What the column holds")}}
    tables["generation_assumptions.md"] = {"description": "How the traces were simulated, the parameters for each group, and the limits of the simulation."}
    K.write_metadata(OUT, {
        "title": TITLE, "subtitle": SUBTITLE, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["healthcare", "medicine", "time series analysis", "education", "regression"],
        "description": description(user, people, readings, mets, lesson_url),
        "resources": K.resources(tables),
        "userSpecifiedSources": f"Generated by {K.AUTHOR} ({K.AUTHOR_DEGREE}) for the Medical AI course at {K.SITE}/medical-ai, using scripts/datasets/cgm.py (seed 7) in {K.REPO}. A simple teaching simulation of glucose dynamics; not a validated physiological model. CGM metrics follow the International Consensus on Time in Range (Battelino et al., 2019). No real patients.",
        "expectedUpdateFrequency": "annually",
        "image": "dataset-cover-image.png",
    })
    starter(user, lesson_url)
    print(mets.merge(people).groupby(["diabetes_type", "therapy"])[["time_in_range_70_180_percent", "time_below_70_percent", "cv_percent", "mean_glucose_mg_dl"]].mean().round(1).to_string())
    print(f"readings {len(readings):,}; missing {readings.glucose_mg_dl.isna().mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    main(ap.parse_args().user)

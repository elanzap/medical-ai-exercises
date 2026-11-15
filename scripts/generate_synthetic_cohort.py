"""Generates diabetes_clinic_cohort.csv: 5,000 SYNTHETIC adults attending a diabetes clinic.

No real patients. Values are plausible and correlated, but the cohort is NOT epidemiologically
calibrated to any population; use it to practise methods, never to draw clinical conclusions.
Every assumption is written below and in cohort_generation_assumptions.md.

Usage: python scripts/generate_synthetic_cohort.py <output.csv> [--n 5000] [--seed 42]
Needs numpy and pandas.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# True coefficients of the logistic model that generates admission_12m (log-odds scale).
# Learners can fit a model and compare their estimates with these.
ADMISSION_MODEL = {
    "intercept": -2.3,
    "age_per_10y_over_55": 0.25,
    "hba1c_per_1pct_over_7.5": 0.30,
    "egfr_per_10_below_90": 0.15,
    "on_insulin": 0.50,
    "retinopathy": 0.35,
    "neuropathy": 0.30,
    "foot_ulcer_history": 0.80,
    "smoker": 0.30,
    "log_uacr_over_30": 0.15,
}


def reflect(x, low, high):
    """Folds values back inside [low, high] instead of clipping, so no artificial pile-up at the bounds."""
    x = np.where(x < low, 2 * low - x, x)
    x = np.where(x > high, 2 * high - x, x)
    return np.clip(x, low, high)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t1 = rng.random(n) < 0.06
    dtype = np.where(t1, "type 1", "type 2")
    male = rng.random(n) < 0.52
    age = np.where(t1, rng.normal(30, 10, n), rng.normal(54, 11, n))
    age = np.clip(age, 18, 88).round()
    duration = np.where(t1, rng.uniform(0, 1, n) * (age - 10), rng.gamma(2.0, 4.0, n))
    duration = np.clip(np.minimum(duration, age - 12), 0, 45).round()

    bmi = np.where(t1, rng.normal(23, 3.5, n), rng.normal(28, 4.5, n))
    bmi = reflect(bmi, 16, 50).round(1)
    waist = np.clip(2.6 * bmi + 22 + 4 * male + rng.normal(0, 6, n), 60, 160).round()
    smoker = rng.random(n) < np.where(male, 0.18, 0.02)

    htn_p = sigmoid(-0.5 + 0.06 * (age - 50) + 0.12 * (bmi - 25) + 0.03 * duration - 1.0 * t1)
    hypertension = rng.random(n) < htn_p
    sbp = np.clip(121 + 0.45 * (age - 50) + 0.8 * (bmi - 25) + 10 * hypertension + rng.normal(0, 12, n), 90, 210).round()
    dbp = np.clip(0.45 * sbp + 20 + rng.normal(0, 7, n), 50, 125).round()

    hba1c = np.where(
        t1,
        7.9 + rng.normal(0, 1.3, n),
        6.5 + 0.07 * duration + 0.04 * (bmi - 25) + rng.gamma(2.0, 0.45, n) - 0.9 + rng.normal(0, 0.8, n),
    )
    hba1c = reflect(hba1c, 5.0, 14.5).round(1)
    fasting = np.clip(28.7 * hba1c - 46.7 + rng.normal(0, 22, n), 60, 450).round()

    statin = rng.random(n) < np.where((age >= 40) & ~t1, 0.68, 0.22)
    ldl = np.clip(rng.normal(112, 32, n) - 28 * statin, 30, 260).round()
    hdl = np.clip(np.where(male, rng.normal(40, 9, n), rng.normal(47, 10, n)), 20, 100).round()
    tg = np.clip(np.exp(np.log(130) + 0.03 * (bmi - 25) + 0.08 * (hba1c - 7) + rng.normal(0, 0.4, n)), 40, 1500).round()

    egfr = 110 - 1.0 * (age - 30) - 0.4 * duration - 0.3 * duration * (hba1c > 8) - 6 * hypertension + rng.normal(0, 15, n)
    egfr = reflect(egfr, 8, 130).round()
    log_uacr = np.log(9) + 0.28 * (hba1c - 7) + 0.02 * (sbp - 130) + 0.04 * duration + 0.012 * (90 - egfr) + rng.normal(0, 1.0, n)
    uacr = np.clip(np.exp(log_uacr), 1, 3000).round(1)

    metformin = (~t1) & (egfr >= 30) & (rng.random(n) < 0.86)
    sglt2i = (~t1) & (egfr >= 20) & (rng.random(n) < np.where((egfr < 60) | (uacr >= 30), 0.55, 0.25))
    glp1ra = (~t1) & (rng.random(n) < np.where(bmi >= 30, 0.28, 0.08))
    insulin = t1 | (rng.random(n) < sigmoid(-3.0 + 0.12 * duration + 0.45 * (hba1c - 8)))

    retinopathy = rng.random(n) < sigmoid(-3.2 + 0.12 * duration + 0.35 * (hba1c - 7) + 0.02 * (sbp - 130))
    neuropathy = rng.random(n) < sigmoid(-2.3 + 0.10 * duration + 0.30 * (hba1c - 7) + 0.03 * (age - 50))
    foot_ulcer = rng.random(n) < sigmoid(-4.6 + 1.5 * neuropathy + 0.05 * duration + 0.6 * smoker)

    m = ADMISSION_MODEL
    logit = (
        m["intercept"]
        + m["age_per_10y_over_55"] * (age - 55) / 10
        + m["hba1c_per_1pct_over_7.5"] * (hba1c - 7.5)
        + m["egfr_per_10_below_90"] * np.maximum(90 - egfr, 0) / 10
        + m["on_insulin"] * insulin
        + m["retinopathy"] * retinopathy
        + m["neuropathy"] * neuropathy
        + m["foot_ulcer_history"] * foot_ulcer
        + m["smoker"] * smoker
        + m["log_uacr_over_30"] * np.log(uacr / 30)
    )
    admission = rng.random(n) < sigmoid(logit)

    df = pd.DataFrame({
        "patient_id": [f"SYN-{i:05d}" for i in range(1, n + 1)],
        "age": age.astype(int),
        "sex": np.where(male, "male", "female"),
        "diabetes_type": dtype,
        "diabetes_duration_years": duration.astype(int),
        "bmi": bmi,
        "waist_cm": waist,
        "systolic_bp": sbp.astype(int),
        "diastolic_bp": dbp.astype(int),
        "hba1c_percent": hba1c,
        "fasting_glucose_mg_dl": fasting.astype(int),
        "ldl_mg_dl": ldl,
        "hdl_mg_dl": hdl,
        "triglycerides_mg_dl": tg,
        "egfr": egfr.astype(int),
        "uacr_mg_g": uacr,
        "smoker": smoker,
        "hypertension": hypertension,
        "on_metformin": metformin,
        "on_sglt2i": sglt2i,
        "on_glp1ra": glp1ra,
        "on_insulin": insulin,
        "on_statin": statin,
        "retinopathy": retinopathy,
        "neuropathy": neuropathy,
        "foot_ulcer_history": foot_ulcer,
        "admission_12m": admission,
    })

    # Missing values, for the missing-data lesson: UACR more often missing in younger patients with
    # short duration (missing at random), lipids and waist missing completely at random.
    p_uacr = sigmoid(-1.8 - 0.04 * (age - 50) - 0.08 * duration)
    df.loc[rng.random(n) < p_uacr * 0.6, "uacr_mg_g"] = np.nan
    for col, rate in (("ldl_mg_dl", 0.04), ("hdl_mg_dl", 0.04), ("triglycerides_mg_dl", 0.04), ("waist_cm", 0.06), ("hba1c_percent", 0.02)):
        df.loc[rng.random(n) < rate, col] = np.nan
    for col in ("waist_cm", "ldl_mg_dl", "hdl_mg_dl", "triglycerides_mg_dl"):
        df[col] = df[col].astype("Int64")
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.n, a.seed)
    df.to_csv(a.out, index=False)
    print(df.describe(include="all").T[["count", "mean", "min", "max"]].to_string())
    print("admission rate", round(df.admission_12m.mean(), 3))

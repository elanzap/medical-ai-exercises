"""Builds the Kaggle dataset "Synthetic Diabetes Clinic Data for Medical AI" into kaggle-build/dataset/,
its charts into assets/kaggle/, and its starter notebook into kaggle-build/starter/.

Run with numpy, pandas and matplotlib available, e.g.
  uv run --python 3.12 --with numpy --with pandas --with matplotlib scripts/kaggle_dataset.py --user rakuelli
Needs kaggle-course.json (from `npm run kaggle-export` in the website repo) for lesson links.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_synthetic_cohort import ADMISSION_MODEL, generate  # noqa: E402
import kaggle_common as K  # noqa: E402

OUT = ROOT / "kaggle-build" / "dataset"
STARTER = ROOT / "kaggle-build" / "starter"
ASSETS = ROOT / "assets" / "kaggle"
REPO = "https://github.com/elanzap/medical-ai-exercises"
RAW = "https://raw.githubusercontent.com/elanzap/medical-ai-exercises/main"
SLUG = "medical-ai-course-synthetic-exercise-data"
TITLE = "Synthetic Diabetes Clinic Data for Medical AI"
SUBTITLE = "5,000 synthetic patients plus clinical notes, LLM pairs and AI-regulation cases"
NAVY, TEAL, CYAN, INK, LIGHT, BLOOD = "#0b2533", "#087d7b", "#80e3d1", "#153e46", "#f2f9f6", "#ad1837"
DISCLAIMER = (
    "**Educational use only.** Every record is synthetic: no real patients, clinicians or products. The cohort is "
    "plausible but not calibrated to any population, so it must not be used to draw clinical conclusions or to build "
    "tools for real patient care. Mentions of medicines, models or regulators are for identification, not endorsement."
)


def load_script(rel: str) -> dict:
    path = ROOT / "exercise-data" / rel
    scope: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), scope)  # our own synthetic literals, no imports
    return scope


def lesson_url(course: dict, slug: str) -> str:
    for l in course["lessons"]:
        if l["slug"] == slug:
            return l["url"]
    raise KeyError(slug)


def build_tables(course: dict) -> dict[str, dict]:
    """Returns {filename: {"df", "description", "columns": {name: (type, description)}}}."""
    t: dict[str, dict] = {}
    site = course["site"]

    cohort = generate(5000, 42)
    t["diabetes_clinic_cohort.csv"] = {
        "df": cohort,
        "description": "5,000 synthetic adults attending a diabetes clinic: demographics, vital signs, labs, medicines, complications and a 12-month admission label generated from a known logistic model (see cohort_generation_assumptions.md). Some lab values are deliberately missing. Use for risk prediction, calibration, missing-data and fairness exercises.",
        "columns": {
            "patient_id": ("id", "Synthetic patient identifier (SYN-00001 to SYN-05000)"),
            "age": ("integer", "Age in years"),
            "sex": ("string", "male or female"),
            "diabetes_type": ("string", "type 1 or type 2"),
            "diabetes_duration_years": ("integer", "Years since diabetes diagnosis"),
            "bmi": ("numeric", "Body mass index, kg/m2"),
            "waist_cm": ("integer", "Waist circumference in cm (about 6% missing completely at random)"),
            "systolic_bp": ("integer", "Systolic blood pressure, mmHg"),
            "diastolic_bp": ("integer", "Diastolic blood pressure, mmHg"),
            "hba1c_percent": ("numeric", "HbA1c, % (about 2% missing)"),
            "fasting_glucose_mg_dl": ("integer", "Fasting plasma glucose, mg/dL"),
            "ldl_mg_dl": ("integer", "LDL cholesterol, mg/dL (about 4% missing)"),
            "hdl_mg_dl": ("integer", "HDL cholesterol, mg/dL (about 4% missing)"),
            "triglycerides_mg_dl": ("integer", "Triglycerides, mg/dL (about 4% missing)"),
            "egfr": ("integer", "Estimated glomerular filtration rate, mL/min/1.73m2"),
            "uacr_mg_g": ("numeric", "Urine albumin-to-creatinine ratio, mg/g (missing at random, more often in younger patients with short duration)"),
            "smoker": ("boolean", "Current smoker (1 = yes)"),
            "hypertension": ("boolean", "Diagnosed hypertension (1 = yes)"),
            "on_metformin": ("boolean", "Taking metformin (1 = yes)"),
            "on_sglt2i": ("boolean", "Taking an SGLT2 inhibitor (1 = yes)"),
            "on_glp1ra": ("boolean", "Taking a GLP-1 receptor agonist (1 = yes)"),
            "on_insulin": ("boolean", "Taking insulin (1 = yes; all type 1)"),
            "on_statin": ("boolean", "Taking a statin (1 = yes)"),
            "retinopathy": ("boolean", "Diabetic retinopathy recorded (1 = yes)"),
            "neuropathy": ("boolean", "Diabetic peripheral neuropathy recorded (1 = yes)"),
            "foot_ulcer_history": ("boolean", "History of diabetic foot ulcer (1 = yes)"),
            "admission_12m": ("boolean", "Target: hospital admission in the next 12 months (1 = yes), drawn from the documented logistic model"),
        },
    }

    notes = load_script("clinical-nlp/01-working-with-clinical-notes.py")["clinic_notes"]
    t["clinical_notes.csv"] = {
        "df": pd.DataFrame(notes, columns=["patient_id", "visit_number", "note_text"]),
        "description": f"20 synthetic diabetes clinic notes: 5 made-up patients, 4 visits each, in date order, with abbreviations and copy-forward text. Lesson: {lesson_url(course, 'working-with-clinical-notes')}",
        "columns": {
            "patient_id": ("id", "Synthetic patient (P1 to P5)"),
            "visit_number": ("integer", "Visit order for that patient (1 to 4)"),
            "note_text": ("string", "Free-text clinic note (HISTORY, MEDICATIONS, EXAMINATION, ASSESSMENT, PLAN)"),
        },
    }

    coded = load_script("clinical-nlp/02-clinical-ner-and-coding.py")["CODED_NOTES"]
    t["coded_notes.csv"] = {
        "df": pd.DataFrame([(n, ";".join(c)) for n, c in coded], columns=["note_text", "icd10_codes"]),
        "description": f"30 short synthetic note snippets with the ICD-10 codes a coder would assign, for named-entity recognition and automated coding. Lesson: {lesson_url(course, 'clinical-ner-and-coding')}",
        "columns": {
            "note_text": ("string", "Short clinical note snippet"),
            "icd10_codes": ("string", "Expected ICD-10 codes, separated by semicolons (empty if none)"),
        },
    }

    deid = load_script("clinical-nlp/04-de-identifying-clinical-text.py")["DEID_NOTES"]
    kinds = sorted({k for _, _, d in deid for k in d})
    rows = [{"note_id": n, "patient_id": p, "note_text": text, **{f"phi_{k.lower()}": ";".join(d.get(k, [])) for k in kinds}} for n, (p, text, d) in enumerate(deid, 1)]
    cols = {"note_id": ("id", "Note identifier"), "patient_id": ("string", "Synthetic patient letter; notes from the same patient share it (for consistent surrogates)"), "note_text": ("string", "Synthetic note containing made-up identifiers")}
    cols.update({f"phi_{k.lower()}": ("string", f"Gold-standard {k.lower()} identifiers in the note, separated by semicolons") for k in kinds})
    t["deid_notes.csv"] = {
        "df": pd.DataFrame(rows),
        "description": f"15 synthetic notes containing made-up names, IDs, dates and places, with gold-standard identifier spans for evaluating de-identification. Lesson: {lesson_url(course, 'de-identifying-clinical-text')}",
        "columns": cols,
    }

    p = load_script("generative-ai-llms/02-prompting-for-clinical-tasks.py")
    t["prompting_notes.csv"] = {
        "df": pd.DataFrame({
            "note_id": range(1, len(p["notes"]) + 1),
            "note_text": p["notes"],
            "gold_diabetes_type": [x["diabetes_type"] for x in p["labels"]],
            "gold_latest_hba1c_percent": [x["latest_hba1c_percent"] for x in p["labels"]],
            "gold_on_insulin": [int(bool(x["on_insulin"])) if x["on_insulin"] is not None else None for x in p["labels"]],
            "recorded_output_v1": p["recorded_v1"],
            "recorded_output_v2": p["recorded_v2"],
        }),
        "description": f"20 synthetic notes with gold-standard extraction labels and two recorded LLM outputs (prompt versions 1 and 2), for evaluating structured extraction without calling a live model. Lesson: {lesson_url(course, 'prompting-for-clinical-tasks')}",
        "columns": {
            "note_id": ("id", "Note identifier"),
            "note_text": ("string", "Short synthetic diabetes note"),
            "gold_diabetes_type": ("string", "Gold label: type of diabetes"),
            "gold_latest_hba1c_percent": ("numeric", "Gold label: most recent HbA1c, % (empty if not stated)"),
            "gold_on_insulin": ("boolean", "Gold label: on insulin (1 = yes)"),
            "recorded_output_v1": ("string", "JSON output recorded from prompt version 1"),
            "recorded_output_v2": ("string", "JSON output recorded from prompt version 2"),
        },
    }

    cases = load_script("generative-ai-llms/04-hallucination-and-grounding.py")["cases"]
    t["hallucination_cases.csv"] = {
        "df": pd.DataFrame([{
            "case_id": i + 1,
            "transcript": c["transcript"],
            "draft_summary": "\n".join(c["draft"]),
            "unsupported_sentences": json.dumps(c["errors"]),
            "omissions": json.dumps(c["omissions"]),
            "n_unsupported": len(c["errors"]),
            "n_omissions": len(c["omissions"]),
        } for i, c in enumerate(cases)]),
        "description": f"10 synthetic consultation transcripts with AI-style draft summaries, annotated for unsupported (hallucinated) sentences and omissions with severity. Lesson: {lesson_url(course, 'hallucination-and-grounding')}",
        "columns": {
            "case_id": ("id", "Case identifier"),
            "transcript": ("string", "Synthetic doctor-patient conversation"),
            "draft_summary": ("string", "Draft summary, one sentence per line"),
            "unsupported_sentences": ("string", "JSON list of [draft sentence number, error type, severity] for sentences the transcript does not support"),
            "omissions": ("string", "JSON list of [missed fact, severity] pairs"),
            "n_unsupported": ("integer", "Number of unsupported sentences"),
            "n_omissions": ("integer", "Number of omissions"),
        },
    }

    lora = load_script("generative-ai-llms/05-fine-tuning-lora.py")
    pairs = [(n, s, "train") for n, s in lora["train_pairs"]] + [(n, s, "held_out") for n, s in lora["held_out_pairs"]]
    t["note_summary_pairs.csv"] = {
        "df": pd.DataFrame(pairs, columns=["note_text", "summary", "split"]),
        "description": f"16 synthetic note-to-summary pairs with a train/held-out split, for fine-tuning a small model (for example with LoRA). Lesson: {lesson_url(course, 'fine-tuning-lora')}",
        "columns": {
            "note_text": ("string", "Synthetic clinical note"),
            "summary": ("string", "Target one-line summary"),
            "split": ("string", "train or held_out"),
        },
    }

    amb = load_script("generative-ai-llms/07-summarisation-ambient-documentation.py")["pairs"]
    t["ambient_documentation_pairs.csv"] = {
        "df": pd.DataFrame([{
            "pair_id": i + 1,
            "transcript": a["transcript"],
            "ai_draft_note": "\n".join(a["draft"]),
            "clinician_final_note": "\n".join(x for x in a["final"] if x),
            "edit_labels": json.dumps([x for x in a["labels"] if x]),
            "n_edits": sum(1 for x in a["labels"] if x),
        } for i, a in enumerate(amb)]),
        "description": f"20 synthetic ambient-scribe examples: a consultation summary, the AI draft note and the clinician's edited final note, with each edit labelled (style, factual correction, added safety information, removed fabricated content). Lesson: {lesson_url(course, 'summarisation-ambient-documentation')}",
        "columns": {
            "pair_id": ("id", "Example identifier"),
            "transcript": ("string", "Synthetic consultation summary"),
            "ai_draft_note": ("string", "AI-drafted note, one sentence per line"),
            "clinician_final_note": ("string", "Clinician-edited final note, one sentence per line"),
            "edit_labels": ("string", "JSON list of edit types made by the clinician"),
            "n_edits": ("integer", "Number of edits"),
        },
    }

    rep = load_script("evaluation-safety/03-reporting-guidelines.py")
    t["tripod_ai_items.csv"] = {
        "df": pd.DataFrame({"item_number": range(1, len(rep["TRIPOD_AI_EXTRACT"]) + 1), "checklist_item": rep["TRIPOD_AI_EXTRACT"]}),
        "description": f"A short extract of reporting items (paraphrased from TRIPOD+AI) used to check the fictional study in fictional_study.txt. Lesson: {lesson_url(course, 'reporting-guidelines')}",
        "columns": {"item_number": ("integer", "Item number in this extract"), "checklist_item": ("string", "Reporting item, paraphrased")},
    }

    reg = []
    for prod in load_script("regulation-ethics/01-software-as-a-medical-device.py")["products"]:
        reg.append({"lesson": "software-as-a-medical-device", "jurisdiction": "general (IMDRF)", "name": prod["name"], "description": prod["claims"]})
    fda = load_script("regulation-ethics/02-us-fda-ai-devices.py")
    for tool in fda["cds_tools"]:
        reg.append({"lesson": "us-fda-ai-devices", "jurisdiction": "United States", "name": tool["name"], "description": tool["description"]})
    reg.append({"lesson": "us-fda-ai-devices", "jurisdiction": "United States", "name": fda["retinopathy_model"].split(":")[0], "description": fda["retinopathy_model"]})
    for prod in load_script("regulation-ethics/03-eu-ai-act.py")["eu_products"]:
        reg.append({"lesson": "eu-ai-act", "jurisdiction": "European Union", "name": prod["name"], "purpose": prod["purpose"], "description": prod["description"],
                    "worst_consequence": prod.get("worst_consequence"), "vital_immediate_danger": prod.get("vital_immediate_danger"), "device_class": prod.get("device_class")})
    uk = load_script("regulation-ethics/04-uk-mhra-ai.py")
    reg.append({"lesson": "uk-mhra-ai", "jurisdiction": "United Kingdom", "name": uk["uk_product"].split(".")[0], "description": uk["uk_product"]})
    for tool in load_script("regulation-ethics/05-india-cdsco-ai.py")["india_tools"]:
        reg.append({"lesson": "india-cdsco-ai", "jurisdiction": "India", "name": tool["name"], "made_in": tool["made_in"], "description": tool["description"]})
    who = load_script("regulation-ethics/06-who-guidance-ethics.py")["proposal"]
    reg.append({"lesson": "who-guidance-ethics", "jurisdiction": "international (WHO)", "name": who.split(":")[1].split(" triage")[0].strip(" '") if ":" in who else "Proposal", "description": who.strip()})
    regdf = pd.DataFrame(reg).reindex(columns=["lesson", "jurisdiction", "name", "purpose", "made_in", "description", "worst_consequence", "vital_immediate_danger", "device_class"])
    regdf.insert(0, "case_id", range(1, len(regdf) + 1))
    t["regulation_cases.csv"] = {
        "df": regdf,
        "description": f"{len(regdf)} fictional AI products and proposals used to practise regulatory classification (SaMD, US FDA, EU AI Act, UK MHRA, India CDSCO) and ethics review (WHO guidance). All names are made up. Track: {site}/medical-ai/regulation-ethics",
        "columns": {
            "case_id": ("id", "Case identifier"),
            "lesson": ("string", "Course lesson slug the case belongs to"),
            "jurisdiction": ("string", "Regulatory setting the case is framed in"),
            "name": ("string", "Fictional product or proposal name"),
            "purpose": ("string", "Intended purpose (EU AI Act cases)"),
            "made_in": ("string", "Where the product is made (India CDSCO cases)"),
            "description": ("string", "What the product claims or does"),
            "worst_consequence": ("string", "Worst plausible consequence of an error (some EU AI Act cases)"),
            "vital_immediate_danger": ("string", "Whether an error could put life in immediate danger (some EU AI Act cases)"),
            "device_class": ("string", "Medical-device class already assigned (where stated)"),
        },
    }

    sc = load_script("regulation-ethics/07-liability-and-consent.py")["scenarios"]
    t["liability_scenarios.csv"] = {
        "df": pd.DataFrame([{"scenario_id": k.split(".")[0], "title": k.split(". ", 1)[1], "scenario_text": v.strip()} for k, v in sc.items()]),
        "description": f"3 fictional case scenarios about liability and consent when AI is used in care. Lesson: {lesson_url(course, 'liability-and-consent')}",
        "columns": {"scenario_id": ("id", "Scenario letter"), "title": ("string", "Short title"), "scenario_text": ("string", "Scenario description")},
    }
    return t


def make_charts(cohort: pd.DataFrame):
    ASSETS.mkdir(parents=True, exist_ok=True)
    K.style()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), dpi=110)
    fig.subplots_adjust(hspace=0.6, wspace=0.28, top=0.81, bottom=0.08)

    bins = np.arange(4.99, 13.2, 0.3)
    t2 = cohort.loc[cohort.diabetes_type == "type 2", "hba1c_percent"].dropna()
    t1 = cohort.loc[cohort.diabetes_type == "type 1", "hba1c_percent"].dropna()
    ax[0, 0].hist(t2, bins=bins, density=True, color=K.BLUE, rwidth=0.88, label=f"Type 2 (n = {len(t2):,})")
    ax[0, 0].hist(t1, bins=bins, density=True, histtype="step", color=K.ORANGE, lw=2, label=f"Type 1 (n = {len(t1):,})")
    ax[0, 0].axvline(7, color=K.INK_2, lw=1)
    ax[0, 0].annotate("7% target", xy=(7, 0.97), xycoords=("data", "axes fraction"), xytext=(-4, 0), textcoords="offset points", fontsize=8.5, color=K.INK_2, va="top", ha="right")
    ax[0, 0].legend(loc="upper right")
    ax[0, 0].set(xlabel="HbA1c (%)", ylabel="Share of patients")
    K.titled(ax[0, 0], "HbA1c by diabetes type", "Type 1 sits higher on average")

    n_comp = cohort[["retinopathy", "neuropathy", "foot_ulcer_history"]].sum(axis=1).clip(upper=2)
    rate = cohort.groupby(n_comp)["admission_12m"].mean() * 100
    colors = [K.BLUE, K.BLUE, K.ORANGE]
    ax[0, 1].bar(["None", "One", "Two or more"], rate.values, width=0.5, color=colors)
    for i, v in enumerate(rate.values):
        ax[0, 1].text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9.5, color=K.INK)
    ax[0, 1].set(ylabel="Admitted within 12 months (%)", xlabel="Complications (retinopathy, neuropathy, foot ulcer)")
    K.titled(ax[0, 1], "Admission rate by number of complications", "Risk climbs with each complication")

    s = cohort.sample(1500, random_state=1)
    alb = s.uacr_mg_g >= 30
    ax[1, 0].scatter(s.age[~alb], s.egfr[~alb], s=9, color=K.MUTED, alpha=0.45, lw=0, label="UACR below 30 mg/g")
    ax[1, 0].scatter(s.age[alb], s.egfr[alb], s=9, color=K.ORANGE, alpha=0.75, lw=0, label="UACR 30 mg/g or more")
    ax[1, 0].axhline(60, color=K.INK_2, lw=1)
    ax[1, 0].text(17, 56, "eGFR 60", fontsize=8.5, color=K.INK_2, ha="left", va="top")
    ax[1, 0].legend(loc="upper right", markerscale=1.8)
    ax[1, 0].set(xlabel="Age (years)", ylabel="eGFR (mL/min/1.73m²)")
    K.titled(ax[1, 0], "Kidney function by age", "Albuminuria clusters where eGFR is lower")

    miss = cohort.isna().mean().mul(100).loc[lambda x: x > 0].sort_values()
    ax[1, 1].barh(miss.index, miss.values, height=0.5, color=K.BLUE)
    ax[1, 1].grid(axis="y", visible=False)
    ax[1, 1].grid(axis="x", visible=True)
    for i, v in enumerate(miss.values):
        ax[1, 1].text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9, color=K.INK)
    ax[1, 1].set(xlabel="Missing (%)")
    K.titled(ax[1, 1], "Deliberately missing values", "UACR is missing at random; the rest completely at random")

    fig.text(0.01, 0.965, "Synthetic diabetes clinic cohort, 5,000 patients", fontsize=14, color=K.INK, ha="left")
    fig.text(0.01, 0.93, "Synthetic data generated for teaching; no real patients", fontsize=9.5, color=K.INK_2, ha="left")
    K.source_note(fig, "Source: diabetes_clinic_cohort.csv, Synthetic Diabetes Clinic Data for Medical AI (Kaggle)")
    fig.savefig(ASSETS / "cohort-overview.png")
    plt.close(fig)

    vals = cohort.hba1c_percent.dropna()
    counts, _ = np.histogram(vals, bins=np.arange(4.99, 12.5, 0.2))

    def draw(axc):
        axc.bar(range(len(counts)), counts, width=0.8, color=K.BLUE_LIGHT)
        axc.set_xlim(-1, len(counts))
        axc.set_ylim(0, counts.max() * 1.05)

    K.cover(OUT / "dataset-cover-image.png", ("Synthetic Diabetes", "Clinic Data"), "for Medical AI", "5,000 SYNTHETIC PATIENTS  |  NOTES  |  LLM PAIRS", draw)


def write_assumptions():
    m = ADMISSION_MODEL
    rows = "\n".join(f"| {k} | {v:+.2f} |" for k, v in m.items())
    (OUT / "cohort_generation_assumptions.md").write_text(f"""# How diabetes_clinic_cohort.csv was generated

The cohort is **synthetic**: generated by `scripts/generate_synthetic_cohort.py` (numpy, seed 42) in {REPO}. Values are plausible and correlated, but **not calibrated to any real population**. Use it to practise methods, not to estimate prevalence or effects.

## Main assumptions

- 6% type 1, 94% type 2 diabetes; 52% male. Age about 54 ± 11 years (type 2) or 30 ± 10 (type 1), limited to 18–88.
- Diabetes duration: gamma-distributed for type 2 (mean about 8 years), never more than age minus 12.
- BMI about 28 ± 4.5 (type 2) or 23 ± 3.5 (type 1); waist rises with BMI and is larger in men.
- Smoking: 18% of men, 2% of women. Hypertension is more likely with age, BMI and duration (about 55% overall).
- HbA1c rises with duration and BMI, with a right skew; fasting glucose follows HbA1c (about 28.7 × HbA1c − 46.7, plus noise).
- Statins are more common from age 40 in type 2 diabetes and lower LDL by about 28 mg/dL on average.
- eGFR falls with age, duration (faster when HbA1c is above 8%) and hypertension; UACR rises with HbA1c, blood pressure, duration and lower eGFR.
- Medicines: metformin for most people with type 2 diabetes and eGFR of 30 or more; SGLT2 inhibitors more often with eGFR below 60 or albuminuria; GLP-1 receptor agonists more often with BMI of 30 or more; insulin for everyone with type 1 and more often with longer duration and higher HbA1c.
- Retinopathy, neuropathy and foot-ulcer history are drawn from logistic models of duration, HbA1c, blood pressure, age, neuropathy and smoking.

## The admission_12m label

`admission_12m` is drawn from a Bernoulli distribution whose log-odds are this linear model (so learners can compare fitted coefficients with the true values):

| Term | Coefficient (log-odds) |
|---|---|
{rows}

Terms: age_per_10y_over_55 = (age − 55) / 10; hba1c_per_1pct_over_7.5 = HbA1c − 7.5; egfr_per_10_below_90 = max(90 − eGFR, 0) / 10; log_uacr_over_30 = ln(UACR / 30); the rest are 0/1 indicators. The label is generated **before** missing values are added.

## Missing values

UACR is missing at random (more often in younger patients with short duration, about 6%); waist (6%), LDL, HDL, triglycerides (about 4% each) and HbA1c (2%) are missing completely at random.
""")


def description(tables: dict, course: dict) -> str:
    site = course["site"]
    inside = "\n".join(f"| `{name}` | {len(v['df']):,} | {v['description'].split('. ')[0].rstrip('.')} |" for name, v in tables.items())
    cohort = tables["diabetes_clinic_cohort.csv"]["df"]
    summ = [
        ("Age (years)", cohort.age), ("BMI (kg/m²)", cohort.bmi), ("HbA1c (%)", cohort.hba1c_percent),
        ("Systolic BP (mmHg)", cohort.systolic_bp), ("LDL (mg/dL)", cohort.ldl_mg_dl.astype(float)), ("eGFR", cohort.egfr),
    ]
    summary = "\n".join(f"| {n} | {s.mean():.1f} | {s.median():.1f} | {s.min():g}–{s.max():g} |" for n, s in summ)
    rates = "\n".join(f"| {n} | {cohort[c].mean() * 100:.1f}% |" for n, c in (
        ("Hypertension", "hypertension"), ("On insulin", "on_insulin"), ("Retinopathy", "retinopathy"),
        ("Neuropathy", "neuropathy"), ("Foot-ulcer history", "foot_ulcer_history"), ("Admitted within 12 months (target)", "admission_12m")))
    return f"""# {TITLE}

Synthetic, privacy-safe data for learning healthcare AI: a **5,000-patient diabetes clinic cohort** with a known ground-truth risk model, plus small annotated text sets for **clinical NLP, de-identification, LLM extraction, hallucination checking and ambient-scribe evaluation**, and fictional cases for **AI regulation and ethics**. It accompanies the free [Medical AI course]({site}/medical-ai) by [Dr. Ram Kumar G]({site}/about), a physician and diabetologist.

**No real patients.** Everything was written or generated for teaching, so you can share, fork and publish results freely (CC BY 4.0).

![Overview of the synthetic cohort]({RAW}/assets/kaggle/cohort-overview.png)

## What's inside

| File | Rows | What it is |
|---|---|---|
{inside}
| `data_dictionary.csv` | {sum(len(v['columns']) for v in tables.values())} | Every column in every file, with type and description |
| `cohort_generation_assumptions.md` | | How the cohort was generated, including the true admission model |
| `fictional_study.txt` | | A fictional prediction-model paper to check against `tripod_ai_items.csv` |
| `source/` | | The original Python data files used in the course notebooks |

## The cohort at a glance

| Measure | Mean | Median | Range |
|---|---|---|---|
{summary}

| Characteristic | Share of patients |
|---|---|
{rates}

## Ideas for what to do with it

- **Risk prediction:** predict `admission_12m`; compare logistic regression, gradient boosting and calibration, then check your coefficients against the true model in `cohort_generation_assumptions.md`.
- **Missing data:** UACR is missing at random and some labs completely at random; compare complete-case analysis with imputation.
- **Fairness:** compare model performance by sex and diabetes type.
- **Clinical NLP:** extract medicines and diagnoses from `clinical_notes.csv`; assign ICD-10 codes in `coded_notes.csv`.
- **De-identification:** score a de-identifier against the gold spans in `deid_notes.csv`.
- **LLM evaluation:** measure extraction accuracy (`prompting_notes.csv`), find hallucinations and omissions (`hallucination_cases.csv`) and classify scribe edits (`ambient_documentation_pairs.csv`).
- **Regulation:** classify the fictional products in `regulation_cases.csv` under SaMD, FDA, EU AI Act, MHRA and CDSCO frameworks.

A ready-to-run [starter notebook](https://www.kaggle.com/code/{course.get('kaggleUser', 'rakuelli')}/synthetic-diabetes-clinic-data-starter) loads every file, draws the charts above and fits a baseline model.

## Learn the methods

Each table belongs to a lesson with a full explanation, worked example, exercise and solution:

- Course home and all ten tracks: {site}/medical-ai
- Start here: {site}/medical-ai/start-here
- Exercises and solutions: {site}/medical-ai/exercises
- Notebooks on GitHub (MIT): {REPO}
- About the author: {site}/about

{K.related_datasets(course.get('kaggleUser', 'rakuelli'), SLUG)}
## Licence and citation

Data and text: CC BY 4.0. Code: MIT. Please cite as: *Ram Kumar G. Synthetic Diabetes Clinic Data for Medical AI. Kaggle, 2026. {site}/medical-ai*

{DISCLAIMER}
"""


def write_starter(user: str, course: dict):
    site = course["site"]
    STARTER.mkdir(parents=True, exist_ok=True)
    cells = []

    def mdc(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)})

    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(keepends=True)})

    mdc(f"""
# Synthetic Diabetes Clinic Data: starter notebook

> Starter notebook for the dataset **[{TITLE}](https://www.kaggle.com/datasets/{user}/{SLUG})**, part of the free **[Medical AI course]({site}/medical-ai)** by [Dr. Ram Kumar G]({site}/about).
>
> - Every record is **synthetic**: no real patients.
> - Learn the methods used here in the course lessons, for example [logistic-regression risk scores]({site}/medical-ai/machine-learning/logistic-regression-risk-score), [missing data and class imbalance]({site}/medical-ai/machine-learning/missing-data-and-imbalance) and [calibration]({site}/medical-ai/machine-learning/calibration-and-decision-curves).
""")
    code("""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Find the dataset folder on Kaggle (or use DATA_DIR when running elsewhere)
hits = glob.glob('/kaggle/input/**/diabetes_clinic_cohort.csv', recursive=True)
DATA = os.path.dirname(hits[0]) if hits else os.environ.get('DATA_DIR', '.')
files = sorted(f for f in os.listdir(DATA) if f.endswith('.csv'))
pd.DataFrame([(f, len(pd.read_csv(os.path.join(DATA, f)))) for f in files], columns=['file', 'rows'])
""")
    mdc("## The cohort\n\n5,000 synthetic adults attending a diabetes clinic. `admission_12m` is the target; it was generated from a known logistic model.")
    code("""
cohort = pd.read_csv(os.path.join(DATA, 'diabetes_clinic_cohort.csv'))
print(cohort.shape)
cohort.head()
""")
    code("""
cohort.describe().T[['count', 'mean', 'min', '50%', 'max']].round(1)
""")
    mdc("## Charts")
    code(f"""
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for dt, col in (('type 2', '{K.BLUE}'), ('type 1', '{K.ORANGE}')):
    ax[0].hist(cohort.loc[cohort.diabetes_type == dt, 'hba1c_percent'].dropna(), bins=36, alpha=0.8, color=col, label=dt)
ax[0].axvline(7, color='{K.INK_2}', lw=1); ax[0].set(title='HbA1c by diabetes type', xlabel='HbA1c (%)'); ax[0].legend()
n_comp = cohort[['retinopathy', 'neuropathy', 'foot_ulcer_history']].sum(axis=1).clip(upper=2)
(cohort.groupby(n_comp)['admission_12m'].mean() * 100).plot.bar(ax=ax[1], color=['{K.BLUE}', '{K.BLUE}', '{K.ORANGE}'], rot=0)
ax[1].set(title='Admission rate by number of complications', xlabel='Complications', ylabel='%')
(cohort.isna().mean() * 100).loc[lambda s: s > 0].sort_values().plot.barh(ax=ax[2], color='{K.BLUE}')
ax[2].set(title='Missing values (%)')
plt.tight_layout(); plt.show()
""")
    mdc("## A baseline model, and the true coefficients\n\nBecause the label came from a known model, we can fit the same terms and see how close the estimates get. Missing values are filled with the median here; try better imputation as an exercise.")
    code("""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X = pd.DataFrame({
    'age_per_10y_over_55': (cohort.age - 55) / 10,
    'hba1c_per_1pct_over_7.5': cohort.hba1c_percent.fillna(cohort.hba1c_percent.median()) - 7.5,
    'egfr_per_10_below_90': np.maximum(90 - cohort.egfr, 0) / 10,
    'on_insulin': cohort.on_insulin,
    'retinopathy': cohort.retinopathy,
    'neuropathy': cohort.neuropathy,
    'foot_ulcer_history': cohort.foot_ulcer_history,
    'smoker': cohort.smoker,
    'log_uacr_over_30': np.log(cohort.uacr_mg_g.fillna(cohort.uacr_mg_g.median()) / 30),
})
y = cohort.admission_12m
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = LogisticRegression(C=1e6, max_iter=1000).fit(X_tr, y_tr)
print('Test AUROC:', round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 3))
""")
    true = json.dumps(ADMISSION_MODEL)
    code(f"""
true = {true}
pd.DataFrame({{'estimated': [model.intercept_[0], *model.coef_[0]], 'true': [true['intercept'], *[true[c] for c in X.columns]]}},
             index=['intercept', *X.columns]).round(2)
""")
    mdc("## The text tables")
    code("""
for f in ['clinical_notes.csv', 'coded_notes.csv', 'deid_notes.csv', 'ambient_documentation_pairs.csv']:
    print(f); display(pd.read_csv(os.path.join(DATA, f)).head(3))
""")
    mdc(f"""
## Continue learning

- Full lessons, exercises and worked solutions: [{site.removeprefix('https://')}/medical-ai]({site}/medical-ai)
- Exercises and solutions: [{site.removeprefix('https://')}/medical-ai/exercises]({site}/medical-ai/exercises)
- About the author: [Dr. Ram Kumar G]({site}/about)

---

{DISCLAIMER}
""")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (STARTER / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    (STARTER / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/synthetic-diabetes-clinic-data-starter",
        "title": "Synthetic Diabetes Clinic Data: Starter",
        "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": False, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "keywords": ["healthcare", "medicine", "beginner"],
        "dataset_sources": [f"{user}/{SLUG}"], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2))


def main(user: str):
    course = json.loads((ROOT / "kaggle-course.json").read_text())
    course["kaggleUser"] = user
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "source").mkdir(parents=True)
    tables = build_tables(course)
    for name, v in tables.items():
        v["df"].to_csv(OUT / name, index=False)
    dictionary = pd.DataFrame([(f, c, t, d) for f, v in tables.items() for c, (t, d) in v["columns"].items()], columns=["file", "column", "type", "description"])
    dictionary.to_csv(OUT / "data_dictionary.csv", index=False)
    for script in sorted((ROOT / "exercise-data").glob("*/*.py")):
        shutil.copy(script, OUT / "source" / f"{script.parent.name}__{script.name}")
    (OUT / "fictional_study.txt").write_text(load_script("evaluation-safety/03-reporting-guidelines.py")["FICTIONAL_STUDY"].strip() + "\n")
    write_assumptions()
    make_charts(tables["diabetes_clinic_cohort.csv"]["df"])

    resources = [{"path": name, "description": v["description"], "schema": {"fields": [{"name": c, "type": t, "description": d} for c, (t, d) in v["columns"].items()]}} for name, v in tables.items()]
    resources += [
        {"path": "data_dictionary.csv", "description": "Every column of every CSV in this dataset, with its type and description.", "schema": {"fields": [
            {"name": "file", "type": "string", "description": "CSV file name"}, {"name": "column", "type": "string", "description": "Column name"},
            {"name": "type", "type": "string", "description": "Column type"}, {"name": "description", "type": "string", "description": "What the column holds"}]}},
        {"path": "cohort_generation_assumptions.md", "description": "How diabetes_clinic_cohort.csv was generated: assumptions, the true admission model and the missing-data mechanism."},
        {"path": "fictional_study.txt", "description": "A fictional prediction-model paper used with tripod_ai_items.csv to practise checking reporting quality."},
    ]
    meta = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "id": f"{user}/{SLUG}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["healthcare", "medicine", "nlp", "classification", "education"],
        "description": description(tables, course),
        "resources": resources,
        "userSpecifiedSources": (
            f"Written and generated by Dr. Ram Kumar G (MBBS, MD (Internal Medicine), MPH) for the Medical AI course at {course['site']}/medical-ai. "
            f"The cohort is produced by scripts/generate_synthetic_cohort.py (seed 42) in {REPO}; the text sets were hand-written for the course "
            "lessons. All patients, clinicians and products are fictional. Reporting items are paraphrased from TRIPOD+AI; regulatory cases are "
            "framed using public guidance from IMDRF, US FDA, the EU AI Act, UK MHRA, India CDSCO and WHO."
        ),
        "expectedUpdateFrequency": "annually",
        "image": "dataset-cover-image.png",
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    # `kaggle datasets metadata --update` wants the licence's display name, so it gets its own folder.
    upd = OUT.parent / "dataset-metadata-update"
    shutil.rmtree(upd, ignore_errors=True)
    upd.mkdir()
    shutil.copy(OUT / "dataset-cover-image.png", upd / "dataset-cover-image.png")
    (upd / "dataset-metadata.json").write_text(json.dumps({**meta, "licenses": [{"name": "Attribution 4.0 International (CC BY 4.0)"}]}, indent=2, ensure_ascii=False))
    write_starter(user, course)
    print(f"Built {len(tables)} tables, {len(dictionary)} dictionary rows, cover, charts and starter notebook")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    main(ap.parse_args().user)

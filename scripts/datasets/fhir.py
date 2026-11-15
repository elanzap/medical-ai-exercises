"""Dataset: Synthetic FHIR R4 Bundles for AI Agents (synthetic-fhir-r4-diabetes-bundles).

500 synthetic diabetes patients as FHIR R4 Bundles (Patient, Condition, Observation, MedicationRequest,
Encounter), flat CSV views of the same records, and ~1,500 agent tasks with expected tool calls and gold answers.
Built from the same generator as the Synthetic Diabetes Clinic Data cohort. No real patients.

Usage: uv run --python 3.12 --with numpy --with pandas --with matplotlib scripts/datasets/fhir.py --user rakuelli
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_common as K  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from generate_synthetic_cohort import generate as generate_cohort  # noqa: E402

SLUG = "synthetic-fhir-r4-diabetes-bundles"
TITLE = "Synthetic FHIR R4 Bundles for AI Agents"
SUBTITLE = "500 synthetic diabetes patients in FHIR R4, with agent tasks and gold answers"
OUT = K.BUILD / "datasets" / SLUG
ASSETS = K.ROOT / "assets" / "kaggle" / SLUG
TODAY = date(2026, 6, 30)  # synthetic "as of" date for the records

SNOMED, ICD10, LOINC, RXNORM = "http://snomed.info/sct", "http://hl7.org/fhir/sid/icd-10", "http://loinc.org", "http://www.nlm.nih.gov/research/umls/rxnorm"
CONDITIONS = {  # flag: (SNOMED code, display, ICD-10 code for type 2 / type 1)
    "type 2 diabetes": ("44054006", "Diabetes mellitus type 2", "E11"),
    "type 1 diabetes": ("46635009", "Diabetes mellitus type 1", "E10"),
    "hypertension": ("38341003", "Hypertensive disorder, systemic arterial", "I10"),
    "retinopathy": ("4855003", "Diabetic retinopathy", "E11.3"),
    "neuropathy": ("230572002", "Diabetic neuropathy", "E11.4"),
    "foot_ulcer_history": ("371087003", "Diabetic foot ulcer", "E11.621"),
    "ckd": ("709044004", "Chronic kidney disease", "N18"),
}
OBS = {  # column: (LOINC, display, unit)
    "hba1c_percent": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "fasting_glucose_mg_dl": ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    "egfr": ("98979-8", "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI 2021)", "mL/min/{1.73_m2}"),
    "uacr_mg_g": ("9318-7", "Albumin/Creatinine [Mass Ratio] in Urine", "mg/g"),
    "ldl_mg_dl": ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "mg/dL"),
    "hdl_mg_dl": ("2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "mg/dL"),
    "triglycerides_mg_dl": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
    "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2"),
    "waist_cm": ("8280-0", "Waist Circumference at umbilicus by Tape measure", "cm"),
}
MEDS = {  # flag: (RxNorm ingredient code, text)
    "on_metformin": ("6809", "Metformin 500 mg tablets"),
    "on_sglt2i": ("1545653", "Empagliflozin 10 mg tablets"),
    "on_glp1ra": ("1991302", "Semaglutide injection"),
    "on_insulin": ("274783", "Insulin glargine injection"),
    "on_statin": ("83367", "Atorvastatin 20 mg tablets"),
}


def coding(system, code, display):
    return {"coding": [{"system": system, "code": code, "display": display}], "text": display}


def build_records(n=500, seed=11):
    rng = np.random.default_rng(seed)
    cohort = generate_cohort(n, seed)
    bundles, rows = [], {"patients": [], "conditions": [], "observations": [], "medications": [], "encounters": []}
    for r in cohort.itertuples(index=False):
        pid = r.patient_id.replace("SYN", "FHIR")
        res = []
        birth = TODAY - timedelta(days=int(r.age * 365.25 + rng.integers(0, 365)))
        res.append({"resourceType": "Patient", "id": pid, "identifier": [{"system": "urn:synthetic:patient-id", "value": pid}],
                    "name": [{"text": f"Synthetic Patient {pid[5:]}"}], "gender": r.sex, "birthDate": birth.isoformat()})
        rows["patients"].append({"patient_id": pid, "gender": r.sex, "birth_date": birth.isoformat(), "age_on_2026_06_30": r.age})

        visits = sorted({TODAY - timedelta(days=int(d)) for d in [rng.integers(0, 40), rng.integers(150, 220), rng.integers(330, 400)]})
        for i, v in enumerate(visits, 1):
            eid = f"{pid}-enc{i}"
            res.append({"resourceType": "Encounter", "id": eid, "status": "finished", "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
                        "subject": {"reference": f"Patient/{pid}"}, "period": {"start": f"{v.isoformat()}T09:30:00+05:30", "end": f"{v.isoformat()}T09:50:00+05:30"}})
            rows["encounters"].append({"encounter_id": eid, "patient_id": pid, "date": v.isoformat(), "class": "AMB"})

        dtype = "type 1 diabetes" if r.diabetes_type == "type 1" else "type 2 diabetes"
        flags = [dtype] + [f for f, on in (("hypertension", r.hypertension), ("retinopathy", r.retinopathy), ("neuropathy", r.neuropathy),
                                           ("foot_ulcer_history", r.foot_ulcer_history), ("ckd", r.egfr < 60)) if on]
        for j, f in enumerate(flags, 1):
            sn, disp, icd = CONDITIONS[f]
            if dtype == "type 1 diabetes" and icd.startswith("E11"):
                icd = "E10" + icd[3:]
            onset = TODAY - timedelta(days=int(365.25 * (r.diabetes_duration_years if j == 1 else rng.uniform(0, max(r.diabetes_duration_years, 1)))))
            cid = f"{pid}-cond{j}"
            res.append({"resourceType": "Condition", "id": cid, "subject": {"reference": f"Patient/{pid}"},
                        "clinicalStatus": coding("http://terminology.hl7.org/CodeSystem/condition-clinical", "active", "Active"),
                        "code": {"coding": [{"system": SNOMED, "code": sn, "display": disp}, {"system": ICD10, "code": icd}], "text": disp},
                        "onsetDateTime": onset.isoformat()})
            rows["conditions"].append({"condition_id": cid, "patient_id": pid, "snomed_code": sn, "icd10_code": icd, "display": disp, "onset_date": onset.isoformat()})

        k = 0
        for vi, v in enumerate(visits):
            latest = vi == len(visits) - 1
            for col, (code, disp, unit) in OBS.items():
                val = getattr(r, col)
                if pd.isna(val):
                    continue
                if col not in ("hba1c_percent", "egfr", "bmi") and not latest:
                    continue  # earlier visits record HbA1c, eGFR and BMI only
                if not latest:  # earlier values drift back from today's value
                    back = len(visits) - 1 - vi
                    val = {"hba1c_percent": round(float(val) + rng.normal(0.25 * back, 0.4), 1), "egfr": int(round(float(val) + rng.normal(1.5 * back, 3))), "bmi": round(float(val) + rng.normal(0.3 * back, 0.4), 1)}[col]
                val = float(val)
                k += 1
                oid = f"{pid}-obs{k}"
                res.append({"resourceType": "Observation", "id": oid, "status": "final",
                            "category": [coding("http://terminology.hl7.org/CodeSystem/observation-category", "vital-signs" if col in ("bmi", "waist_cm") else "laboratory", "Vital Signs" if col in ("bmi", "waist_cm") else "Laboratory")],
                            "code": coding(LOINC, code, disp), "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": v.isoformat(),
                            "valueQuantity": {"value": val, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}})
                rows["observations"].append({"observation_id": oid, "patient_id": pid, "loinc_code": code, "measure": col, "value": val, "unit": unit, "date": v.isoformat()})
            if latest:
                k += 1
                oid = f"{pid}-obs{k}"
                res.append({"resourceType": "Observation", "id": oid, "status": "final", "category": [coding("http://terminology.hl7.org/CodeSystem/observation-category", "vital-signs", "Vital Signs")],
                            "code": coding(LOINC, "85354-9", "Blood pressure panel with all children optional"), "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": v.isoformat(),
                            "component": [{"code": coding(LOINC, "8480-6", "Systolic blood pressure"), "valueQuantity": {"value": int(r.systolic_bp), "unit": "mm[Hg]", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}},
                                          {"code": coding(LOINC, "8462-4", "Diastolic blood pressure"), "valueQuantity": {"value": int(r.diastolic_bp), "unit": "mm[Hg]", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}}]})
                rows["observations"] += [{"observation_id": oid, "patient_id": pid, "loinc_code": "8480-6", "measure": "systolic_bp", "value": int(r.systolic_bp), "unit": "mm[Hg]", "date": v.isoformat()},
                                         {"observation_id": oid, "patient_id": pid, "loinc_code": "8462-4", "measure": "diastolic_bp", "value": int(r.diastolic_bp), "unit": "mm[Hg]", "date": v.isoformat()}]

        m = 0
        for flag, (rx, text) in MEDS.items():
            if getattr(r, flag):
                m += 1
                mid = f"{pid}-med{m}"
                authored = TODAY - timedelta(days=int(rng.integers(30, 365 * 3)))
                res.append({"resourceType": "MedicationRequest", "id": mid, "status": "active", "intent": "order", "subject": {"reference": f"Patient/{pid}"},
                            "medicationCodeableConcept": coding(RXNORM, rx, text), "authoredOn": authored.isoformat()})
                rows["medications"].append({"medication_request_id": mid, "patient_id": pid, "rxnorm_code": rx, "medication": text, "status": "active", "authored_on": authored.isoformat()})
        bundles.append({"resourceType": "Bundle", "id": f"{pid}-bundle", "type": "collection", "timestamp": f"{TODAY.isoformat()}T00:00:00+05:30",
                        "entry": [{"fullUrl": f"urn:uuid:{x['resourceType']}-{x['id']}", "resource": x} for x in res]})
    return bundles, {k: pd.DataFrame(v) for k, v in rows.items()}


def tasks(t: dict, seed=12):
    rng = np.random.default_rng(seed)
    obs, cond, meds = t["observations"], t["conditions"], t["medications"]
    out = []
    for pid in t["patients"].patient_id:
        po = obs[obs.patient_id == pid]
        a1c = po[po.measure == "hba1c_percent"].sort_values("date")
        eg = po[po.measure == "egfr"].sort_values("date")
        bp = po[po.measure.isin(["systolic_bp", "diastolic_bp"])]
        names = set(cond[cond.patient_id == pid].display)
        mlist = sorted(meds[meds.patient_id == pid].medication)
        choices = []
        if len(a1c):
            last = a1c.iloc[-1]
            choices.append(("latest_value", f"What is the most recent HbA1c for patient {pid}, and when was it measured?",
                            [{"tool": "get_observations", "arguments": {"patient_id": pid, "loinc_code": "4548-4"}}], f"{last.value:g}% on {last.date}"))
        if len(a1c) >= 2:
            ch = round(a1c.iloc[-1].value - a1c.iloc[0].value, 1)
            choices.append(("trend", f"By how much has patient {pid}'s HbA1c changed between their first and latest result in the record?",
                            [{"tool": "get_observations", "arguments": {"patient_id": pid, "loinc_code": "4548-4"}}], f"{ch:+g} percentage points ({a1c.iloc[0].value:g}% to {a1c.iloc[-1].value:g}%)"))
        if len(eg):
            e = eg.iloc[-1].value
            on_met = any(x.startswith("Metformin") for x in mlist)
            verdict = ("stop metformin (eGFR below 30)" if e < 30 else "do not start metformin (eGFR below 45)" if e < 45 and not on_met else
                       "continue metformin and review (eGFR 30-44)" if e < 45 else "metformin can be used (eGFR 45 or more)")
            choices.append(("protocol_check", f"Under the course's teaching protocol (do not start metformin if eGFR is below 45; stop it if eGFR falls below 30), what applies to patient {pid}?",
                            [{"tool": "get_observations", "arguments": {"patient_id": pid, "loinc_code": "98979-8"}}, {"tool": "get_medications", "arguments": {"patient_id": pid}}], f"latest eGFR {e:g}: {verdict}"))
        cname = rng.choice(["Diabetic retinopathy", "Diabetic neuropathy", "Chronic kidney disease", "Hypertensive disorder, systemic arterial"])
        choices.append(("condition_lookup", f"Does patient {pid} have a recorded diagnosis of {cname.lower()}?",
                        [{"tool": "get_conditions", "arguments": {"patient_id": pid}}], "yes" if cname in names else "no"))
        choices.append(("medication_list", f"List patient {pid}'s active medicines.", [{"tool": "get_medications", "arguments": {"patient_id": pid}}], "; ".join(mlist) if mlist else "none recorded"))
        if len(bp) == 2:
            s, d = bp[bp.measure == "systolic_bp"].value.iloc[0], bp[bp.measure == "diastolic_bp"].value.iloc[0]
            choices.append(("threshold_check", f"Is patient {pid}'s latest blood pressure below 130/80 mmHg?",
                            [{"tool": "get_observations", "arguments": {"patient_id": pid, "loinc_code": "85354-9"}}], f"{'yes' if s < 130 and d < 80 else 'no'} ({int(s)}/{int(d)} mmHg)"))
        for i in rng.choice(len(choices), size=min(3, len(choices)), replace=False):
            ttype, q, calls, ans = choices[i]
            out.append({"task_type": ttype, "patient_id": pid, "question": q, "expected_tool_calls": json.dumps(calls), "expected_answer": ans})
    df = pd.DataFrame(out)
    df.insert(0, "task_id", [f"T{i:04d}" for i in range(1, len(df) + 1)])
    return df


def validate(bundles):
    ids = set()
    counts = {}
    for b in bundles:
        assert b["resourceType"] == "Bundle" and b["type"] == "collection"
        local = {f"{e['resource']['resourceType']}/{e['resource']['id']}" for e in b["entry"]}
        for e in b["entry"]:
            r = e["resource"]
            key = f"{r['resourceType']}/{r['id']}"
            assert key not in ids, key
            ids.add(key)
            counts[r["resourceType"]] = counts.get(r["resourceType"], 0) + 1
            subj = r.get("subject", {}).get("reference")
            assert subj is None or subj in local, f"dangling reference {subj}"
            if r["resourceType"] == "Observation":
                assert "valueQuantity" in r or "component" in r
    return counts


TABLES_DOC = {
    "fhir_bundles.ndjson": ("One FHIR R4 Bundle (type collection) per line, one per patient: Patient, Encounter, Condition, Observation and MedicationRequest resources with SNOMED CT, ICD-10, LOINC and RxNorm codes.", None),
    "agent_tasks.csv": ("Questions an AI agent should answer from the records, with the tool calls expected and the gold answer.", {
        "task_id": ("id", "Task identifier"), "task_type": ("string", "latest_value, trend, protocol_check, condition_lookup, medication_list or threshold_check"),
        "patient_id": ("string", "Patient the question is about"), "question": ("string", "The question in natural language"),
        "expected_tool_calls": ("string", "JSON list of the tool calls a correct agent would make (get_observations, get_conditions, get_medications)"),
        "expected_answer": ("string", "Gold answer computed from the records"),
    }),
    "patients.csv": ("Flat view: one row per synthetic patient.", {"patient_id": ("id", "Patient identifier (FHIR-00001 ...)"), "gender": ("string", "FHIR administrative gender"), "birth_date": ("datetime", "Synthetic birth date"), "age_on_2026_06_30": ("integer", "Age on the records' as-of date")}),
    "conditions.csv": ("Flat view of Condition resources.", {"condition_id": ("id", "Condition resource id"), "patient_id": ("string", "Patient"), "snomed_code": ("string", "SNOMED CT concept id"), "icd10_code": ("string", "ICD-10 code"), "display": ("string", "Condition name"), "onset_date": ("datetime", "Onset date")}),
    "observations.csv": ("Flat view of Observation resources (blood pressure split into systolic and diastolic rows).", {"observation_id": ("id", "Observation resource id"), "patient_id": ("string", "Patient"), "loinc_code": ("string", "LOINC code"), "measure": ("string", "Short measure name"), "value": ("numeric", "Result value"), "unit": ("string", "UCUM unit"), "date": ("datetime", "Date measured")}),
    "medications.csv": ("Flat view of MedicationRequest resources.", {"medication_request_id": ("id", "MedicationRequest id"), "patient_id": ("string", "Patient"), "rxnorm_code": ("string", "RxNorm ingredient code"), "medication": ("string", "Medicine as written"), "status": ("string", "Request status"), "authored_on": ("datetime", "Date prescribed")}),
    "encounters.csv": ("Flat view of Encounter resources (outpatient visits).", {"encounter_id": ("id", "Encounter id"), "patient_id": ("string", "Patient"), "date": ("datetime", "Visit date"), "class": ("string", "Encounter class (AMB = ambulatory)")}),
}


def charts(counts, task_df, t):
    ASSETS.mkdir(parents=True, exist_ok=True)
    K.style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), dpi=110)
    fig.subplots_adjust(top=0.72, bottom=0.14, wspace=0.45, left=0.13, right=0.97)
    c = pd.Series(counts).sort_values()
    ax[0].barh(c.index, c.values, height=0.5, color=K.BLUE)
    ax[0].grid(axis="y", visible=False)
    ax[0].grid(axis="x", visible=True)
    for i, v in enumerate(c.values):
        ax[0].text(v + c.max() * 0.01, i, f"{v:,}", va="center", fontsize=9)
    ax[0].set(xlabel="Resources")
    K.titled(ax[0], "FHIR resources in the bundles", "500 patients; most resources are lab and vital-sign Observations")
    tt = task_df.task_type.value_counts().sort_values()
    ax[1].barh([x.replace("_", " ") for x in tt.index], tt.values, height=0.5, color=K.BLUE)
    ax[1].grid(axis="y", visible=False)
    ax[1].grid(axis="x", visible=True)
    for i, v in enumerate(tt.values):
        ax[1].text(v + tt.max() * 0.01, i, f"{v:,}", va="center", fontsize=9)
    ax[1].set(xlabel="Tasks")
    K.titled(ax[1], "Agent tasks by type", "Each has expected tool calls and a gold answer")
    fig.text(0.01, 0.95, "Synthetic FHIR R4 bundles for AI agents", fontsize=14, color=K.INK, ha="left")
    fig.text(0.01, 0.9, "Synthetic diabetes patients; no real people", fontsize=9.5, color=K.INK_2, ha="left")
    K.source_note(fig, "Source: fhir_bundles.ndjson and agent_tasks.csv, Synthetic FHIR R4 Bundles for AI Agents (Kaggle)")
    fig.savefig(ASSETS / "overview.png")
    plt.close(fig)

    def draw(axc):
        axc.bar(range(len(c)), sorted(c.values), width=0.6, color=K.BLUE_LIGHT)
        axc.set_xlim(-0.7, len(c) - 0.3)

    K.cover(OUT / "dataset-cover-image.png", ("Synthetic FHIR R4", "Bundles"), "for AI agents and tool calling", "500 SYNTHETIC PATIENTS  |  LOINC  |  SNOMED CT  |  RXNORM", draw)


def description(user, counts, task_df, t, lessons):
    res = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    ex = task_df.groupby("task_type").head(1)
    examples = "\n".join(f"| {r.task_type.replace('_', ' ')} | {r.question} | {r.expected_answer} |" for r in ex.itertuples())
    return f"""# {TITLE}

Realistic-looking but **fully synthetic** electronic health records in **HL7 FHIR R4**, built for practising **AI agents, tool calling and clinical data retrieval**: 500 patients with diabetes, each as a FHIR Bundle with Patient, Encounter, Condition, Observation and MedicationRequest resources coded in **SNOMED CT, ICD-10, LOINC and RxNorm**, plus **{len(task_df):,} agent tasks** with the expected tool calls and gold answers. Flat CSV views make the same records easy to explore with pandas. It accompanies the free [Medical AI course]({K.SITE}/medical-ai) by [{K.AUTHOR}]({K.SITE}/about), a physician and diabetologist.

**No real patients.** Records are generated from the same synthetic cohort as [Synthetic Diabetes Clinic Data for Medical AI](https://www.kaggle.com/datasets/{user}/medical-ai-course-synthetic-exercise-data) (CC BY 4.0).

![Overview of the synthetic FHIR bundles]({K.RAW}/assets/kaggle/{SLUG}/overview.png)

## What's inside

| File | Rows | What it is |
|---|---|---|
| `fhir_bundles.ndjson` | 500 | One FHIR R4 Bundle per line, one per patient |
| `agent_tasks.csv` | {len(task_df):,} | Questions, expected tool calls and gold answers |
| `patients.csv` | {len(t['patients'])} | Flat view of Patient |
| `conditions.csv` | {len(t['conditions']):,} | Flat view of Condition (SNOMED CT and ICD-10) |
| `observations.csv` | {len(t['observations']):,} | Flat view of Observation (LOINC, UCUM units) |
| `medications.csv` | {len(t['medications']):,} | Flat view of MedicationRequest (RxNorm) |
| `encounters.csv` | {len(t['encounters']):,} | Flat view of Encounter |
| `code_systems.csv` | | Every code used, with its system and meaning |
| `data_dictionary.csv` | | Every column, with type and description |

## Resources

| FHIR resource | Count |
|---|---|
{res}

## Example tasks

| Type | Question | Gold answer |
|---|---|---|
{examples}

Expected tools: `get_observations(patient_id, loinc_code)`, `get_conditions(patient_id)` and `get_medications(patient_id)`. The protocol rules in the tasks come from the course's **synthetic teaching protocol**, not real clinical guidance.

## Ideas for what to do with it

- **Tool-calling agents:** give a model the three tools and score its answers and tool use against `agent_tasks.csv`.
- **FHIR skills:** parse bundles, follow references, and convert to OMOP-style or analysis-ready tables.
- **Retrieval:** answer questions over the records with and without structured tools, and compare hallucination rates.
- **Validation:** check codes, units and references, or deliberately corrupt records and build a FHIR quality checker.

A ready-to-run [starter notebook](https://www.kaggle.com/code/{user}/synthetic-fhir-r4-bundles-starter) parses a bundle, implements the three tools and scores a simple rule-based agent on the tasks.

{K.links_block(lessons)}
{K.related_datasets(user, SLUG)}
## Licence and citation

Data and text: CC BY 4.0. Code: MIT. Codes from SNOMED CT, ICD-10, LOINC and RxNorm are used only as identifiers for teaching; check each terminology's own licence before production use. Please cite as: *Ram Kumar G. Synthetic FHIR R4 Bundles for AI Agents. Kaggle, 2026. {K.SITE}/medical-ai*

{K.DISCLAIMER}
"""


STARTER = [
    ("md", "## Load the bundles"),
    ("code", """
import json, pandas as pd
bundles = [json.loads(line) for line in open(f'{DATA}/fhir_bundles.ndjson')]
tasks = pd.read_csv(f'{DATA}/agent_tasks.csv')
print(len(bundles), 'bundles;', len(tasks), 'tasks')
print(json.dumps(bundles[0]['entry'][0]['resource'], indent=1))
"""),
    ("md", "## Three tools an agent can call"),
    ("code", """
index = {b['entry'][0]['resource']['id']: [e['resource'] for e in b['entry']] for b in bundles}

def get_observations(patient_id, loinc_code):
    out = []
    for r in index[patient_id]:
        if r['resourceType'] == 'Observation' and r['code']['coding'][0]['code'] == loinc_code:
            out.append({'date': r['effectiveDateTime'], 'value': r.get('valueQuantity', {}).get('value'),
                        'components': [(c['code']['text'], c['valueQuantity']['value']) for c in r.get('component', [])]})
    return sorted(out, key=lambda o: o['date'])

def get_conditions(patient_id):
    return [r['code']['text'] for r in index[patient_id] if r['resourceType'] == 'Condition']

def get_medications(patient_id):
    return [r['medicationCodeableConcept']['text'] for r in index[patient_id] if r['resourceType'] == 'MedicationRequest' and r['status'] == 'active']

pid = tasks.patient_id.iloc[0]
print(get_observations(pid, '4548-4'), get_conditions(pid), get_medications(pid), sep='\\n')
"""),
    ("md", "## Score a simple rule-based agent\n\nA stand-in for a language model: it reads the question, calls the matching tool and formats an answer. Replace `agent()` with a call to your own model and keep the scoring. The `protocol_check` tasks are left unimplemented as an exercise: add the eGFR rule from the question."),
    ("code", """
def agent(question, patient_id):
    q = question.lower()
    if 'most recent hba1c' in q:
        o = get_observations(patient_id, '4548-4')[-1]; return f"{o['value']:g}% on {o['date']}"
    if 'changed' in q:
        o = get_observations(patient_id, '4548-4'); d = round(o[-1]['value'] - o[0]['value'], 1)
        return f"{d:+g} percentage points ({o[0]['value']:g}% to {o[-1]['value']:g}%)"
    if 'active medicines' in q:
        m = sorted(get_medications(patient_id)); return '; '.join(m) if m else 'none recorded'
    if 'diagnosis of' in q:
        name = q.split('diagnosis of ')[1].rstrip('?')
        return 'yes' if any(c.lower() == name for c in get_conditions(patient_id)) else 'no'
    if 'blood pressure below' in q:
        c = dict(get_observations(patient_id, '85354-9')[-1]['components'])
        s, d = c['Systolic blood pressure'], c['Diastolic blood pressure']
        return f"{'yes' if s < 130 and d < 80 else 'no'} ({s}/{d} mmHg)"
    return 'not implemented'

tasks['agent_answer'] = [agent(q, p) for q, p in zip(tasks.question, tasks.patient_id)]
tasks['correct'] = tasks.agent_answer == tasks.expected_answer
tasks.groupby('task_type').correct.mean().round(2)
"""),
]


def starter(user, lessons):
    cells = [K.md_cell(f"""
# Synthetic FHIR R4 Bundles: starter notebook

> Starter notebook for the dataset **[{TITLE}](https://www.kaggle.com/datasets/{user}/{SLUG})**, part of the free **[Medical AI course]({K.SITE}/medical-ai)** by [{K.AUTHOR}]({K.SITE}/about).
>
> - All records are **synthetic**: no real patients.
> - Lessons behind this data: [{lessons[0][0]}]({lessons[0][1]}) and [{lessons[1][0]}]({lessons[1][1]}).
"""), K.code_cell(K.find_data_snippet("fhir_bundles.ndjson"))]
    for kind, src in STARTER:
        cells.append(K.md_cell(src) if kind == "md" else K.code_cell(src))
    cells.append(K.md_cell(f"""
## Continue learning

- The agentic AI track: [{K.SITE.removeprefix('https://')}/medical-ai/agentic-ai]({K.SITE}/medical-ai/agentic-ai)
- The whole course: [{K.SITE.removeprefix('https://')}/medical-ai]({K.SITE}/medical-ai)
- About the author: [{K.AUTHOR}]({K.SITE}/about)

---

{K.DISCLAIMER}
"""))
    K.write_starter(K.BUILD / "starters" / SLUG, user, "synthetic-fhir-r4-bundles-starter", "Synthetic FHIR R4 Bundles: Starter", SLUG, cells)


def main(user):
    course = json.loads((K.ROOT / "kaggle-course.json").read_text())
    url = {l["slug"]: (l["title"], l["url"]) for l in course["lessons"]}
    lessons = [url[s] for s in ("fhir-tool-calling", "agents-and-tool-use", "multi-agent-clinical-workflows", "evaluating-agents", "healthcare-data-standards")]
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    bundles, t = build_records()
    counts = validate(bundles)
    task_df = tasks(t)
    with open(OUT / "fhir_bundles.ndjson", "w") as f:
        for b in bundles:
            f.write(json.dumps(b, separators=(",", ":")) + "\n")
    task_df.to_csv(OUT / "agent_tasks.csv", index=False)
    for k, v in t.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    codes = [("SNOMED CT", c[0], c[1]) for c in CONDITIONS.values()] + [("ICD-10", c[2], c[1]) for c in CONDITIONS.values()] + \
            [("LOINC", o[0], o[1]) for o in OBS.values()] + [("LOINC", "85354-9", "Blood pressure panel"), ("LOINC", "8480-6", "Systolic blood pressure"), ("LOINC", "8462-4", "Diastolic blood pressure")] + \
            [("RxNorm", m[0], m[1]) for m in MEDS.values()]
    pd.DataFrame(codes, columns=["system", "code", "meaning"]).to_csv(OUT / "code_systems.csv", index=False)
    TABLES_DOC["code_systems.csv"] = ("Every code used in the bundles, with its code system and meaning.", {"system": ("string", "Code system"), "code": ("string", "Code"), "meaning": ("string", "What the code means")})
    pd.DataFrame([(f, c, ty, d) for f, (_, cols) in TABLES_DOC.items() if cols for c, (ty, d) in cols.items()], columns=["file", "column", "type", "description"]).to_csv(OUT / "data_dictionary.csv", index=False)
    charts(counts, task_df, t)
    tables = {f: {"description": d, "columns": c} for f, (d, c) in TABLES_DOC.items()}
    tables["data_dictionary.csv"] = {"description": "Every column of every CSV in this dataset, with its type and description.", "columns": {"file": ("string", "CSV file name"), "column": ("string", "Column name"), "type": ("string", "Column type"), "description": ("string", "What the column holds")}}
    K.write_metadata(OUT, {
        "title": TITLE, "subtitle": SUBTITLE, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["healthcare", "medicine", "nlp", "education", "artificial intelligence"],
        "description": description(user, counts, task_df, t, lessons),
        "resources": K.resources(tables),
        "userSpecifiedSources": f"Generated by {K.AUTHOR} ({K.AUTHOR_DEGREE}) for the Medical AI course at {K.SITE}/medical-ai, using scripts/datasets/fhir.py (seed 11) in {K.REPO}, from the same synthetic cohort generator as Synthetic Diabetes Clinic Data for Medical AI. Resources follow HL7 FHIR R4; codes from SNOMED CT, ICD-10, LOINC and RxNorm are used as identifiers for teaching. Protocol rules in the tasks come from the course's synthetic teaching protocol. No real patients.",
        "expectedUpdateFrequency": "annually",
        "image": "dataset-cover-image.png",
    })
    starter(user, lessons)
    print("resources", counts, "| tasks", task_df.task_type.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    main(ap.parse_args().user)

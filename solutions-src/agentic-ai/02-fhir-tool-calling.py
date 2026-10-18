# %% Step 1
import datetime

extra = [
    {"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/p1"},
     "clinicalStatus": {"coding": [{"code": "active"}]},
     "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Type 2 diabetes mellitus"}]}},
    {"resourceType": "Condition", "id": "c2", "subject": {"reference": "Patient/p1"},
     "clinicalStatus": {"coding": [{"code": "resolved"}]},
     "code": {"coding": [{"system": "http://snomed.info/sct", "code": "195662009", "display": "Acute viral pharyngitis"}]}},
    # AllergyIntolerance points to the patient with "patient", not "subject"
    {"resourceType": "AllergyIntolerance", "id": "a1", "patient": {"reference": "Patient/p1"},
     "code": {"text": "Penicillin"}, "reaction": [{"description": "Rash"}]},
    {"resourceType": "Observation", "id": "o4", "subject": {"reference": "Patient/p1"},
     "code": {"coding": [{"system": "http://loinc.org", "code": "13457-7", "display": "LDL cholesterol (calculated)"}]},
     "effectiveDateTime": "2025-03-14", "valueQuantity": {"value": 3.1, "unit": "mmol/L"}},
    {"resourceType": "Observation", "id": "o5", "subject": {"reference": "Patient/p2"},
     "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
     "effectiveDateTime": "2026-08-01", "valueQuantity": {"value": 9.6, "unit": "%"}},
]
all_resources = resources + extra

class MoreTools(PatientTools):
    def _mine(self, resource_type):
        return [r for r in self.resources if r["resourceType"] == resource_type
                and (r.get("subject") or r.get("patient") or {}).get("reference") == self.patient_ref]

    def active_conditions(self):
        self.audit.append(("active_conditions", None))
        return [{"condition": c["code"]["coding"][0]["display"], "snomed": c["code"]["coding"][0]["code"],
                 "source": f"Condition/{c['id']}"}
                for c in self._mine("Condition") if c["clinicalStatus"]["coding"][0]["code"] == "active"]

    def allergies(self):
        self.audit.append(("allergies", None))
        return [{"substance": a["code"]["text"], "reaction": a.get("reaction", [{}])[0].get("description"),
                 "source": f"AllergyIntolerance/{a['id']}"} for a in self._mine("AllergyIntolerance")]

t1 = MoreTools(all_resources, "p1")
print(t1.active_conditions())
print(t1.allergies())

# %% Note
# The lesson's `_mine` only looked at `subject`, so it would have returned *no allergies* for every patient,
# with no error. A tool that silently returns an empty list is more dangerous than one that fails; test each tool
# against a record where you know the answer.

# %% Step 2
TODAY = datetime.date(2026, 10, 18)      # fixed so the result is reproducible

class StaleAwareTools(MoreTools):
    def latest_observation(self, loinc_code, max_age_days=365):
        result = super().latest_observation(loinc_code)
        if result.get("date"):
            age = (TODAY - datetime.date.fromisoformat(result["date"][:10])).days   # date part of a FHIR dateTime
            result["days_old"] = age
            result["stale"] = age > max_age_days
        return result

t2 = StaleAwareTools(all_resources, "p1")
for code in ["4548-4", "13457-7"]:
    print(t2.latest_observation(code))

# %% Step 3
t_p1 = StaleAwareTools(all_resources, "p1")
print("p1 HbA1c:", t_p1.latest_observation("4548-4"))
try:
    t_p1.latest_observation("4548-4", patient_id="p2")          # the model tries to choose the patient
except TypeError as e:
    print("Asking for p2 through the p1 tools fails:", e)
print("Observation o5 belongs to p2 and exists in the bundle:", any(r["id"] == "o5" for r in all_resources))
print("Can the p1 tools see it?", any(t_p1.latest_observation(c).get("source") == "Observation/o5" for c in ["4548-4"]))

# %% Note
# The patient is fixed when the tools are created, by the clinician's context, and no tool takes a patient
# argument, so there is nothing for a model (or an injected instruction) to change. Scoping is a property of the
# code, which can be tested, rather than an instruction the model is asked to follow.

# %% Step 4
def summary_model(messages):
    """Scripted model that gathers facts with the tools, then writes a summary citing every source."""
    plan = [("latest_observation", {"loinc_code": "4548-4"}), ("latest_observation", {"loinc_code": "98979-8"}),
            ("latest_observation", {"loinc_code": "13457-7"}), ("active_medications", {}),
            ("active_conditions", {}), ("allergies", {})]
    done = [m for m in messages if m["role"] == "tool"]
    if len(done) < len(plan):
        name, args = plan[len(done)]
        return {"tool_call": {"name": name, "arguments": args}}
    r = {i: m["content"] for i, m in enumerate(done)}
    meds = "; ".join(f"{m['medication']} [{m['source']}]" for m in r[3])
    conds = "; ".join(f"{c['condition']} [{c['source']}]" for c in r[4])
    allergies = "; ".join(f"{a['substance']} ({a['reaction']}) [{a['source']}]" for a in r[5])
    ldl = r[2]
    return {"answer": "\n".join([
        "Pre-clinic summary (synthetic patient p1) - draft for clinician review",
        f"- Conditions: {conds}",
        f"- HbA1c {r[0]['value']}{r[0]['unit']} on {r[0]['date']} [{r[0]['source']}]",
        f"- eGFR {r[1]['value']} {r[1]['unit']} on {r[1]['date']} [{r[1]['source']}]",
        f"- LDL {ldl['value']} {ldl['unit']} on {ldl['date']} [{ldl['source']}]" + (" - STALE, over 12 months old" if ldl["stale"] else ""),
        f"- Active medicines: {meds}",
        f"- Allergies: {allergies}",
    ])}

def run_with_tools(question, model, tool_obj, allowed, max_steps=10):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        reply = model(messages)
        if "answer" in reply:
            return reply["answer"], messages
        call = reply["tool_call"]
        if call["name"] not in allowed:
            raise ValueError(f"tool {call['name']} not allowed")
        messages.append({"role": "tool", "name": call["name"], "content": getattr(tool_obj, call["name"])(**call["arguments"])})
    raise RuntimeError("step limit reached")

ALLOWED = {"latest_observation", "active_medications", "active_conditions", "allergies"}
summary, _ = run_with_tools("Prepare a pre-clinic summary.", summary_model, StaleAwareTools(all_resources, "p1"), ALLOWED)
print(summary)

# %% Note
# Every fact carries the FHIR resource it came from, so a clinician can check any line in one click, and stale
# results are labelled rather than presented as current. The checks in the next lesson rely on those references.

# %% Step 5
# Optional: the same tools on a Synthea FHIR R4 bundle (synthetic patients, Apache 2.0; about 85 MB download).
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_r4_sep2019.zip"
zip_path = Path("data/synthea_fhir_r4_sep2019.zip")
zip_path.parent.mkdir(exist_ok=True)
if not zip_path.exists():
    urlretrieve(URL, zip_path)

with zipfile.ZipFile(zip_path) as z:
    name = next(n for n in sorted(z.namelist()) if n.endswith(".json") and "Information" not in n)
    synthea = [e["resource"] for e in json.load(z.open(name))["entry"]]

# Synthea references resources as "urn:uuid:<id>"; rewrite them to the "Patient/<id>" form the tools expect
patient = next(r for r in synthea if r["resourceType"] == "Patient")
for r in synthea:
    for key in ("subject", "patient"):
        if r.get(key, {}).get("reference", "").startswith("urn:uuid:"):
            r[key] = {"reference": "Patient/" + r[key]["reference"].removeprefix("urn:uuid:")}
real_tools = StaleAwareTools(synthea, patient["id"])
print("HbA1c:", real_tools.latest_observation("4548-4"))
print("Glucose (2339-0):", real_tools.latest_observation("2339-0"))
print("Active conditions:", [c["condition"] for c in real_tools.active_conditions()][:6])
from collections import Counter
print("Active medications:", len(real_tools.active_medications()),
      "| MedicationRequest statuses in this bundle:", dict(Counter(r["status"] for r in synthea if r["resourceType"] == "MedicationRequest")))
print("Allergies:", real_tools.allergies())

# %% Note
# Real bundles expose the assumptions in toy tools: Synthea writes references as `urn:uuid:` rather than
# `Patient/<id>`, dates carry a time and time zone ("2019-04-22T07:22:41-04:00", which first broke the stale
# check), and many results are years old (hence the stale flag). Each mismatch needs handling explicitly; a tool should fail loudly on a
# resource shape it does not understand.

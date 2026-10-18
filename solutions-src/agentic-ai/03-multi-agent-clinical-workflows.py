# %% Step 1
facts = extractor_agent(tools)
print(drafter_agent(facts, checker_agent(facts, tools)))

# %% Step 2
import copy

# Add an allergy for p1 and a result for p2, so that two of the injected errors are possible
record = resources + [
    {"resourceType": "AllergyIntolerance", "id": "a1", "patient": {"reference": "Patient/p1"}, "code": {"text": "Penicillin"}},
    {"resourceType": "Observation", "id": "o9", "subject": {"reference": "Patient/p2"},
     "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
     "effectiveDateTime": "2026-08-01", "valueQuantity": {"value": 9.6, "unit": "%"}},
]

class RecordTools(PatientTools):
    def allergies(self):
        return [{"substance": a["code"]["text"], "source": f"AllergyIntolerance/{a['id']}"}
                for a in self.resources if a["resourceType"] == "AllergyIntolerance"
                and a["patient"]["reference"] == self.patient_ref]

rt = RecordTools(record, "p1")

def correct_facts(t):
    h = t.latest_observation("4548-4")
    return {"hba1c": {"value": h["value"], "unit": h["unit"], "source": h["source"]},
            "egfr": t.latest_observation("98979-8"), "medications": t.active_medications(), "allergies": t.allergies()}

def inject(kind):
    f = copy.deepcopy(correct_facts(rt))
    if kind == "wrong value":
        f["hba1c"]["value"] = 6.4
    elif kind == "stopped medicine reported as active":
        f["medications"].append({"medication": "Gliclazide 80 mg tablets", "source": "MedicationRequest/m2"})
    elif kind == "wrong unit":
        f["hba1c"]["unit"] = "mmol/mol"
    elif kind == "missing allergy":
        f["allergies"] = []
    elif kind == "wrong patient":
        f["hba1c"] = {"value": 9.6, "unit": "%", "source": "Observation/o9"}
    return f

ERRORS = ["wrong value", "stopped medicine reported as active", "wrong unit", "missing allergy", "wrong patient"]

def lesson_checker(f, t):
    """The lesson's checker, unchanged apart from the extra keys it ignores."""
    return checker_agent(f, t)

def caught(checker, kind):
    clean_issues = set(checker(correct_facts(rt), rt))         # issues raised even with correct facts
    return bool(set(checker(inject(kind), rt)) - clean_issues)  # anything new is a detection

for kind in ERRORS:
    print(f"{kind:<38} caught by the lesson's checker: {caught(lesson_checker, kind)}")

# %% Note
# The lesson's checker re-queries only the latest HbA1c value (and applies one clinical rule), so it catches the
# wrong value and the other patient's result, which also has a different value. It misses the stopped medicine,
# the wrong unit and the missing allergy, because it never looks at them. A checker only checks what it checks.

# %% Step 3
def full_checker(f, t):
    issues = []
    truth = correct_facts(t)                                   # re-query the source record independently
    for key in ("hba1c", "egfr"):
        for field in ("value", "unit", "source"):
            if f[key].get(field) != truth[key].get(field):
                issues.append(f"{key} {field}: summary has {f[key].get(field)!r}, record has {truth[key].get(field)!r}")
    if {m["source"] for m in f["medications"]} != {m["source"] for m in truth["medications"]}:
        issues.append(f"medications differ from active MedicationRequests: {[m['medication'] for m in truth['medications']]}")
    if {a["source"] for a in f["allergies"]} != {a["source"] for a in truth["allergies"]}:
        issues.append(f"allergies differ from the record: {[a['substance'] for a in truth['allergies']]}")
    # Every cited source must belong to this patient
    ours = {f"{r['resourceType']}/{r['id']}" for r in t.resources
            if (r.get("subject") or r.get("patient") or {}).get("reference") == t.patient_ref}
    cited = [f[k]["source"] for k in ("hba1c", "egfr")] + [m["source"] for m in f["medications"] + f["allergies"]]
    issues += [f"source {s} does not belong to {t.patient_ref}" for s in cited if s not in ours]
    return issues

print("Full checker on correct facts:", full_checker(correct_facts(rt), rt))
for kind in ERRORS:
    print(f"{kind:<38} caught: {caught(full_checker, kind)}  {full_checker(inject(kind), rt)[:1]}")

# %% Note
# The extended checker compares every field of the summary with a fresh query of the record, and checks that
# every cited resource belongs to the patient. It uses only re-queries, never the extractor's own reasoning, which
# is what makes it independent. It still cannot catch errors in the record itself, or a fact that both agents get
# wrong in the same way because they share the same tool.

# %% Step 4
import numpy as np

rng = np.random.default_rng(0)
N, STEPS, P_STEP = 1000, 4, 0.95
P_CATCH = 0.90          # assumption: the checker detects 90% of errors (it is not perfect either)

step_ok = rng.random((N, STEPS)) < P_STEP
run_has_error = ~step_ok.all(axis=1)
detected = run_has_error & (rng.random(N) < P_CATCH)
print(f"Runs with at least one step error: {run_has_error.mean():.1%} (theory {1 - P_STEP ** STEPS:.1%})")
print(f"Without a checker, errors reaching the clinician unflagged: {run_has_error.mean():.1%}")
print(f"With the checker, errors reaching the clinician unflagged: {(run_has_error & ~detected).mean():.1%}")
print(f"Runs flagged for review: {detected.mean():.1%}")

# %% Note
# Four steps at 95% each give an error in about one run in five. An independent checker that catches 90% of them
# cuts unflagged errors to about 2%, at the cost of flagging about one run in six for human review. The checker's
# value depends on its errors being independent of the extractor's: two agents built on the same model and the
# same tools tend to make the same mistakes, which this simulation does not capture.

# %% Step 5
# %% Note
# Example approval rules for a pre-clinic summary agent:
#
# 1. **Read-only.** The agent can only read the record of the patient whose clinic slot is open, through
#    patient-scoped tools. It cannot order, prescribe, message or edit anything.
# 2. **Every fact cited.** Each line must cite the FHIR resource it came from; uncited lines are removed before
#    display.
# 3. **Independent check before display.** A checker re-queries the record; any mismatch, stale result (over 12
#    months), or rule flag (for example metformin with eGFR below 45) is shown at the top of the summary.
# 4. **Marked as a draft.** The summary is labelled as AI-generated and is never filed in the record or sent to
#    the patient without the clinician editing and signing it.
# 5. **Clinician decides.** Flags are prompts for review, not recommendations; prescribing decisions stay with the
#    clinician.
# 6. **Logged and audited.** Every run's tool calls, checker output and the clinician's edits are logged; a sample
#    is audited monthly, and any summary error that reached a decision is reported as an incident.
# 7. **Stop rules.** The clinical safety officer can switch the agent off; it is suspended automatically if the
#    audit error rate exceeds the agreed threshold or the record system changes.

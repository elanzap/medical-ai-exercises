# %% Step 1
# Synthea FHIR R4 sample (synthetic patients, Apache 2.0), pinned to the September 2019 release.
# The zip is about 85 MB; we read only the first five patient bundles straight from it.
import json
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URL = "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_r4_sep2019.zip"
zip_path = Path("data/synthea_fhir_r4_sep2019.zip")
zip_path.parent.mkdir(exist_ok=True)
if not zip_path.exists():
    urlretrieve(URL, zip_path)

with zipfile.ZipFile(zip_path) as z:
    names = sorted(n for n in z.namelist()
                   if n.endswith(".json") and "hospitalInformation" not in n and "practitionerInformation" not in n)
    bundles = [json.load(z.open(n)) for n in names[:5]]

counts = Counter(entry["resource"]["resourceType"] for b in bundles for entry in b["entry"])
print(f"{len(bundles)} bundles, {sum(counts.values()):,} resources")
for resource_type, n in counts.most_common(10):
    print(f"  {resource_type:<22} {n:>5}")

# %% Step 2
# The lesson used 2345-7 (glucose in serum/plasma); Synthea codes glucose as 2339-0 (in blood).
# We accept both, and keep the original code so the difference stays visible.
CODES = {"4548-4": "hba1c", "2345-7": "glucose", "2339-0": "glucose"}

rows = []
for bundle in bundles:
    for entry in bundle["entry"]:
        r = entry["resource"]
        if r["resourceType"] != "Observation" or "valueQuantity" not in r:
            continue
        for coding in r["code"]["coding"]:
            if coding.get("system") == "http://loinc.org" and coding["code"] in CODES:
                rows.append({
                    "patient": r["subject"]["reference"].replace("urn:uuid:", ""),
                    "date": r["effectiveDateTime"][:10],
                    "loinc": coding["code"],
                    "test": CODES[coding["code"]],
                    "value": r["valueQuantity"]["value"],
                    "unit": r["valueQuantity"]["unit"],
                })

site_a = pd.DataFrame(rows).assign(site="synthea")
print(site_a.groupby(["test", "loinc", "unit"]).size())
print(site_a.head())

# %% Step 3
# A made-up second site that reports SI units: glucose in mmol/L and HbA1c in IFCC mmol/mol.
# We build it by converting a copy of the Synthea values, so we know what the harmonised answer should be.
site_b = site_a.copy()
site_b["site"] = "site_b"
is_glucose = site_b.test == "glucose"
site_b.loc[is_glucose, "value"] = (site_b.loc[is_glucose, "value"] / 18.0).round(1)
site_b.loc[is_glucose, "unit"] = "mmol/L"
site_b.loc[~is_glucose, "value"] = ((site_b.loc[~is_glucose, "value"] - 2.15) * 10.929).round(0)
site_b.loc[~is_glucose, "unit"] = "mmol/mol"

def to_standard(row):
    """Convert to one unit per test: glucose mg/dL, HbA1c % (NGSP)."""
    value, unit = row.value, row.unit
    if row.test == "glucose" and unit == "mmol/L":
        value, unit = value * 18.0, "mg/dL"
    elif row.test == "hba1c" and unit == "mmol/mol":
        value, unit = value / 10.929 + 2.15, "%"
    return pd.Series({"value_std": round(value, 1), "unit_std": unit})

combined = pd.concat([site_a, site_b], ignore_index=True)
combined = combined.join(combined.apply(to_standard, axis=1))
print(combined.groupby(["site", "test", "unit", "unit_std"]).size())

# Check: after harmonisation the two sites should agree to within rounding.
# Row i of site B is a converted copy of row i of the Synthea site.
a, b = combined[combined.site == "synthea"], combined[combined.site == "site_b"]
diff = (a.value_std.to_numpy() - b.value_std.to_numpy())
for test in ["glucose", "hba1c"]:
    mask = a.test.to_numpy() == test
    print(f"Largest {test} difference between sites after harmonisation: {abs(diff[mask]).max():.1f} {a.unit_std[a.test == test].iloc[0]}")

# %% Note
# The small remaining differences come from rounding site B's values when we created them, exactly as
# happens with real laboratory reports. Always keep the original value and unit next to the
# harmonised ones, so a conversion error can be traced.

# %% Step 4
# %% Note
# What you should find (as of October 2026; browsers: SNOMED International browser, WHO ICD-10 and ICD-11 browsers):
#
# - **SNOMED CT 44054006** is *Diabetes mellitus type 2 (disorder)*. It is a clinical concept in a
#   polyhierarchy: it can have several parents and has many more specific children (for example type 2
#   diabetes with particular complications). It is designed for recording care at full clinical detail.
# - **ICD-10 E11** is *Type 2 diabetes mellitus*, a category in a single-parent classification. Detail
#   comes from the fourth character (E11.2 with kidney complications, E11.9 without complications, and so on).
# - **ICD-11 5A11** is *Type 2 diabetes mellitus*. ICD-11 adds detail by post-coordination: you link
#   codes together (for example the diabetes code with a separate code for a complication such as
#   diabetic retinopathy) rather than choosing from a fixed list of combined sub-codes.
#
# The key difference: SNOMED CT describes what the clinician observed; ICD *classifies* it into
# statistical categories for reporting and billing. Mapping SNOMED CT to ICD loses detail, and the
# reverse cannot restore it.

# %% Step 5
# A sketch of where each extracted item lands in the OMOP Common Data Model.
omop = pd.DataFrame([
    ("Patient resource (id, birth date, gender)", "person", "person_id, year_of_birth, gender_concept_id"),
    ("HbA1c Observation, LOINC 4548-4", "measurement", "measurement_concept_id (LOINC 4548-4), value_as_number, unit_concept_id, measurement_date"),
    ("Glucose Observation, LOINC 2339-0 / 2345-7", "measurement", "a separate measurement_concept_id for each LOINC code"),
    ("Original site code and unit", "measurement", "measurement_source_value, unit_source_value"),
    ("Condition: type 2 diabetes (SNOMED CT 44054006)", "condition_occurrence", "condition_concept_id, condition_start_date"),
    ("Encounter the result belongs to", "visit_occurrence", "linked through visit_occurrence_id"),
], columns=["Extracted item", "OMOP table", "Key fields"])
print(omop.to_string(index=False))

# %% Note
# In OMOP, measurements are stored with standard concepts (LOINC for lab tests) and conditions with
# standard SNOMED CT concepts, while the original codes are kept in the `_source_value` fields. ICD codes
# from billing data are mapped to SNOMED CT standard concepts on load.

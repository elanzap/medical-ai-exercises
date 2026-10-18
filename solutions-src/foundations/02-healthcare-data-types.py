# %% Step 1
# Synthea sample data (synthetic patients, Apache 2.0), pinned to the April 2020 release so the numbers below stay stable.
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URL = "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip"
zip_path = Path("data/synthea_csv_apr2020.zip")
zip_path.parent.mkdir(exist_ok=True)
if not zip_path.exists():
    urlretrieve(URL, zip_path)  # about 9 MB

with zipfile.ZipFile(zip_path) as z:
    patients = pd.read_csv(z.open("csv/patients.csv"), usecols=["Id", "BIRTHDATE", "DEATHDATE", "GENDER"],
                           parse_dates=["BIRTHDATE", "DEATHDATE"])
    observations = pd.read_csv(z.open("csv/observations.csv"), parse_dates=["DATE"])

print(f"{len(patients):,} patients, {len(observations):,} observations")
print(observations.head(3))

# %% Step 2
# Synthea codes glucose as LOINC 2339-0 (Glucose [Mass/volume] in Blood), not 2345-7 (in Serum or Plasma).
# Checking which codes a source actually uses, before filtering, is part of the job.
LABS = {"4548-4": "hba1c", "2339-0": "glucose", "2345-7": "glucose", "38483-4": "creatinine"}
labs = observations[observations.CODE.isin(LABS)].copy()
labs["test"] = labs.CODE.map(LABS)
labs["value"] = pd.to_numeric(labs.VALUE, errors="coerce")
labs["DATE"] = labs.DATE.dt.tz_localize(None)

print(labs.groupby(["test", "CODE", "UNITS"]).size())

# %% Step 3
# Index date: 1 January 2020. Only results on or before it count, otherwise the table "knows the future".
# Patients who died before the index date are excluded: they could not have a prediction made for them.
INDEX_DATE = pd.Timestamp("2020-01-01")
alive = patients[(patients.BIRTHDATE < INDEX_DATE) &
                 (patients.DEATHDATE.isna() | (patients.DEATHDATE > INDEX_DATE))]

before = labs[labs.DATE <= INDEX_DATE].sort_values("DATE")
last = before.groupby(["PATIENT", "test"]).last()
wide = last["value"].unstack("test")
days_ago = (INDEX_DATE - last["DATE"]).dt.days.unstack("test").add_suffix("_days_ago")

table = (alive.set_index("Id")[["BIRTHDATE", "GENDER"]]
         .join(wide).join(days_ago))
table["age"] = ((INDEX_DATE - table.BIRTHDATE).dt.days / 365.25).astype(int)
table = table.drop(columns="BIRTHDATE")
print(f"{len(table):,} patients alive at the index date")
print(table.dropna(subset=["hba1c"]).head().round(1))   # a few patients who have an HbA1c

# %% Note
# The `_days_ago` columns matter as much as the values: an HbA1c from last month and one from
# eight years ago are not the same piece of information, but a model sees only a number unless you tell it.

# %% Step 4
tests = ["hba1c", "glucose", "creatinine"]
print("Missing per test:")
print(table[tests].isna().sum())

table["age_group"] = pd.cut(table.age, bins=[0, 18, 40, 65, 120], right=False,
                            labels=["<18", "18-39", "40-64", "65+"])
missing_by_age = table.groupby("age_group", observed=True)[tests].agg(lambda s: s.isna().mean())
print("\nShare missing by age group:")
print(missing_by_age.round(2))

# %% Note
# Missingness is far from random: in this sample no child has any of the three tests, while adults
# aged 65 and over are the most likely to have them. Whether a test was ordered is itself a signal about the patient (and about the clinician),
# so "missing" should be modelled explicitly, not silently filled with an average.

# %% Step 5
# %% Note
# Three ways this table could mislead a model (example answers):
#
# 1. **Informative missingness.** Tests are ordered when a clinician is worried. A model can learn that
#    "has an HbA1c" predicts diabetes without learning anything about glucose metabolism, and it will
#    fail in a setting with universal screening.
# 2. **Stale values.** The latest value may be years old. Without the `_days_ago` columns, a 2012 HbA1c
#    looks as current as one from last week.
# 3. **Synthetic and single-region data.** Synthea simulates Massachusetts residents from published
#    disease models. Its distributions, testing patterns and units are not those of any real hospital,
#    so anything learned here must be revalidated on real (properly governed) data before it means anything.
#
# Other good answers: one-row-per-patient hides trends (a falling HbA1c and a rising one can have the same
# latest value); the index-date choice changes who is included; codes differ between sources (2339-0 vs 2345-7).

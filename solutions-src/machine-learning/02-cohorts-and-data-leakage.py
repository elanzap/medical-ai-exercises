# %% Step 1
# Synthea sample data (synthetic patients, Apache 2.0), pinned to the April 2020 release.
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip"
zip_path = Path("data/synthea_csv_apr2020.zip")
zip_path.parent.mkdir(exist_ok=True)
if not zip_path.exists():
    urlretrieve(URL, zip_path)  # about 9 MB

with zipfile.ZipFile(zip_path) as z:
    encounters = pd.read_csv(z.open("csv/encounters.csv"), usecols=["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS"])
    patients = pd.read_csv(z.open("csv/patients.csv"), usecols=["Id", "BIRTHDATE", "DEATHDATE", "GENDER"])
    conditions = pd.read_csv(z.open("csv/conditions.csv"), usecols=["START", "PATIENT", "CODE"])
for frame, cols in [(encounters, ["START", "STOP"]), (patients, ["BIRTHDATE", "DEATHDATE"]), (conditions, ["START"])]:
    for c in cols:
        frame[c] = pd.to_datetime(frame[c], utc=True).dt.tz_localize(None)   # one time zone for all dates

inpatient = (encounters[encounters.ENCOUNTERCLASS == "inpatient"]
             .sort_values(["PATIENT", "START"]).reset_index(drop=True))
inpatient["los_days"] = (inpatient.STOP - inpatient.START).dt.total_seconds() / 86400
inpatient["next_admission"] = inpatient.groupby("PATIENT").START.shift(-1)
inpatient["readmit_30d"] = ((inpatient.next_admission - inpatient.STOP).dt.days <= 30).astype(int)

# Time zero is discharge. Drop stays whose 30-day window runs past the end of the data (we cannot see their
# outcome) and patients who died within 30 days without readmission (they could not be readmitted).
data_end = encounters.STOP.max()
cohort = inpatient[inpatient.STOP <= data_end - pd.Timedelta(days=30)].merge(
    patients, left_on="PATIENT", right_on="Id", suffixes=("", "_patient"))
died_early = (cohort.DEATHDATE - cohort.STOP).dt.days.between(0, 30) & (cohort.readmit_30d == 0)
acute = cohort.los_days <= 30          # a few Synthea "inpatient" encounters last years (long-term care)
cohort = cohort[~died_early & acute].reset_index(drop=True)
print(f"{len(cohort):,} index stays from {cohort.PATIENT.nunique()} patients; "
      f"30-day readmission {cohort.readmit_30d.mean():.1%}")

# %% Note
# Synthea does not mark admissions as planned or unplanned, so every readmission counts here. A real study
# would exclude planned readmissions (for example with the US CMS planned-readmission algorithm).

# %% Step 2
def history(stay, before_days=365):
    """Features using only data recorded up to discharge."""
    pid, t0 = stay.PATIENT, stay.STOP
    past = encounters[(encounters.PATIENT == pid) & (encounters.STOP <= t0)]
    recent = past[past.START >= t0 - pd.Timedelta(days=before_days)]
    return pd.Series({
        "age": (t0 - stay.BIRTHDATE).days / 365.25,
        "female": int(stay.GENDER == "F"),
        "length_of_stay": stay.los_days,
        "inpatient_past_year": int((recent.ENCOUNTERCLASS == "inpatient").sum()) - 1,   # minus this stay
        "emergency_past_year": int((recent.ENCOUNTERCLASS == "emergency").sum()),
        "encounters_past_year": len(recent),
        "conditions_so_far": int(((conditions.PATIENT == pid) & (conditions.START <= t0)).sum()),
    })

features = cohort.apply(history, axis=1)

# A common real-world leak: a patient summary table, joined in, that counts ALL the patient's inpatient
# stays, including those after this discharge. It looks like history but contains the future.
total_stays = inpatient.groupby("PATIENT").size()
features["total_inpatient_stays"] = cohort.PATIENT.map(total_stays).to_numpy()
honest_cols = [c for c in features.columns if c != "total_inpatient_stays"]
print(features.describe().round(1).T[["mean", "50%", "max"]])

# %% Step 3
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split

y, groups = cohort.readmit_30d, cohort.PATIENT

def auroc(cols, train_idx, test_idx):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    m.fit(features.iloc[train_idx][cols], y.iloc[train_idx])
    return roc_auc_score(y.iloc[test_idx], m.predict_proba(features.iloc[test_idx][cols])[:, 1])

results = []
for seed in range(5):   # repeat the splits: with ~280 patients a single split is noisy
    row = train_test_split(np.arange(len(cohort)), test_size=0.3, random_state=seed)
    pat = next(GroupShuffleSplit(test_size=0.3, random_state=seed).split(features, y, groups))
    for split_name, (tr, te) in [("row split", row), ("patient split", pat)]:
        for feat_name, cols in [("honest features", honest_cols), ("+ leaky feature", honest_cols + ["total_inpatient_stays"])]:
            results.append({"split": split_name, "features": feat_name, "AUROC": auroc(cols, tr, te)})
summary = pd.DataFrame(results).groupby(["split", "features"]).AUROC.agg(["mean", "std"]).round(3)
print(summary)

# %% Step 4
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

tr, te = next(GroupShuffleSplit(test_size=0.3, random_state=0).split(features, y, groups))
X_train, X_test = features.iloc[tr][honest_cols], features.iloc[te][honest_cols]

def fit_pipeline():
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y.iloc[tr])
    return roc_auc_score(y.iloc[te], pipe.predict_proba(X_test)[:, 1])

print(f"Pipeline, run 1: AUROC {fit_pipeline():.4f}")
print(f"Pipeline, run 2: AUROC {fit_pipeline():.4f}")

# The leaky alternative: scale using statistics from ALL rows, including the test patients.
scaled_all = pd.DataFrame(StandardScaler().fit_transform(features[honest_cols]), columns=honest_cols)
lr = LogisticRegression(max_iter=1000).fit(scaled_all.iloc[tr], y.iloc[tr])
print(f"Scaling before splitting: AUROC {roc_auc_score(y.iloc[te], lr.predict_proba(scaled_all.iloc[te])[:, 1]):.4f}")

# %% Note
# The Pipeline gives the same answer every run, because all the steps are fitted on the training patients
# only, in a fixed order. With a simple scaler the leak from scaling on all rows is tiny; the habit matters
# because the same mistake with feature selection or target encoding can be large (see the worked example).

# %% Step 5
cohort["year"] = cohort.STOP.dt.year
last_year = cohort.year.max()
if (cohort.year == last_year).sum() < 100:      # the last calendar year in the sample is partial
    last_year -= 1
train_t = np.where(cohort.year < last_year)[0]
test_t = np.where(cohort.year == last_year)[0]
print(f"Temporal split: train {len(train_t)} stays (to {last_year - 1}), test {len(test_t)} stays ({last_year})")
print(f"Temporal split AUROC        {auroc(honest_cols, train_t, test_t):.3f}")
print(f"Random patient split (mean) {summary.loc[('patient split', 'honest features'), 'mean']:.3f}")

# %% Note
# What the numbers show (April 2020 Synthea sample):
#
# - **Row split vs patient split.** Splitting by stay instead of by patient raises AUROC by about 0.04,
#   because stays of the same patient end up in both training and test sets and the model partly learns to
#   recognise patients. The patient split is the honest estimate.
# - **The leaky feature** adds a little more (about 0.015 with the patient split). It is small here only
#   because the honest "inpatient stays in the past year" already carries most of the same signal; in real
#   data, leaks built from future information can add far more, and they are easy to miss because the
#   column name sounds harmless.
# - **The temporal split** scores *higher* than the random patient split. Synthea's simulated care does not
#   change over the years, so there is no drift to expose, and the 2019 test set is small (120 stays) and
#   different in mix. In real hospital data, temporal validation is usually the harshest test, because coding,
#   practice and populations change; do not take this result as typical.
#
# Two cautions about the data: Synthea's 43% readmission rate and very high AUROCs reflect how its disease
# modules schedule repeat admissions, not real-world readmission patterns; and with under 250 patients every
# estimate is noisy (see the standard deviations in step 3). Reporting that uncertainty is part of the answer.

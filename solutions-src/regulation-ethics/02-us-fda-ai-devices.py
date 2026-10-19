# %% Step 1
import pandas as pd
from pathlib import Path

def load_fda_list(path=FDA_LIST_FILE):
    """Reads the FDA AI-enabled device list CSV (columns as published: Date of Final Decision, Submission Number,
    Device, Company, Panel (Lead), Primary Product Code). Header rows above the table are skipped."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines()
    start = next(i for i, line in enumerate(raw) if line.startswith("Date of Final Decision"))
    from io import StringIO
    return pd.read_csv(StringIO("\n".join(raw[start:])))

def pathway(submission):
    s = str(submission).upper()
    return "De Novo" if s.startswith("DEN") else "PMA" if s.startswith("P") else "510(k)" if s.startswith("K") else "other"

if Path(FDA_LIST_FILE).exists():
    devices = load_fda_list()
    devices["pathway"] = devices["Submission Number"].map(pathway)
    print(f"{len(devices)} devices in the list")
    print(devices["pathway"].value_counts().to_string())
    print(devices["Panel (Lead)"].value_counts().head(8).to_string())
    print(f"Radiology: {(devices['Panel (Lead)'].str.contains('Radiology', case=False)).sum()} devices")
else:
    print(f"{FDA_LIST_FILE} not found: download the CSV from the FDA page in a browser and upload it, then re-run.")

# %% Note
# We could not download the list from a script when writing this solution (the FDA site blocked automated requests),
# so the code was tested on a small file with the published column names, not on the real list. The submission
# number tells you the pathway: K numbers are 510(k) clearances, DEN numbers De Novo, and P numbers PMA. As the lesson
# says, the large majority are 510(k) and most are in radiology; your counts will depend on the list's date.

# %% Step 2
if Path(FDA_LIST_FILE).exists():
    diabetes = devices[devices["Device"].str.contains("retin|diabet|insulin|glucose", case=False, na=False)]
    print(diabetes[["Date of Final Decision", "Submission Number", "Device", "Company", "pathway"]].to_string(index=False))

# %% Note
# Open each device's FDA database entry from its submission number and read the decision summary. What to note:
# the pathway and predicate (for 510(k)), the intended use statement, the study design (retrospective images or a
# prospective study in the intended setting), the population and sites, the reference standard, the primary results
# with confidence intervals, and any limitations in the labelling.
#
# A useful comparison point is the first autonomous diabetic retinopathy system, authorised through De Novo in 2018
# (DEN180001): its pivotal study enrolled 900 people with diabetes at 10 primary-care sites and reported sensitivity
# 87.2% and specificity 90.7% against a reading-centre reference standard (Abràmoff et al., npj Digital Medicine
# 2018). Many 510(k) clearances rest on smaller, retrospective comparisons with a predicate.

# %% Step 3
CRITERIA = [
    "1. Does NOT analyse a medical image, an IVD signal, or a signal/pattern from a signal-acquisition system",
    "2. Displays, analyses or prints medical information (for example patient data, guidelines)",
    "3. Supports or provides recommendations to a health care professional",
    "4. The professional can independently review the basis of the recommendation (not rely primarily on it)",
]
judgements = {   # one True/False per criterion, with the deciding reason
    "StatinGuide": ([True, True, True, True], "meets all four: likely non-device CDS"),
    "ECG-Rhythm": ([False, False, True, False], "analyses a physiological signal: a device"),
    "A1cTrend": ([True, True, True, True], "displays data for a professional; likely non-device (it may not even be CDS)"),
    "SepsisNow": ([True, True, True, False], "time-critical alert with a hidden basis: the clinician cannot independently review it, so a device"),
    "PatientDose": ([True, True, False, False], "recommendations go to patients, not professionals: a device"),
}
for name, (met, reason) in judgements.items():
    print(f"{name:<12} {''.join('Y' if m else 'n' for m in met)}  {reason}")

# %% Note
# All four criteria must be met for software to fall outside the device definition under the 21st Century Cures Act.
# The FDA's guidance (revised January 2026, per the lesson) explains how it reads them; the conclusions above are
# teaching judgements on fictional products. Note how often criterion 4 decides the case: urgency and opacity both
# make independent review unrealistic.

# %% Step 4
# %% Note
# PCCP outline for RetinaNext (fictional), yearly retraining:
#
# **Description of modifications**
# - Annual retraining of the grading model on new de-identified images from partner clinics, including images from
#   additional camera models listed in an appendix.
# - No change to the intended use, the user, the output categories or the operating threshold's target performance.
#
# **Modification protocol**
# - Data: inclusion rules for new images; reference standard grading by certified graders; a locked, sequestered
#   test set (plus a new test set for each added camera) never used for training.
# - Training: the same architecture family and training procedure; documented random seeds and versions.
# - Acceptance criteria: sensitivity and specificity for referable retinopathy no lower than the cleared version's
#   lower confidence bounds, overall and in each subgroup (camera model, site, age, sex, ethnicity where recorded),
#   and image-gradability rate within a set margin.
# - Implementation: staged release, the ability to roll back to the previous version, and user notification of each
#   update with a summary of performance.
#
# **Impact assessment**
# - Benefits: better performance on new cameras and populations. Risks: performance falling in a subgroup, drift in
#   gradability, users not noticing a change. Controls: subgroup acceptance criteria, post-release monitoring of
#   referral rates and graded samples, and a stop rule.
#
# Changes outside this plan (a new output, autonomous use in a new setting, a different reference standard) would
# need a new submission.

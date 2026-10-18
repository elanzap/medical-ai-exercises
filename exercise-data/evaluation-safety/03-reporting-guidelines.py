# A FICTIONAL study summary written for this exercise. The model, authors, hospital and results do not exist.
FICTIONAL_STUDY = """
Title: Machine learning predicts kidney disease progression in type 2 diabetes

Abstract. We developed DKD-Predict, a gradient-boosting model that predicts progression of diabetic kidney disease
within 3 years. Using electronic records from one tertiary hospital (2016-2021), we included 1,200 adults with type 2
diabetes and an eGFR measurement. The model achieved an AUROC of 0.91 on the test set and outperformed logistic
regression (AUROC 0.78). DKD-Predict could help clinicians identify patients for early nephrology referral.

Methods.
- Participants: adults with type 2 diabetes who had at least two eGFR results during follow-up. Patients with
  missing values in any predictor were excluded (n = 410), leaving 1,200.
- Outcome: progression, defined as a fall in eGFR of 40% or more, or kidney failure, within 3 years of the first
  eGFR. Outcome assessors had access to all predictors.
- Predictors (32): age, sex, HbA1c, eGFR, urine albumin-to-creatinine ratio, blood pressure, medicines, number of
  nephrology clinic visits during follow-up, and 24 other laboratory values.
- Analysis: patients were randomly split 80/20 by record into training and test sets. Hyperparameters were tuned to
  maximise test-set AUROC. 96 patients (8%) had the outcome.
- Performance: AUROC on the test set. A feature-importance plot is shown.
- No external validation was performed. Code is available on request.

Limitations. Our model performed excellently and is ready for clinical implementation. Future work could include
other hospitals.
"""

# A selection of topics covered by TRIPOD+AI items, PARAPHRASED for this exercise (it is not the official checklist;
# for real appraisals use the full checklist from the TRIPOD+AI statement, BMJ 2024;385:e078378).
TRIPOD_AI_EXTRACT = [
    "Title identifies the study as developing or evaluating a prediction model, the population and the outcome",
    "Source of data, setting and dates are described",
    "Eligibility criteria for participants are described",
    "Outcome definition, timing and whether assessment was blinded to predictors",
    "Predictors: definition and when they were measured relative to the start of prediction",
    "Sample size: how it was decided, including number of events",
    "Missing data: how they were handled",
    "Model development: type of model, tuning of hyperparameters, internal validation method",
    "Performance measures: discrimination AND calibration, with confidence intervals",
    "Fairness: performance in relevant subgroups",
    "Model availability: how the model can be accessed or used (and code)",
    "Limitations, including risk of bias and generalisability",
    "Patient and public involvement, registration/protocol, funding and conflicts of interest",
]
print(f"Study text: {len(FICTIONAL_STUDY.split())} words; checklist extract: {len(TRIPOD_AI_EXTRACT)} items")

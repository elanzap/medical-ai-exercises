# %% Step 1
print(FICTIONAL_STUDY)

# %% Note
# The study develops a model that predicts a future outcome for individuals from several predictors, and evaluates
# it on held-out data. That is a prediction-model study, so **TRIPOD+AI** applies (it covers regression and machine
# learning). It is not a diagnostic accuracy study (STARD-AI), an imaging study (CLAIM) or a trial (CONSORT-AI), and
# it does not use a language model (TRIPOD-LLM). For appraising it, **PROBAST+AI** is the matching reader's tool.

# %% Step 2
assessment = [   # (status, reason) for each item in TRIPOD_AI_EXTRACT, in order
    ("partly", "says machine learning and kidney progression, but not 'development' or the population (type 2 diabetes)"),
    ("reported", "one tertiary hospital, 2016-2021"),
    ("partly", "inclusion stated, but requiring two eGFR results during follow-up selects on the future"),
    ("partly", "definition and 3-year horizon given; assessors were NOT blinded to predictors"),
    ("partly", "listed, but timing not stated; nephrology visits 'during follow-up' happen after prediction time"),
    ("missing", "no sample size justification; 96 events for 32 predictors"),
    ("reported", "complete-case analysis (410 excluded) is reported, though it is a poor choice"),
    ("partly", "model type given; tuning used the test set; internal validation is a single random split by record"),
    ("partly", "AUROC only, no confidence interval, no calibration"),
    ("missing", "no subgroup results"),
    ("partly", "code 'on request'; no model specification or access"),
    ("partly", "limitations paragraph overstates readiness and ignores bias"),
    ("missing", "no patient involvement, registration, funding or conflicts reported"),
]
counts = {"reported": 0, "partly": 0, "missing": 0}
for item, (status, reason) in zip(TRIPOD_AI_EXTRACT, assessment):
    counts[status] += 1
    print(f"[{status:<8}] {item}\n            {reason}")
print("\n", counts)

# %% Step 3
probast = {
    "Participants and data sources": ("HIGH", "single centre; inclusion requires two eGFR results during follow-up, "
                                      "which excludes patients who left care or died early; 410 excluded for missing data"),
    "Predictors": ("HIGH", "'nephrology visits during follow-up' are measured after time zero and are partly caused "
                   "by the outcome: leakage"),
    "Outcome": ("HIGH", "assessors could see the predictors; the outcome depends on how often eGFR was tested, which "
                "itself depends on clinical suspicion"),
    "Analysis": ("HIGH", "8% events with 32 predictors; complete-case analysis; random split by record (patients may be in "
                 "both sets); hyperparameters tuned on the test set; no calibration"),
}
for domain, (judgement, reason) in probast.items():
    print(f"{domain:<30} risk of bias {judgement}: {reason}")
print("\nOverall: HIGH risk of bias. Applicability to other settings: unclear (single tertiary centre).")

# %% Note
# One high-risk domain is enough for an overall high risk of bias; here every domain has a serious problem. The
# reported AUROC of 0.91 is very likely optimistic: test-set tuning and a leaky predictor each inflate it.

# %% Step 4
# %% Note
# The three pieces of missing information that would most change trust in the results:
#
# 1. **Performance without the leaky predictor, on a test set never used for tuning**, ideally at another
#    hospital or a later time period. This is what the model would achieve in use.
# 2. **Calibration** (calibration plot, slope and intercept, with confidence intervals). A referral tool shows
#    risks to clinicians, and a well-ranking but miscalibrated model can still mislead.
# 3. **Who was excluded and their outcomes**: the 410 with missing data and those without two eGFR results. If they
#    differ (for example sicker patients who were tested less), the model was built on an unrepresentative group.

# %% Step 5
# %% Note
# Rewritten limitations paragraph:
#
# "This was a single-centre, retrospective study, and the results should be regarded as preliminary. Several design
# choices are likely to have made performance look better than it would be in practice: the number of nephrology
# visits during follow-up, one of the predictors, is recorded after the time of prediction and is related to the
# outcome; hyperparameters were tuned on the test set; and the test set was formed by a random split, so the same
# patients may contribute to both sets. We did not assess calibration or performance in subgroups. Patients with
# missing data, or with fewer than two eGFR results, were excluded, which may limit generalisability to the wider
# population with type 2 diabetes. With 96 outcome events, estimates are imprecise. The model should not be used in
# clinical care until it has been re-developed without post-baseline predictors and validated externally, including
# calibration, in the settings where it would be used."

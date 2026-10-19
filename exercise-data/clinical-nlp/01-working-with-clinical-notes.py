# Twenty SYNTHETIC clinic notes: five made-up patients, four visits each, in date order. No real patients.
clinic_notes = [
    ("P1", 1, """HISTORY: 61F with T2DM x 8 yrs, HTN. c/o tingling in feet x 2 mo. Denies chest pain.
MEDICATIONS: Metformin 1 g BD.
EXAMINATION: Reduced monofilament sensation both feet. No ulcers.
ASSESSMENT: Possible diabetic peripheral neuropathy. Family history of kidney disease (mother).
PLAN: Refer for DPN testing. Review in 3 mo."""),
    ("P1", 2, """HISTORY: 61F with T2DM x 8 yrs, HTN. c/o tingling in feet x 2 mo. Denies chest pain.
MEDICATIONS: Metformin 1 g BD.
EXAMINATION: Reduced monofilament sensation both feet. No ulcers.
ASSESSMENT: Possible diabetic peripheral neuropathy. Family history of kidney disease (mother).
PLAN: Refer for DPN testing. Review in 3 mo."""),
    ("P1", 3, """HISTORY: 61F with T2DM, HTN. Tingling persists. DPN test done, reduced sural response.
MEDICATIONS: Metformin 1 g BD. Pregabalin 75 mg OD started.
EXAMINATION: Reduced monofilament sensation both feet. No ulcers.
ASSESSMENT: Diabetic peripheral neuropathy confirmed.
PLAN: Foot care advice. Review in 3 mo."""),
    ("P1", 4, """HISTORY: 61F with T2DM, HTN. Tingling better on pregabalin. No SOB.
MEDICATIONS: Metformin 1 g BD. Pregabalin 75 mg OD.
EXAMINATION: No ulcers. Pedal pulses present.
ASSESSMENT: Diabetic peripheral neuropathy, stable.
PLAN: Continue. Review in 6 mo."""),
    ("P2", 1, """HISTORY: 45M, new T2DM. No chest pain. Father had a heart attack at 52.
MEDICATIONS: None.
EXAMINATION: BMI 33. Pedal pulses present.
ASSESSMENT: Type 2 diabetes, new diagnosis. Query fatty liver on scan.
PLAN: Start metformin 500 mg OD. Diet advice."""),
    ("P2", 2, """HISTORY: 45M with T2DM. Mild nausea on metformin. No chest pain.
MEDICATIONS: Metformin 500 mg BD.
EXAMINATION: BMI 32.
ASSESSMENT: Type 2 diabetes. Probable MASLD.
PLAN: Liver fibrosis score. Review in 3 mo."""),
    ("P2", 3, """HISTORY: 45M with T2DM. Mild nausea on metformin. No chest pain.
MEDICATIONS: Metformin 500 mg BD.
EXAMINATION: BMI 32.
ASSESSMENT: Type 2 diabetes. Probable MASLD.
PLAN: Liver fibrosis score. Review in 3 mo."""),
    ("P2", 4, """HISTORY: 45M with T2DM. Nausea resolved. Denies SOB.
MEDICATIONS: Metformin 1 g BD.
EXAMINATION: BMI 31.
ASSESSMENT: Type 2 diabetes. MASLD, low fibrosis risk.
PLAN: Review in 6 mo."""),
    ("P3", 1, """HISTORY: 70F with T2DM x 20 yrs, CKD 3. Left heel ulcer x 2 wk. No fever.
MEDICATIONS: Insulin glargine 20 u. Amlodipine 5 mg OD.
EXAMINATION: 2 cm ulcer left heel, no signs of infection. Pedal pulses absent.
ASSESSMENT: Diabetic foot ulcer. Possible peripheral arterial disease.
PLAN: Urgent podiatry. Arterial Doppler."""),
    ("P3", 2, """HISTORY: 70F with T2DM, CKD 3. Ulcer slowly healing. No fever.
MEDICATIONS: Insulin glargine 20 u. Amlodipine 5 mg OD.
EXAMINATION: 1 cm ulcer left heel, clean. Pedal pulses absent.
ASSESSMENT: Diabetic foot ulcer, improving. Peripheral arterial disease on Doppler.
PLAN: Vascular referral. Weekly dressings."""),
    ("P3", 3, """HISTORY: 70F with T2DM, CKD 3. Ulcer slowly healing. No fever.
MEDICATIONS: Insulin glargine 20 u. Amlodipine 5 mg OD.
EXAMINATION: 1 cm ulcer left heel, clean. Pedal pulses absent.
ASSESSMENT: Diabetic foot ulcer, improving. Peripheral arterial disease on Doppler.
PLAN: Vascular referral. Weekly dressings."""),
    ("P3", 4, """HISTORY: 70F with T2DM, CKD 3. Ulcer healed. Denies rest pain.
MEDICATIONS: Insulin glargine 18 u. Amlodipine 5 mg OD.
EXAMINATION: Heel skin intact. Pedal pulses absent.
ASSESSMENT: Healed ulcer. Peripheral arterial disease.
PLAN: Foot protection service. Review in 3 mo."""),
    ("P4", 1, """HISTORY: 29F, 26 weeks pregnant. Family history of diabetes (mother and sister).
MEDICATIONS: Folic acid.
EXAMINATION: BP 118/76.
ASSESSMENT: Gestational diabetes on OGTT.
PLAN: Diet advice. Home glucose monitoring."""),
    ("P4", 2, """HISTORY: 29F, 28 weeks pregnant. Fasting readings above target. No headache.
MEDICATIONS: Folic acid.
EXAMINATION: BP 120/78. No oedema.
ASSESSMENT: Gestational diabetes, not controlled on diet.
PLAN: Start insulin at night."""),
    ("P4", 3, """HISTORY: 29F, 32 weeks pregnant. Readings at target. No hypos.
MEDICATIONS: Isophane insulin 8 u at night.
EXAMINATION: BP 122/80.
ASSESSMENT: Gestational diabetes, controlled.
PLAN: Continue. Growth scan."""),
    ("P4", 4, """HISTORY: Postnatal, 6 weeks. Insulin stopped after delivery.
MEDICATIONS: None.
EXAMINATION: BP 116/74.
ASSESSMENT: Previous gestational diabetes. Rule out type 2 diabetes.
PLAN: HbA1c at 12 weeks postpartum."""),
    ("P5", 1, """HISTORY: 38M with T1DM since age 12. 3 hypos a week, mostly at night. No seizures.
MEDICATIONS: Basal-bolus insulin.
EXAMINATION: Injection sites: lipohypertrophy abdomen.
ASSESSMENT: Recurrent hypoglycaemia. Lipohypertrophy.
PLAN: Rotate sites. Reduce basal 10%. CGM."""),
    ("P5", 2, """HISTORY: 38M with T1DM. Hypos fewer since basal reduced. No seizures.
MEDICATIONS: Basal-bolus insulin.
EXAMINATION: Lipohypertrophy smaller.
ASSESSMENT: Hypoglycaemia improving.
PLAN: CGM review in 2 wk."""),
    ("P5", 3, """HISTORY: 38M with T1DM. Hypos fewer since basal reduced. No seizures.
MEDICATIONS: Basal-bolus insulin.
EXAMINATION: Lipohypertrophy smaller.
ASSESSMENT: Hypoglycaemia improving.
PLAN: CGM review in 2 wk."""),
    ("P5", 4, """HISTORY: 38M with T1DM. No hypos in last month. Brother has coeliac disease.
MEDICATIONS: Basal-bolus insulin.
EXAMINATION: Injection sites normal.
ASSESSMENT: Type 1 diabetes, stable. ?coeliac screen.
PLAN: Coeliac serology. Review in 3 mo."""),
]
print(f"{len(clinic_notes)} notes for {len({p for p, _, _ in clinic_notes})} synthetic patients")

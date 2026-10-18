# Synthetic note-to-summary pairs for the optional fine-tuning step (no real patients).
# The first 12 are for training, the last 4 are held out for comparison.
note_summary_pairs = [
    ("58M T2DM 10y. Metformin 1 g BD. HbA1c 8.6% (was 7.9%). BP 142/88. eGFR 72. Plan: add sitagliptin, diet referral, review 3 mo.",
     "Type 2 diabetes, HbA1c rising to 8.6% on metformin. Sitagliptin added; dietitian referral; review in 3 months."),
    ("34F T1DM since 15. Basal-bolus. 3 hypos/wk, mostly night. HbA1c 6.9%. Plan: reduce basal 10%, CGM review 2 wk.",
     "Type 1 diabetes with frequent night-time hypoglycaemia. Basal insulin reduced by 10%; CGM review in 2 weeks."),
    ("66F T2DM, CKD 3a (eGFR 52), UACR 45 mg/mmol. On metformin. HbA1c 7.4%. Plan: start empagliflozin 10 mg, recheck U&E 4 wk.",
     "Type 2 diabetes with CKD stage 3a and albuminuria. Empagliflozin 10 mg started; kidney function check in 4 weeks."),
    ("29F, 26/40 pregnant. OGTT fasting 5.8, 2h 8.9 mmol/L. Plan: GDM diet advice, home glucose monitoring, review 1 wk.",
     "Gestational diabetes diagnosed at 26 weeks. Diet advice and home glucose monitoring; review in 1 week."),
    ("71M T2DM 20y. Insulin glargine 24 u. Numb feet both sides, monofilament absent L. No ulcers. Plan: podiatry, foot education.",
     "Long-standing type 2 diabetes on glargine with reduced sensation in both feet. Podiatry referral and foot-care education."),
    ("45M T2DM new dx. HbA1c 9.8%. BMI 33. No symptoms. Plan: metformin titrate to 1 g BD, structured education, review 3 mo.",
     "New type 2 diabetes, HbA1c 9.8%. Metformin started and titrated; structured education; review in 3 months."),
    ("52F T2DM. HbA1c 7.1%. LDL 3.4 mmol/L, not on statin. Plan: discussed statin, patient agreed atorvastatin 20 mg.",
     "Type 2 diabetes, HbA1c 7.1%. Not on a statin with LDL 3.4; atorvastatin 20 mg started after discussion."),
    ("38M T1DM. DKA admission last month after missed doses. HbA1c 11.2%. Plan: sick-day rules, psychology referral, review 2 wk.",
     "Type 1 diabetes, recent DKA after missed insulin, HbA1c 11.2%. Sick-day rules taught; psychology referral; review in 2 weeks."),
    ("60F T2DM. Retinal screen: background retinopathy R. HbA1c 7.8%. BP 150/90. Plan: start ramipril 2.5 mg, rescreen 12 mo.",
     "Type 2 diabetes with background retinopathy and raised blood pressure. Ramipril 2.5 mg started; repeat eye screening in 12 months."),
    ("48M T2DM. On gliclazide + metformin. Weight up 6 kg. HbA1c 8.0%. Plan: stop gliclazide, start semaglutide, review 6 wk.",
     "Type 2 diabetes with weight gain on gliclazide. Gliclazide stopped and semaglutide started; review in 6 weeks."),
    ("75F T2DM, frail, lives alone. HbA1c 6.2% on gliclazide. Two falls. Plan: stop gliclazide, relax target, review 4 wk.",
     "Frail older woman with type 2 diabetes and falls on gliclazide, HbA1c 6.2%. Gliclazide stopped; less strict target; review in 4 weeks."),
    ("41F LADA suspected. GAD antibodies positive. HbA1c 8.9%. Plan: start basal insulin 10 u, education, review 2 wk.",
     "Probable LADA with positive GAD antibodies, HbA1c 8.9%. Basal insulin 10 units started with education; review in 2 weeks."),
    # held out
    ("63M T2DM. Foot ulcer R hallux 1 cm, no infection signs. HbA1c 8.3%. Plan: offloading, dressing, podiatry within 1 wk.",
     "Type 2 diabetes with a 1 cm right big-toe ulcer without infection. Offloading, dressings and podiatry within 1 week."),
    ("27M T1DM. Starting pump. HbA1c 7.6%. Plan: pump training booked, carb counting refresher, review after training.",
     "Type 1 diabetes, HbA1c 7.6%, starting an insulin pump. Pump training and carbohydrate-counting refresher booked."),
    ("55F T2DM. eGFR 38. On metformin 1 g BD. HbA1c 7.2%. Plan: reduce metformin to 500 mg BD, recheck eGFR 3 mo.",
     "Type 2 diabetes with eGFR 38. Metformin reduced to 500 mg twice daily; kidney function recheck in 3 months."),
    ("69M T2DM. Chest tightness on exertion 2 wk. ECG today normal. Plan: urgent cardiology referral, GTN spray.",
     "Type 2 diabetes with two weeks of exertional chest tightness, ECG normal. Urgent cardiology referral; GTN spray given."),
]
train_pairs, held_out_pairs = note_summary_pairs[:12], note_summary_pairs[12:]
print(f"{len(train_pairs)} training pairs, {len(held_out_pairs)} held-out pairs")

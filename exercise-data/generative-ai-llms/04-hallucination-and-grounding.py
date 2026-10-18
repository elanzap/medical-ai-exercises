# Ten synthetic consultation transcripts (no real patients), each with an AI-style draft note written to
# contain the kinds of errors seen in AI documentation, and clinician-style labels.
# Label types: "hallucination" (not supported by the transcript), "contradiction" (says the opposite of the
# transcript). Omissions are listed separately. Severity: "major" could change care; "minor" would not.
cases = [
    {"transcript": "Doctor: How have your sugars been? Patient: Mostly 140 to 180 fasting. Doctor: Any low sugars? "
                   "Patient: No. Doctor: You're on metformin 500 mg twice daily. HbA1c is 8.2%. Let's increase metformin "
                   "to 1 g twice daily. Patient: I'm allergic to penicillin, by the way. Doctor: Noted. Review in 3 months.",
     "draft": ["Fasting glucose 140-180 mg/dL.", "No hypoglycaemia.", "On metformin 500 mg twice daily.",
               "HbA1c 8.2%.", "Metformin increased to 1 g twice daily.", "Review in 3 months."],
     "errors": [], "omissions": [("penicillin allergy", "major")]},
    {"transcript": "Patient: My feet tingle at night. Doctor: Any wounds? Patient: No. Doctor: Monofilament reduced on both "
                   "feet. HbA1c 7.6%. I'll refer you to podiatry. Blood pressure 134/82.",
     "draft": ["Night-time tingling in both feet.", "Small ulcer on the left heel.", "Reduced monofilament sensation bilaterally.",
               "HbA1c 7.6%; blood pressure 134/82.", "Referred to podiatry."],
     "errors": [(1, "contradiction", "major")], "omissions": []},
    {"transcript": "Doctor: Your kidney test shows eGFR 48. We'll start empagliflozin 10 mg daily. Any urinary symptoms? "
                   "Patient: No. Doctor: Recheck kidney function in 4 weeks.",
     "draft": ["eGFR 48.", "Started empagliflozin 25 mg daily.", "No urinary symptoms.", "Kidney function recheck in 4 weeks."],
     "errors": [(1, "hallucination", "major")], "omissions": []},
    {"transcript": "Patient: I've been more tired lately. Doctor: Any chest pain or breathlessness? Patient: No chest pain. "
                   "Doctor: Haemoglobin 10.9. Let's check iron studies and B12. You're on metformin 1 g twice daily.",
     "draft": ["Tiredness recently.", "Reports chest pain on exertion.", "Haemoglobin 10.9.",
               "Iron studies and B12 requested.", "On metformin 1 g twice daily."],
     "errors": [(1, "contradiction", "major")], "omissions": []},
    {"transcript": "Doctor: Your HbA1c is 6.4%. You're on gliclazide 80 mg. Any dizzy spells? Patient: Yes, twice last week, "
                   "before lunch. Doctor: Let's reduce gliclazide to 40 mg. Patient: OK.",
     "draft": ["HbA1c 6.4%.", "Two episodes of dizziness before lunch last week, likely hypoglycaemia.",
               "Gliclazide reduced to 40 mg.", "Advised to stop driving for 6 weeks."],
     "errors": [(3, "hallucination", "major")], "omissions": []},
    {"transcript": "Doctor: Blood pressure today 148/92. You take ramipril 5 mg. Patient: I sometimes forget it. Doctor: Let's "
                   "use a pill organiser and recheck in 2 weeks. Your cholesterol LDL is 2.1.",
     "draft": ["Blood pressure 148/92 on ramipril 5 mg.", "Occasionally forgets doses.", "Pill organiser suggested.",
               "LDL 2.1.", "Smoker, 10 cigarettes a day.", "Blood pressure recheck in 2 weeks."],
     "errors": [(4, "hallucination", "minor")], "omissions": []},
    {"transcript": "Patient: I'm 30 weeks pregnant. Doctor: Your fasting readings are 5.6 to 6.0. We'll start insulin at "
                   "night, 6 units. Any headaches or swelling? Patient: Some ankle swelling. Doctor: Blood pressure 128/80.",
     "draft": ["30 weeks pregnant.", "Fasting glucose 5.6-6.0 mmol/L.", "Night-time insulin started, 6 units.",
               "No swelling.", "Blood pressure 128/80."],
     "errors": [(3, "contradiction", "minor")], "omissions": []},
    {"transcript": "Doctor: You've lost 4 kg since starting semaglutide. Any nausea? Patient: A little in the first week, "
                   "gone now. Doctor: HbA1c 7.1%. Continue the same dose. Review in 3 months.",
     "draft": ["Weight down 4 kg on semaglutide.", "Mild nausea in the first week, now resolved.", "HbA1c 7.1%.",
               "Dose increased.", "Review in 3 months."],
     "errors": [(3, "contradiction", "major")], "omissions": [("continue same dose", "major")]},
    {"transcript": "Patient: My vision is blurry sometimes. Doctor: Your last eye screen was 2 years ago. I'll book retinal "
                   "screening. HbA1c 9.4%. Doctor: We'll add a second medicine; let's discuss options next visit.",
     "draft": ["Intermittent blurred vision.", "Last eye screening 2 years ago; retinal screening booked.",
               "HbA1c 9.4%.", "Started sitagliptin 100 mg daily.", "Shortness of breath on stairs."],
     "errors": [(3, "hallucination", "major"), (4, "hallucination", "minor")], "omissions": []},
    {"transcript": "Doctor: Any foot problems? Patient: No. Doctor: Pulses present, sensation normal. HbA1c 6.9%. "
                   "Blood pressure 126/78. Keep going as you are. Review in 6 months.",
     "draft": ["No foot problems.", "Foot pulses present, sensation normal.", "HbA1c 6.9%.",
               "Blood pressure 126/78.", "Review in 6 months."],
     "errors": [], "omissions": []},
]
print(f"{len(cases)} cases, {sum(len(c['draft']) for c in cases)} draft sentences")

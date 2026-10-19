# Eight FICTIONAL products for classification practice (no real companies or products).
eu_products = [
    {"name": "DR-Grade", "purpose": "decision support (diagnosis or therapy)",
     "description": "Grades retinal photographs and decides whether the patient is referred to an eye clinic.",
     "worst_consequence": "serious deterioration or surgical intervention"},
    {"name": "DoseWise", "purpose": "decision support (diagnosis or therapy)",
     "description": "Recommends insulin dose changes for hospital inpatients to the prescribing doctor.",
     "worst_consequence": "death or irreversible deterioration"},
    {"name": "DrugInfo", "purpose": "decision support (diagnosis or therapy)",
     "description": "Shows prescribers interaction information for a chosen medicine combination.", "worst_consequence": None},
    {"name": "WardWatch", "purpose": "monitoring physiological processes",
     "description": "Monitors heart rhythm on a cardiac ward and alarms on dangerous arrhythmias.", "vital_immediate_danger": True},
    {"name": "SleepTrend", "purpose": "monitoring physiological processes",
     "description": "Tracks overnight heart rate in people with diabetes for review at the next clinic visit.", "vital_immediate_danger": False},
    {"name": "TriageAI", "purpose": "decision support (diagnosis or therapy)",
     "description": "Sets the priority of patients arriving at an emergency department from their symptoms and vital signs.",
     "worst_consequence": "death or irreversible deterioration"},
    {"name": "PumpLogic", "purpose": "drives a device",
     "description": "Software that controls the dose delivered by a class IIb insulin pump.", "device_class": "IIb"},
    {"name": "RecordView", "purpose": "other",
     "description": "Displays the patient record and lab results to clinicians without interpretation."},
]
print(f"{len(eu_products)} fictional products")

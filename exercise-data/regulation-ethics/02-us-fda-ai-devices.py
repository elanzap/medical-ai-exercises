# Five FICTIONAL decision-support tools, and a fictional retinopathy model, for the CDS and PCCP steps.
cds_tools = [
    {"name": "StatinGuide", "description": "Shows the clinician the guideline recommendation for statin therapy, with the "
     "patient's risk factors and the guideline passage it used, for the clinician to review."},
    {"name": "ECG-Rhythm", "description": "Analyses the ECG waveform from a recorder and labels atrial fibrillation."},
    {"name": "A1cTrend", "description": "Displays a patient's HbA1c results over time with the clinic's target line."},
    {"name": "SepsisNow", "description": "Continuously scores ward patients' vital signs and alerts nurses to start the sepsis "
     "protocol within 1 hour; the score's inputs and weights are not shown."},
    {"name": "PatientDose", "description": "Tells patients at home how much to change their basal insulin from their "
     "fasting glucose readings."},
]

retinopathy_model = ("RetinaNext (fictional): a cleared AI system that grades retinal photographs for referable "
                     "diabetic retinopathy. The manufacturer wants to retrain it every year on new images from its "
                     "partner clinics, including from new camera models.")

# The FDA's AI-enabled device list: download the CSV in a web browser from the FDA's
# "Artificial Intelligence-Enabled Medical Devices" page (the site may block scripted downloads),
# then upload it here (in Colab: the folder icon on the left) as fda_ai_devices.csv.
FDA_LIST_FILE = "fda_ai_devices.csv"
print(f"{len(cds_tools)} fictional CDS tools; expected FDA list file: {FDA_LIST_FILE}")

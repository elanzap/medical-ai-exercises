# Three FICTIONAL incident scenarios (no real patients, clinicians, organisations or products).
scenarios = {
    "A. The missed retinopathy": (
        "A clinic used a cleared AI retinal-screening tool. An image was blurred, and the tool reported 'no referable "
        "retinopathy' instead of 'ungradable'. The vendor had issued a safety notice about blurred images two months "
        "earlier, but the clinic had not installed the update. The nurse did not look at the image. A year later the "
        "patient was found to have proliferative retinopathy."),
    "B. The followed recommendation": (
        "An insulin dosing assistant, approved for use in the hospital, recommended a large basal insulin increase for "
        "a frail older patient who was eating poorly. The doctor accepted it without recording any reasoning; the "
        "patient had a severe hypoglycaemic episode overnight. The tool's instructions said it was not validated for "
        "patients with reduced food intake."),
    "C. The AI-drafted message": (
        "A clinic's AI tool drafted replies to patients' portal messages. A busy clinician sent a draft unchanged that "
        "told a patient their foot 'sounded like a minor blister' and to 'check again next week'. The patient had "
        "described spreading redness. Patients had not been told replies might be AI-drafted."),
}
for title, text in scenarios.items():
    print(title, "-", text, "\n")

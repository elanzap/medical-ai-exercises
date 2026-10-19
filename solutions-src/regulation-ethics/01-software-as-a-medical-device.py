# %% Step 1
import pandas as pd

intended_purpose = {
    "GlucoDiary": "Records and displays a person's self-entered glucose readings and lifestyle data.",
    "RetinaCheck": "Detects referable diabetic retinopathy in retinal photographs to decide referral, without specialist review.",
    "BolusMate": "Calculates mealtime insulin doses for people with type 1 diabetes.",
    "ClinicSlots": "Manages clinic appointment bookings and reminders.",
    "FootScan AI": "Flags patient-taken foot photographs that may show an ulcer, to prioritise clinical review.",
    "PumpCore": "Controls insulin delivery within an insulin pump.",
    "KidneyRisk": "Estimates 5-year kidney-failure risk to inform clinicians' referral decisions.",
    "WellStep": "Encourages general healthy habits; no medical purpose.",
    "NoteDraft": "Drafts clinic letters from consultation audio for the clinician to edit.",
    "HypoAlert": "Predicts and alarms imminent dangerous hypoglycaemia from CGM data.",
}
for p in products:
    print(f"{p['name']:<12} {intended_purpose[p['name']]}")

# %% Step 2
status = {   # (likely status, reason)
    "GlucoDiary": ("Not a device (probably)", "records and displays data without interpreting it; adding 'tells you if your sugar is too high' could change this"),
    "RetinaCheck": ("SaMD", "diagnosis of a disease from images"),
    "BolusMate": ("SaMD", "calculates a treatment dose"),
    "ClinicSlots": ("Not a device", "administrative purpose only"),
    "FootScan AI": ("SaMD", "detects a possible disease (ulcer) in images to direct care"),
    "PumpCore": ("Software in a medical device", "part of the pump, regulated with it"),
    "KidneyRisk": ("SaMD", "predicts a disease outcome to inform clinical decisions"),
    "WellStep": ("Not a device", "general wellness, explicitly no medical condition"),
    "NoteDraft": ("Usually not a device", "documentation; becomes a question if it starts suggesting diagnoses or treatment"),
    "HypoAlert": ("SaMD", "monitors and warns of a dangerous physiological state"),
}
for name, (s, why) in status.items():
    print(f"{name:<12} {s:<30} {why}")

# %% Note
# The deciding question is always the intended purpose in the maker's own claims. NoteDraft and GlucoDiary show the
# grey zone: documentation and data display are usually not devices, but one extra feature that interprets the data
# for a clinical decision can turn them into one. National rules differ in detail, so these are likely outcomes,
# not legal conclusions.

# %% Step 3
assumptions = {   # (situation, significance) - teaching assumptions, stated explicitly
    "RetinaCheck": ("serious", "treat or diagnose"),          # autonomous referral decision
    "BolusMate": ("critical", "drive clinical management"),    # wrong insulin dose can be life-threatening
    "FootScan AI": ("serious", "drive clinical management"),   # triages who gets seen
    "KidneyRisk": ("serious", "inform clinical management"),   # one input among others for referral
    "HypoAlert": ("critical", "drive clinical management"),    # alarm prompts immediate action
}
rows = [{"product": n, "situation": s, "significance": g, "IMDRF category": CATEGORY[(s, g)]}
        for n, (s, g) in assumptions.items()]
print(pd.DataFrame(rows).to_string(index=False))

# %% Note
# Change one assumption and the category moves: if FootScan AI only displayed the photos for the nurse, it would be
# "inform" (category I), and if RetinaCheck's output were reviewed by an ophthalmologist before any decision, it would
# drive rather than diagnose (category II). Writing down the assumptions is the useful part of the exercise.

# %% Step 4
# %% Note
# Rewriting **FootScan AI** so that it is no longer a medical device:
#
# *"Lets patients send photographs of their feet securely to their diabetes clinic, where a clinician looks at every
# image. The app does not analyse the photographs."*
#
# It is now a secure messaging and image-transfer tool: no automated detection, no prioritisation. Is it still
# useful? Partly: patients with poor mobility can share images, but every image now needs human review, so the
# workload benefit of triage is lost. That is the usual trade-off: the clinical value of AI often comes from exactly
# the function that makes it a medical device, which is why the answer is usually to meet the requirements, not to
# avoid them by rewording. Changing claims while keeping the function would not work either: regulators look at
# what the software actually does, and disclaimers do not override it.

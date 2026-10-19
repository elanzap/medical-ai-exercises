# %% Step 1
# %% Note
# **Is FootSafe a medical device in Great Britain?** Very likely yes. Its intended purpose is to detect possible ulcers
# or infection in images and prioritise clinical review, a diagnostic and triage purpose, so it is software as a
# medical device under the UK Medical Devices Regulations 2002 (UK MDR).
#
# **Likely current class.** The UK MDR is based on the old EU directives, under which much standalone software has
# historically fallen into class I (the lesson notes this mismatch with modern AI). A manufacturer might argue for a
# higher class if the software is intended to allow direct diagnosis, which needs regulatory advice. Under the EU
# MDR's Rule 11 the same product would be at least class IIa. The class is expected to change once the MHRA's new
# qualification and classification approach (consultation spring 2027) is in place, so a vendor should plan for
# higher-class evidence now rather than rely on today's class.

# %% Step 2
requirements = {
    "UK MDR registration and conformity": "MHRA registration, UKCA (or CE where accepted) marking, technical file, "
        "post-market surveillance under the strengthened requirements in force since 16 June 2025",
    "DCB0129 (manufacturer)": "clinical risk management by the vendor: a clinical safety officer, hazard log and "
        "clinical safety case report",
    "DCB0160 (each NHS trust)": "the deploying trust's own clinical risk management: local hazards (for example "
        "patients without smartphones, image quality at home), its own safety case and a named clinical safety officer",
    "DTAC": "clinical safety, data protection, technical security, interoperability, usability and accessibility",
    "NICE Evidence Standards Framework": "evidence proportionate to the tool's function: for a tool that informs "
        "triage of a serious condition, expect evidence of accuracy in the intended population and of effect on care "
        "(for example time to treatment), plus economic information",
    "UK GDPR and Data Protection Act 2018": "a data protection impact assessment, a lawful basis and condition for "
        "health data, data-processing agreements, and transparency to patients",
}
for item, needs in requirements.items():
    print(f"- {item}: {needs}")

# %% Step 3
print(national_commission_summary)

# %% Note
# Three recommendations (priority areas) that would most change FootSafe's path:
#
# 1. **Safe iteration (change-control plans).** FootSafe retrains every six months. Today, significant changes may need
#    fresh regulatory work each time; a PCCP route (draft MHRA guidance due by December 2026) would let the vendor agree
#    the retraining process and acceptance criteria once.
# 2. **Modernised qualification and classification.** A new approach could move FootSafe into a higher class, raising
#    the evidence and oversight required before sale.
# 3. **Stronger post-market assurance and assurance for real-world deployment.** More weight on monitoring how the app
#    performs in each service (missed ulcers, false alarms, uptake by different patient groups) would make post-market
#    data collection a condition of use, not an afterthought.

# %% Step 4
questions = [
    "What is the exact intended purpose and the UK regulatory status and class? Can we see the registration and declaration of conformity?",
    "Can we see the DCB0129 clinical safety case report and hazard log, and who is your clinical safety officer?",
    "What was the sensitivity for ulcers and infection in a population like ours, with confidence intervals, and how was the reference standard set?",
    "How does performance vary by skin tone, age, camera model and image quality, and what happens with ungradable images?",
    "What does the podiatry team see, and how are urgent cases (spreading infection, a cold pale foot) handled outside the weekly cycle?",
    "How does six-monthly retraining work, how is each version validated before release, and will we be told what changed?",
    "Where is the data stored and processed, who can access it, and is any patient data used to train your models?",
    "What monitoring data will you give us, how do we report incidents, and how fast will you act on a safety issue?",
    "What happens to the service and the data if we stop using the product or your company stops supporting it?",
]
for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")

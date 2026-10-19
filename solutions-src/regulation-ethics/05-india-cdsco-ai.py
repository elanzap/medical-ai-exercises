# %% Step 1
import pandas as pd

assessment = {   # (medical device software?, likely class, assumption)
    "SugarLog": ("probably not", "-", "records and displays data only; adding interpretation would change this"),
    "DR-Screen": ("yes", "C", "diagnosis of a serious, sight-threatening condition that decides referral"),
    "InsulinCalc": ("yes", "C or D", "drives treatment of a critical situation; a wrong dose can be life-threatening"),
    "RiskBoard": ("yes", "B", "informs, but does not drive, management of a serious condition"),
    "ChatSahayak": ("depends on claims", "B or C if a device", "general diet education may be out of scope; telling patients when "
                    "to seek care is triage, which is a medical purpose"),
}
table = pd.DataFrame([{"tool": k, "device?": v[0], "likely class": v[1], "assumption": v[2]} for k, v in assessment.items()])
print(table.to_string(index=False))

# %% Note
# CDSCO's final guidance on medical device software (July 2026) uses an approach similar to the IMDRF framework: the
# seriousness of the situation and the significance of the output. The classes above are teaching judgements on
# fictional tools; the official guidance and regulatory advice decide a real product's class. Disclaimers such as
# "for information only" cannot override what the software actually does.

# %% Step 2
def licensing_authority(cls, made_in):
    if cls in ("-", ""):
        return "not applicable (not a device)"
    if made_in == "imported":
        return "central licensing authority (CDSCO): import licence"
    if any(c in cls for c in ("C", "D")):
        return "central licensing authority (CDSCO): manufacturing licence"
    return "state licensing authority: manufacturing licence (class A/B)"

for name, (dev, cls, _) in assessment.items():
    made_in = next(t["made_in"] for t in india_tools if t["name"] == name)
    print(f"{name:<12} class {cls:<18} -> {licensing_authority(cls, made_in)}")

# %% Note
# Under the Medical Devices Rules, 2017, manufacturing licences for classes A and B are granted by state licensing
# authorities and for classes C and D centrally by CDSCO; imported devices are licensed centrally whatever their class.
# Check the current forms and portal (the lesson's "check the official text" note applies).

# %% Step 3
# %% Note
# **DR-Screen against the ICMR 2023 ethical principles** (the lesson lists autonomy, safety, data privacy,
# accountability, fairness, trustworthiness and validity, among others):
#
# | Principle | What it means for DR-Screen |
# |---|---|
# | Autonomy | Patients told that AI reads their images; the option of a clinician reading where available |
# | Safety and risk minimisation | Ungradable images and positive results always referred; a fallback when the system is down |
# | Data privacy | Images de-identified for any training; secure storage; DPDP compliance |
# | Accountability | Clear responsibility for missed disease: vendor, health centre and supervising ophthalmologist |
# | Fairness and non-discrimination | Performance tested across age, sex, camera models and image quality in Indian primary care |
# | Trustworthiness and validity | Prospective validation in the intended setting against a graded reference standard |
#
# **What an ethics committee would ask** (if the deployment is studied as research): Is there a protocol with a sample
# size and a reference standard? How is consent obtained, in which languages? Is the tool an investigational device
# needing CDSCO permission for clinical investigation? What happens to participants with a positive result, and how
# quickly? Who owns and can reuse the images? How will harms (missed disease, delayed referral) be monitored and reported?

# %% Step 4
# %% Note
# DR-Screen processes digital personal data (images linked to patients), so the Digital Personal Data Protection Act,
# 2023 applies through the DPDP Rules, 2025. The core obligations take full effect on **13 May 2027**: a clear notice and
# consent (or another lawful ground) for the stated purpose; purpose limitation and data minimisation; reasonable
# security safeguards; breach notification; honouring data principals' rights (access, correction, erasure, grievance
# redressal); and limits on retention. The research exemption described in the privacy lesson applies from the same date.
# Until then, the IT Act 2000 and the 2011 SPDI Rules are the baseline, so a product launched now should be built to the
# 2027 standard from the start.

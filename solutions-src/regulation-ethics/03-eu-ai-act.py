# %% Step 1
import pandas as pd

def classify_software(product):
    """Rule 11 (MDR Annex VIII), plus the implementing rule that software driving a device takes the device's class.
    Simplified learning aid: it assumes the product is already qualified as medical device software."""
    if product["purpose"] == "drives a device":
        return product["device_class"], "implementing rule: same class as the device it drives"
    if product["purpose"] == "decision support (diagnosis or therapy)":
        c = mdr_rule_11(product["purpose"], product.get("worst_consequence"))
        return c, "Rule 11(a): information used for diagnostic or therapeutic decisions"
    if product["purpose"] == "monitoring physiological processes":
        c = mdr_rule_11(product["purpose"], monitors_vital_parameters_with_immediate_danger=product["vital_immediate_danger"])
        return c, "Rule 11(b): monitoring" + (" of vital parameters with possible immediate danger" if c == "IIb" else "")
    return "I", "Rule 11(c): all other software"

rows = []
for p in eu_products:
    cls, basis = classify_software(p)
    rows.append({"product": p["name"], "class": cls, "basis": basis})
classes = pd.DataFrame(rows)
print(classes.to_string(index=False))

# %% Note
# The lesson's sketch already contained simplified monitoring and "all other software" branches; the extension makes
# the sub-rules explicit and adds the rule that software which drives or influences a device falls in that device's
# class. RecordView reaches class I only if it is medical device software at all; software that only displays
# records without interpreting them may not qualify as a device, which is a qualification question that comes first.

# %% Step 2
AI_ACT_HIGH_RISK = [
    "Risk management system across the life cycle, including risks to health, safety and fundamental rights (Art. 9)",
    "Data and data governance: relevant, representative data; examination for possible biases (Art. 10)",
    "Technical documentation (Art. 11), which can be combined with the MDR technical documentation",
    "Automatic logging of events (record-keeping) (Art. 12)",
    "Transparency and instructions for use for deployers (Art. 13)",
    "Design for effective human oversight (Art. 14)",
    "Accuracy, robustness and cybersecurity (Art. 15)",
    "Quality management system covering the AI-specific requirements (Art. 17)",
    "Conformity assessment by the notified body, together with the MDR assessment",
]
high_risk = classes[classes["class"].isin(["IIa", "IIb", "III"])]
print("Products needing a notified body under the MDR, and so high-risk AI systems under Art. 6(1)")
print("(obligations apply from 2 August 2028 after the 2026 Digital Omnibus):", ", ".join(high_risk["product"]))
for r in AI_ACT_HIGH_RISK:
    print(" -", r)
print("\nDeployers (the hospitals) also have duties: use according to instructions, assign trained human oversight,"
      " monitor operation and keep logs, and make sure staff have appropriate AI literacy.")

# %% Note
# Article 6(1) applies only to products that are AI systems as defined in the AI Act: a rule-based lookup such as
# DrugInfo may well not be one, in which case only the MDR applies. The same requirement list applies to every
# high-risk product, but the effort scales with the class: DoseWise and
# TriageAI (class III) need the most evidence. If AI Act requirements duplicate MDR ones, the Commission may limit
# them by implementing act (see the lesson), so check the current position before 2028. The lesson's caution
# applies: this is a learning sketch, not regulatory advice.

# %% Step 3
# %% Note
# How to do this step: open MDCG 2019-11 rev.1 (June 2025) on the European Commission's medical devices guidance page
# and go to its Rule 11 classification examples. For each product, find the example with the closest *intended
# purpose and decision impact* (not the closest technology), and compare its class with yours. Record any difference
# and why: usually the example's purpose differs (for example, "information to support" versus "decides referral"),
# or the worst consequence was judged differently.
#
# We have not reproduced the guidance's examples here; use the official text, because the June 2025 revision added
# and changed examples. Products with no close example (TriageAI, PumpLogic) are the ones most worth discussing with a
# notified body early.

# %% Step 4
# %% Note
# **TriageAI** is the product most likely to meet Annex III of the AI Act directly: Annex III lists AI systems for
# emergency healthcare patient triage. It is also probably medical device software (it uses symptoms and vital signs to
# set clinical priority), so it would be high-risk by both routes, and as a device in class IIa or above it follows
# the Article 6(1) route, with the notified body assessing both. If a version of it were *not* a medical device (for
# example a purely administrative queueing tool), Annex III would still make it high-risk, with its obligations applying
# from 2 December 2027 after the omnibus.
#
# Annex III also covers AI used to evaluate eligibility for essential public services, including healthcare services,
# which could catch, for instance, a tool that decides who is offered a publicly funded diabetes programme. None of
# the other seven products here does that.

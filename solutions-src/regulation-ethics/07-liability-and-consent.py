# %% Step 1
contributions = {
    "A. The missed retinopathy": {
        "clinician": "did not look at a blurred image before accepting a negative result",
        "organisation": "did not act on the vendor's safety notice or install the update; no process for safety notices",
        "manufacturer": "the tool classed an ungradable image as negative, a possible defect, though it issued a warning and a fix",
    },
    "B. The followed recommendation": {
        "clinician": "accepted a recommendation outside the tool's validated use, with no recorded reasoning",
        "organisation": "approved the tool without making its limits visible at the point of use or training staff on them",
        "manufacturer": "limits were stated in the instructions; a question is whether the tool should have warned or refused",
    },
    "C. The AI-drafted message": {
        "clinician": "sent an AI draft unchanged without checking it against the patient's message",
        "organisation": "no review rules, no red-flag checks on drafts, and patients not told about AI drafting",
        "manufacturer": "the drafting tool ignored a red-flag symptom; depends on its claims and intended use",
    },
}
for scenario, parties in contributions.items():
    print(scenario)
    for party, action in parties.items():
        print(f"  {party:<13} {action}")

# %% Note
# In each scenario responsibility is shared, and the clinician's exposure comes from departing from the standard of
# care (not checking, using a tool outside its intended purpose), as Price, Gerke and Cohen describe. In the EU, the
# revised Product Liability Directive treats software, including AI and updates, as a product for products placed on
# the market from 9 December 2026, which matters for the manufacturer's position in scenario A.

# %% Step 2
documentation = {
    "A": ["a log showing the safety notice was received, assessed and the update installed (the organisation)",
          "the screening protocol requiring image-quality review and a record that it was followed (the nurse)",
          "the vendor's safety notice and its distribution record (the manufacturer)"],
    "B": ["a note of why the dose was accepted, including food intake and frailty (the doctor)",
          "the approval record, training materials and how the tool's limits were shown to users (the hospital)",
          "the instructions for use stating the validated population (the manufacturer)"],
    "C": ["the policy on reviewing AI drafts and evidence that staff were trained (the clinic)",
          "the patient-information statement on AI use (the clinic)",
          "an audit trail showing what the AI drafted and what the clinician changed (all parties)"],
}
for s, docs in documentation.items():
    print(f"Scenario {s}:")
    for d in docs:
        print("  -", d)

# %% Step 3
# %% Note
# Example patient-information statement:
#
# > **How we use AI in this diabetes clinic**
# >
# > We use some computer tools that use artificial intelligence (AI) to help our team. For example, a tool helps
# > check photographs of the back of your eye for diabetic eye disease, and a tool helps us draft replies to messages
# > you send us. A member of our clinical team checks the results and every reply before it is sent, and makes all
# > decisions about your care.
# >
# > AI tools can make mistakes, which is why a person always checks. If you would rather your message or images were
# > handled without these tools, tell us and we will do that.
# >
# > Your information is kept securely and is used only to provide your care, as described in our privacy notice. It is
# > not used to train AI tools without your agreement or another lawful basis.
# >
# > If you are worried about your health, do not wait for a message reply: call the clinic, or emergency services if it
# > is urgent.

# %% Step 4
clauses = [
    "Intended purpose, validated population and known limitations, with notice of any change",
    "Regulatory status and the manufacturer's obligations (registration, post-market surveillance, field safety notices)",
    "Performance commitments, and access to evidence: validation studies and subgroup results",
    "Update and change control: advance notice of new versions, what changed, the ability to delay or roll back",
    "Incident and safety reporting both ways, with response times; cooperation in investigations",
    "Data protection: roles (controller/processor), location of processing, security, no use of our data for training "
    "without written agreement, deletion at contract end",
    "Logging and audit: access to logs and outputs needed for monitoring, audit and incident review",
    "Liability and indemnity: allocation of responsibility for product defects, and insurance",
    "Service levels, downtime procedures and support",
    "Exit: data return and transition if the contract ends or the vendor stops supporting the product",
]
for i, c in enumerate(clauses, 1):
    print(f"{i:>2}. {c}")

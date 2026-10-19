# %% Step 1
print(proposal)

# %% Step 2
questions = {
    "Protect human autonomy": [
        "Will patients be told they are talking to AI, and can they reach a person instead?",
        "What patient data does the chatbot see and keep, and have patients agreed to that use?",
    ],
    "Promote well-being, safety and the public interest": [
        "Of the 6% of messages where the chatbot disagreed with nurses, how many urgent messages were under-triaged?",
        "What happens when the on-call nurse does not answer an urgent alert, or a patient describes an emergency in vague words?",
    ],
    "Ensure transparency, explainability and intelligibility": [
        "Can staff see why a message was put in a category, and is the tool's role explained in plain language to patients?",
        "Will the evaluation results, including failures, be published or at least shared with clinicians?",
    ],
    "Foster responsibility and accountability": [
        "Who is clinically responsible for a chatbot answer, and who investigates when something goes wrong?",
        "What does the vendor contract say about liability, incident reporting and audit access?",
    ],
    "Ensure inclusiveness and equity": [
        "How will patients who do not use English, or who have low literacy or no smartphone, get the same service?",
        "Was the 94% agreement measured on messages from older patients, other languages and people with complications?",
    ],
    "Promote AI that is responsive and sustainable": [
        "How will performance be monitored after go-live, and what would make the clinic switch it off?",
        "What happens to nurses' skills and workload if the saving is spent elsewhere and the tool later fails?",
    ],
}
for principle, qs in questions.items():
    print(principle)
    for q in qs:
        print("  -", q)

# %% Step 3
# %% Note
# Which questions are also legal requirements depends on the country. Examples:
#
# - **Telling patients they are talking to AI**: in the EU, the AI Act's transparency obligations for chatbots apply from
#   August 2026 (lesson 7). In India, the Telemedicine Practice Guidelines state that AI tools may assist registered
#   medical practitioners but cannot counsel patients directly, which directly constrains a patient-facing triage bot.
# - **Patient data**: data-protection law (GDPR or UK GDPR; India's DPDP Act, fully in force from 13 May 2027; HIPAA in the
#   US) governs what the chatbot may collect and how.
# - **Safety and regulatory status**: a tool that triages clinical urgency is likely to be medical device software in
#   most jurisdictions (and emergency triage is listed in Annex III of the EU AI Act), so evidence and approval are
#   legal requirements, not just good practice.
# - **Accessibility and equality duties**: for example the UK Equality Act 2010 for public services.
#
# The other questions (publishing failures, workforce effects) are ethical expectations that the law may not yet require.

# %% Step 4
conditions = [
    ("Patients are told they are using AI, can reach a person at any time, and urgent symptoms always trigger "
     "a direct call to emergency services or the clinic.",
     "Check at 6 months: review of the chatbot screens; audit of 100 urgent messages for time to human contact."),
    ("A local silent evaluation on at least 1,000 real messages, graded by nurses, before go-live, with no "
     "under-triage of urgent messages in the test set.",
     "Check at 6 months: monthly random sample of 50 conversations graded by a nurse; every under-triaged urgent case "
     "reviewed as an incident."),
    ("An equal-service plan for non-English speakers and people without smartphones, with outcomes monitored by group.",
     "Check at 6 months: use, response times and triage accuracy reported by language and age group."),
]
for i, (condition, check) in enumerate(conditions, 1):
    print(f"Condition {i}: {condition}\n  {check}\n")

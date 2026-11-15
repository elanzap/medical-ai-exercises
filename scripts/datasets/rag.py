"""Dataset: Synthetic Clinical RAG Evaluation Set (synthetic-clinical-rag-evaluation).

Seven synthetic teaching protocols for a diabetes clinic, split into sections; questions with gold sections in
four phrasings (clinical, lay, abbreviation, misspelled); and answer claims labelled for grounding checks.
Extends the synthetic protocol used across the course's RAG lessons. Teaching documents, NOT clinical guidance.

Usage: uv run --python 3.12 --with numpy --with pandas --with matplotlib --with scikit-learn scripts/datasets/rag.py --user rakuelli
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_common as K  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SLUG = "synthetic-clinical-rag-evaluation"
TITLE = "Synthetic Clinical RAG Evaluation Set"
SUBTITLE = "Synthetic clinic protocols, paraphrased questions and labelled claims"
OUT = K.BUILD / "datasets" / SLUG
ASSETS = K.ROOT / "assets" / "kaggle" / SLUG
NOTE = "Synthetic teaching document for evaluating retrieval systems. Not clinical guidance; do not use for patient care."

# Each section: (section_id, heading, text, questions, claims)
#   questions: list of (clinical, lay, abbreviation, misspelled) phrasings, all answered by this section
#   claims: list of (claim, label) where label is supported | contradicts_source | not_in_source
DOCS = [
    ("glucose", "Monitoring glucose control", [
        ("glucose-1", "HbA1c testing", "Check HbA1c every 3 months after any change in treatment. Once targets are met and treatment is stable, check HbA1c every 6 months. A common target for many non-pregnant adults is below 7%, adjusted for the individual: tighter for some younger people, less strict for older adults or anyone prone to low glucose.",
         [("How often should HbA1c be checked after a change in treatment?", "How often do I need the 3-month sugar test after my tablets change?", "HbA1c freq after tx change?", "how often shud hba1c be cheked after changing treatmnt"),
          ("What HbA1c target suits most non-pregnant adults?", "What should my average sugar number be?", "Target HbA1c non-pregnant adults?", "wat is the normal hba1c targt for adults")],
         [("HbA1c is checked every 3 months after a change in treatment.", "supported"), ("Once stable, HbA1c is checked every 6 months.", "supported"),
          ("HbA1c is checked every month after a change in treatment.", "contradicts_source"), ("The HbA1c target is below 5.7% for all adults.", "contradicts_source"),
          ("HbA1c should be checked before every meal.", "not_in_source")]),
        ("glucose-2", "Continuous glucose monitoring targets", "For people using continuous glucose monitoring, aim for more than 70% of readings between 70 and 180 mg/dL, less than 4% below 70 mg/dL and less than 1% below 54 mg/dL. Review the ambulatory glucose profile at each visit, and adjust treatment if lows are frequent.",
         [("What time-in-range target applies to continuous glucose monitoring?", "How much of the day should my sensor show normal sugar?", "CGM TIR target?", "wat time in rang shud my cgm show"),
          ("What is the limit for time below 70 mg/dL on CGM?", "How often is it OK for my sensor to show low sugar?", "CGM TBR <70 limit?", "how much time below 70 is ok on the senser")],
         [("More than 70% of CGM readings should be between 70 and 180 mg/dL.", "supported"), ("Less than 4% of readings should be below 70 mg/dL.", "supported"),
          ("Up to 20% of readings below 70 mg/dL is acceptable.", "contradicts_source"), ("CGM sensors must be replaced every 3 days.", "not_in_source")]),
    ]),
    ("kidney", "Kidney function and metformin", [
        ("kidney-1", "Kidney tests", "Check eGFR and urine albumin-to-creatinine ratio at least once a year in everyone with diabetes, and more often when kidney disease is present. A urine albumin-to-creatinine ratio of 30 mg/g or more on two of three tests over 3 to 6 months indicates albuminuria.",
         [("How often should urine albumin and eGFR be checked in diabetes?", "How often do I need the kidney urine and blood tests?", "UACR and eGFR frequency in DM?", "how offen is the kidny urine test done"),
          ("What urine albumin-to-creatinine ratio indicates albuminuria?", "What urine protein number means my kidneys are leaking?", "UACR cut-off for albuminuria?", "wat uacr number means albumin in urin")],
         [("eGFR and urine albumin are checked at least once a year.", "supported"), ("A UACR of 30 mg/g or more on repeat tests indicates albuminuria.", "supported"),
          ("Kidney tests are only needed every 5 years.", "contradicts_source"), ("A single UACR result above 300 mg/g confirms kidney failure.", "not_in_source")]),
        ("kidney-2", "Metformin and kidney function", "Do not start metformin if eGFR is below 45. If eGFR falls below 30 during treatment, stop metformin. Between 30 and 44, review the dose and the reason for continuing.",
         [("When should metformin not be started because of kidney function?", "Is the diabetes pill safe with weak kidneys?", "Metformin eGFR threshold to start?", "can i start metformn if my kidnys are weak"),
          ("At what eGFR should metformin be stopped?", "When do I have to stop my sugar tablet because of my kidneys?", "Stop metformin eGFR?", "wen to stop metfromin kidney")],
         [("Metformin should not be started if eGFR is below 45.", "supported"), ("Metformin is stopped if eGFR falls below 30.", "supported"),
          ("Metformin can be started at any eGFR above 15.", "contradicts_source"), ("Start metformin if eGFR is below 45.", "contradicts_source"),
          ("Metformin must be taken with food to protect the kidneys.", "not_in_source")]),
    ]),
    ("foot", "Foot care", [
        ("foot-1", "Foot examination", "Examine both feet at every review. Test sensation with a 10 g monofilament at least once a year, and check pulses, skin and footwear. People with loss of sensation, previous ulcers or poor circulation are at high risk and need review every 1 to 3 months.",
         [("How often should foot sensation be tested with a monofilament?", "How often should someone check if my feet have lost feeling?", "Monofilament test frequency?", "how often is the monofilment foot test"),
          ("How often should high-risk feet be reviewed?", "My feet are high risk; how often should they be checked?", "High-risk foot review interval?", "how ofen high risk feet shud be seen")],
         [("Sensation is tested with a 10 g monofilament at least once a year.", "supported"), ("High-risk feet are reviewed every 1 to 3 months.", "supported"),
          ("Monofilament testing is only needed once every 5 years.", "contradicts_source"), ("Everyone with diabetes should wear custom shoes.", "not_in_source")]),
        ("foot-2", "When to refer urgently", "Refer urgently, the same day, if there is an ulcer with spreading redness, a suspected infection of the bone, a hot swollen foot, or a cold, pale or blue foot. A non-healing wound after 2 weeks of treatment also needs specialist review.",
         [("When is a diabetic foot problem an emergency?", "When is my foot wound an emergency?", "Urgent referral criteria diabetic foot?", "wen is a diabetic foot ulcr an emergancy"),
          ("When should a non-healing foot wound be referred?", "My foot sore isn't healing; when should I see a specialist?", "Non-healing DFU referral timing?", "foot wound not heeling when to refer")],
         [("A cold, pale foot needs urgent same-day referral.", "supported"), ("A wound that has not healed after 2 weeks needs specialist review.", "supported"),
          ("A foot ulcer with spreading redness can wait until the next routine visit.", "contradicts_source"), ("All foot ulcers need surgery.", "not_in_source")]),
    ]),
    ("hypo", "Low blood glucose", [
        ("hypo-1", "Treating a low", "A glucose below 70 mg/dL is low. Treat with 15 g of fast-acting carbohydrate, such as glucose tablets or juice, and recheck after 15 minutes. Repeat if glucose is still below 70 mg/dL, then eat a snack or meal.",
         [("How is a glucose below 70 mg/dL treated?", "My sugar dropped really low, what now?", "Hypo treatment 15/15?", "my sugr is low wat do i do"),
          ("How many grams of carbohydrate treat a low glucose?", "How much sugar should I take for a low?", "Grams carb for hypo?", "how many grams of carbs for a hypo")],
         [("Treat a low with 15 g of fast-acting carbohydrate.", "supported"), ("Recheck glucose 15 minutes after treating a low.", "supported"),
          ("Treat a low with 50 g of carbohydrate.", "contradicts_source"), ("Recheck glucose 2 hours after treating a low.", "contradicts_source"), ("Chocolate is the best treatment for a low.", "not_in_source")]),
        ("hypo-2", "Severe lows and review", "If the person cannot swallow safely or is unconscious, do not give anything by mouth: give glucagon if prescribed and call emergency services. After any low, review medicines that may have caused it, especially insulin and sulfonylureas.",
         [("What should be done if someone with a low cannot swallow?", "What do I do if my father is too drowsy to eat during a low?", "Severe hypo unable to swallow management?", "low sugr and cant swallow wat to do"),
          ("Which medicines should be reviewed after a low glucose?", "Which tablets could be causing my lows?", "Meds to review after hypo?", "wich medcines cause low sugar")],
         [("Do not give anything by mouth if the person cannot swallow safely.", "supported"), ("Insulin and sulfonylureas should be reviewed after a low.", "supported"),
          ("Give sugary drinks to an unconscious person.", "contradicts_source"), ("Metformin alone commonly causes severe lows.", "not_in_source")]),
    ]),
    ("eye", "Eye screening", [
        ("eye-1", "When to screen", "Arrange a dilated retinal examination at diagnosis of type 2 diabetes, and within 5 years of diagnosis of type 1 diabetes in adults. If there is no retinopathy and glucose is well controlled, screen every 1 to 2 years; otherwise screen at least yearly.",
         [("When should eye screening start in type 2 diabetes?", "When should I first get my eyes checked after being told I have diabetes?", "Retinal screening start T2DM?", "wen shud eye check start type 2 diabetis"),
          ("How often is retinal screening needed when there is no retinopathy?", "If my eyes are fine, how often do I need the eye test?", "Screening interval no DR?", "how often eye test if no retinopathy")],
         [("Retinal screening is arranged at diagnosis of type 2 diabetes.", "supported"), ("Without retinopathy and with good control, screening is every 1 to 2 years.", "supported"),
          ("Eye screening in type 2 diabetes starts 10 years after diagnosis.", "contradicts_source"), ("Everyone with diabetes needs laser treatment.", "not_in_source")]),
        ("eye-2", "Urgent eye symptoms", "Sudden loss of vision, a curtain over the vision, or many new floaters need same-day eye assessment. Pregnancy increases the risk of retinopathy progressing, so women with diabetes should have an eye check before or early in pregnancy.",
         [("Which eye symptoms need same-day assessment in diabetes?", "When is a problem with my eyes an emergency?", "Urgent eye symptoms DM?", "wich eye problems are emergancy diabetis"),
          ("When should women with diabetes have an eye check in pregnancy?", "I have diabetes and I'm pregnant; when do I need an eye test?", "DR screening in pregnancy timing?", "eye check in pregnancey with diabetes when")],
         [("Sudden loss of vision needs same-day eye assessment.", "supported"), ("Women with diabetes should have an eye check before or early in pregnancy.", "supported"),
          ("New floaters can wait for the next annual check.", "contradicts_source"), ("Reading glasses prevent diabetic retinopathy.", "not_in_source")]),
    ]),
    ("heart", "Blood pressure and cholesterol", [
        ("heart-1", "Blood pressure", "Measure blood pressure at every visit. For most adults with diabetes the target is below 130/80 mmHg, if it can be reached safely. Confirm high readings with home or repeat measurements. An ACE inhibitor or ARB is preferred when there is albuminuria.",
         [("What blood pressure target applies to most adults with diabetes?", "What should my BP be with diabetes?", "BP target DM?", "wat shud my blod pressure be with diabetis"),
          ("Which blood pressure medicines are preferred when there is albuminuria?", "Which BP tablet is best if I have protein in my urine?", "Antihypertensive with albuminuria?", "wich bp medcine for protien in urine")],
         [("The blood pressure target for most adults with diabetes is below 130/80 mmHg.", "supported"), ("An ACE inhibitor or ARB is preferred with albuminuria.", "supported"),
          ("The target is below 160/100 mmHg.", "contradicts_source"), ("Blood pressure only needs checking once a year.", "contradicts_source"), ("Beta blockers are first choice for everyone with diabetes.", "not_in_source")]),
        ("heart-2", "Cholesterol", "Check a lipid profile at diagnosis and regularly after that. Offer a statin to adults aged 40 to 75 with diabetes. For most of them aim for an LDL cholesterol below 70 mg/dL, and lower after a heart attack or stroke.",
         [("Who with diabetes should be offered a statin?", "Do I need a cholesterol tablet because of my diabetes?", "Statin indication DM age?", "who neds a statin with diabetis"),
          ("What LDL target applies to most adults with diabetes?", "What should my bad cholesterol be?", "LDL target DM?", "wat ldl shud i aim for")],
         [("Adults aged 40 to 75 with diabetes are offered a statin.", "supported"), ("Most aim for an LDL below 70 mg/dL.", "supported"),
          ("Statins are only for people over 80.", "contradicts_source"), ("Eggs must be avoided completely in diabetes.", "not_in_source")]),
    ]),
    ("sickday", "Sick-day rules", [
        ("sickday-1", "Monitoring when unwell", "When unwell, check glucose at least every 4 hours, drink fluids regularly and keep eating small amounts of carbohydrate if possible. Never stop basal insulin, even if not eating; the dose may need adjusting with the care team.",
         [("How often should glucose be checked during illness?", "How often should I check my sugar when I have fever?", "Glucose monitoring freq sick day?", "how often check sugr when sick"),
          ("Should basal insulin be stopped when someone is not eating?", "I'm not eating because I'm sick; should I skip my long-acting insulin?", "Basal insulin when NPO/unwell?", "shud i stop my insulin if i dont eat")],
         [("Check glucose at least every 4 hours when unwell.", "supported"), ("Never stop basal insulin when unwell.", "supported"),
          ("Stop all insulin when not eating.", "contradicts_source"), ("Fasting is recommended during a fever.", "contradicts_source"), ("Antibiotics should be started for every fever.", "not_in_source")]),
        ("sickday-2", "Medicines to pause and when to get help", "During vomiting, diarrhoea or dehydration, pause metformin and SGLT2 inhibitors until eating and drinking normally, and restart them after 24 to 48 hours of normal intake. Seek urgent help for persistent vomiting, glucose that stays above 300 mg/dL, moderate or large ketones, or drowsiness.",
         [("Which diabetes medicines should be paused during vomiting or dehydration?", "Which sugar tablets should I stop while I'm vomiting?", "Meds to hold sick day (metformin/SGLT2i)?", "wich tablets to stop when vomitting"),
          ("When should someone with diabetes seek urgent help during illness?", "When should I go to hospital when I'm sick with diabetes?", "Red flags sick day DM?", "wen to go to hosptal when sick with sugar")],
         [("Pause metformin and SGLT2 inhibitors during vomiting or dehydration.", "supported"), ("Seek urgent help for moderate or large ketones.", "supported"),
          ("Keep taking SGLT2 inhibitors during vomiting.", "contradicts_source"), ("Restart metformin as soon as vomiting starts.", "contradicts_source"), ("Ginger tea cures diabetic ketoacidosis.", "not_in_source")]),
    ]),
]
PHRASINGS = ["clinical", "lay", "abbreviation", "misspelled"]


def build(seed=21):
    rng = np.random.default_rng(seed)
    docs, sections, questions, claims = [], [], [], []
    all_sections = [s[0] for _, _, secs in DOCS for s in secs]
    for doc_id, title, secs in DOCS:
        body = [f"# {title} (synthetic teaching protocol)", "", NOTE, ""]
        for sid, heading, text, qs, cl in secs:
            body += [f"## {heading}", text, ""]
            sections.append({"section_id": sid, "doc_id": doc_id, "heading": heading, "text": text, "n_words": len(text.split())})
            for qi, group in enumerate(qs, 1):
                for ph, q in zip(PHRASINGS, group):
                    questions.append({"question_id": f"Q-{sid}-{qi}-{ph[:3]}", "question": q, "phrasing": ph, "gold_section_id": sid, "gold_doc_id": doc_id})
            for ci, (claim, label) in enumerate(cl, 1):
                claims.append({"claim_id": f"C-{sid}-{ci}", "claim": claim, "cited_section_id": sid, "label": label, "true_section_id": sid if label != "not_in_source" else ""})
                if label == "supported" and rng.random() < 0.5:  # same true claim, cited to the wrong section
                    wrong = rng.choice([s for s in all_sections if s != sid])
                    claims.append({"claim_id": f"C-{sid}-{ci}w", "claim": claim, "cited_section_id": wrong, "label": "wrong_citation", "true_section_id": sid})
        docs.append({"doc_id": doc_id, "title": f"{title} (synthetic teaching protocol)", "text": "\n".join(body).strip() + "\n"})
    return pd.DataFrame(docs), pd.DataFrame(sections), pd.DataFrame(questions), pd.DataFrame(claims)


TABLES_DOC = {
    "documents.csv": ("Seven synthetic teaching protocols for a diabetes clinic, as full Markdown text (also as separate files in documents/).", {
        "doc_id": ("id", "Document identifier"), "title": ("string", "Document title"), "text": ("string", "Full Markdown text, with a note that it is a synthetic teaching document")}),
    "sections.csv": ("The protocols split at their headings: the retrieval units.", {
        "section_id": ("id", "Section identifier (gold label for questions)"), "doc_id": ("string", "Document it belongs to"), "heading": ("string", "Section heading"),
        "text": ("string", "Section text"), "n_words": ("integer", "Number of words")}),
    "questions.csv": ("Questions with the section that answers them, each in four phrasings: clinical, lay, abbreviation and misspelled.", {
        "question_id": ("id", "Question identifier"), "question": ("string", "Question text"), "phrasing": ("string", "clinical, lay, abbreviation or misspelled"),
        "gold_section_id": ("string", "Section that answers the question"), "gold_doc_id": ("string", "Document that answers the question")}),
    "answer_claims.csv": ("Claims an assistant might make, each citing a section, labelled for grounding evaluation.", {
        "claim_id": ("id", "Claim identifier"), "claim": ("string", "Claim text"), "cited_section_id": ("string", "Section the claim cites"),
        "label": ("string", "supported, wrong_citation (true but cites the wrong section), contradicts_source or not_in_source"),
        "true_section_id": ("string", "Section that actually supports the claim (empty if none)")}),
}


def charts(sections, questions, claims):
    ASSETS.mkdir(parents=True, exist_ok=True)
    K.style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), dpi=110)
    fig.subplots_adjust(top=0.72, bottom=0.14, wspace=0.35, left=0.07, right=0.97)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    corpus = sections.heading + " " + sections.text
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(corpus)
    best = cosine_similarity(vec.transform(questions.question), vec.transform(corpus)).argmax(axis=1)
    hit = sections.section_id.values[best] == questions.gold_section_id.values
    r1 = pd.Series(hit).groupby(questions.phrasing.values).mean().mul(100).reindex(PHRASINGS)
    per = r1
    ax[0].bar(["Clinical", "Lay", "Abbreviation", "Misspelled"], r1.values, width=0.5, color=[K.BLUE, K.ORANGE, K.BLUE, K.BLUE])
    for i, v in enumerate(r1.values):
        ax[0].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9.5)
    ax[0].set(ylabel="Top result is the right section (%)", ylim=(0, 100))
    K.titled(ax[0], "Keyword search by question phrasing", "TF-IDF baseline; lay language is hardest")
    lab = claims.label.value_counts().reindex(["supported", "wrong_citation", "contradicts_source", "not_in_source"])
    cols = [K.BLUE, K.MUTED, K.ORANGE, K.MUTED]
    ax[1].barh([x.replace("_", " ") for x in lab.index][::-1], lab.values[::-1], height=0.5, color=cols[::-1])
    ax[1].grid(axis="y", visible=False)
    ax[1].grid(axis="x", visible=True)
    for i, v in enumerate(lab.values[::-1]):
        ax[1].text(v + 0.3, i, str(v), va="center", fontsize=9)
    ax[1].set(xlabel="Claims")
    K.titled(ax[1], "Answer claims by label", "Contradictions are the errors a grounding check must catch")
    fig.text(0.01, 0.95, "Synthetic clinical RAG evaluation set", fontsize=14, color=K.INK, ha="left")
    fig.text(0.01, 0.9, "Synthetic teaching protocols; not clinical guidance", fontsize=9.5, color=K.INK_2, ha="left")
    K.source_note(fig, "Source: questions.csv and answer_claims.csv, Synthetic Clinical RAG Evaluation Set (Kaggle)")
    fig.savefig(ASSETS / "overview.png")
    plt.close(fig)

    def draw(axc):
        axc.bar(range(len(per)), per.values, width=0.55, color=K.BLUE_LIGHT)
        axc.set_xlim(-0.7, len(per) - 0.3)
        axc.set_ylim(0, 100)

    K.cover(OUT / "dataset-cover-image.png", ("Synthetic Clinical", "RAG Evaluation Set"), "retrieval and grounding tests", "7 PROTOCOLS  |  QUESTIONS  |  LABELLED CLAIMS", draw)


def description(user, docs, sections, questions, claims, lessons):
    lab = claims.label.value_counts()
    ex = questions[questions.gold_section_id == "kidney-2"].head(4)
    exq = "\n".join(f"| {r.phrasing} | {r.question} |" for r in ex.itertuples())
    return f"""# {TITLE}

A small, carefully labelled benchmark for **retrieval-augmented generation (RAG) in medicine**: **7 synthetic diabetes-clinic protocols** split into **{len(sections)} sections**, **{len(questions)} questions** with the section that answers them, each asked **four ways** (clinical wording, lay language, abbreviations, misspellings), and **{len(claims)} answer claims** labelled as supported, wrong citation, contradicts the source, or not in the source. It accompanies the free [Medical AI course]({K.SITE}/medical-ai) by [{K.AUTHOR}]({K.SITE}/about), a physician and diabetologist.

**Teaching documents, not clinical guidance.** The protocols were written for evaluating retrieval systems; they simplify real guidance and must not be used for patient care (CC BY 4.0).

![Overview of the RAG evaluation set]({K.RAW}/assets/kaggle/{SLUG}/overview.png)

## What's inside

| File | Rows | What it is |
|---|---|---|
| `documents.csv` and `documents/*.md` | {len(docs)} | The protocols as full text |
| `sections.csv` | {len(sections)} | The retrieval units: one row per section |
| `questions.csv` | {len(questions)} | Questions with gold section and phrasing type |
| `answer_claims.csv` | {len(claims)} | Claims with cited section and grounding label |
| `data_dictionary.csv` | | Every column, with type and description |

## One question, four ways

| Phrasing | Question (gold section: Metformin and kidney function) |
|---|---|
{exq}

## Claim labels

| Label | Count | Meaning |
|---|---|---|
| supported | {lab.get('supported', 0)} | True, and the cited section says so |
| wrong citation | {lab.get('wrong_citation', 0)} | True, but cited to the wrong section |
| contradicts source | {lab.get('contradicts_source', 0)} | The cited section says the opposite (often a changed number) |
| not in source | {lab.get('not_in_source', 0)} | Plausible-sounding, but nothing in the protocols supports it |

## Ideas for what to do with it

- **Retrieval:** compare keyword (TF-IDF, BM25) and embedding retrieval with recall@k and MRR, and see how each phrasing type changes the score.
- **Chunking:** re-chunk `documents.csv` by size or sentence and check whether gold sections are still retrieved.
- **Grounding:** build a checker that flags unsupported and contradicting claims; the contradictions often differ from the source only by a number.
- **End-to-end RAG:** generate answers with a model and score them on retrieval, faithfulness and citations.

A ready-to-run [starter notebook](https://www.kaggle.com/code/{user}/synthetic-clinical-rag-evaluation-starter) compares TF-IDF retrieval across the four phrasings and scores a simple claim checker.

{K.links_block(lessons)}
{K.related_datasets(user, SLUG)}
## Licence and citation

Data and text: CC BY 4.0. Code: MIT. Please cite as: *Ram Kumar G. Synthetic Clinical RAG Evaluation Set. Kaggle, 2026. {K.SITE}/medical-ai*

{K.DISCLAIMER}
"""


STARTER = [
    ("md", "## Load the data"),
    ("code", """
import pandas as pd, numpy as np, re
sections = pd.read_csv(f'{DATA}/sections.csv')
questions = pd.read_csv(f'{DATA}/questions.csv')
claims = pd.read_csv(f'{DATA}/answer_claims.csv')
print(len(sections), 'sections;', len(questions), 'questions;', len(claims), 'claims')
questions.head(8)
"""),
    ("md", "## Keyword retrieval, by phrasing\n\nTF-IDF retrieval with recall@1 and recall@3. Lay language and misspellings are where keyword search struggles; try an embedding model as the next step."),
    ("code", """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit(sections.heading + ' ' + sections.text)
S = vec.transform(sections.heading + ' ' + sections.text)
sim = cosine_similarity(vec.transform(questions.question), S)
ranked = np.argsort(-sim, axis=1)
gold = questions.gold_section_id.map({s: i for i, s in enumerate(sections.section_id)}).values
questions['r1'] = ranked[:, 0] == gold
questions['r3'] = [g in r[:3] for g, r in zip(gold, ranked)]
questions.groupby('phrasing')[['r1', 'r3']].mean().round(2)
"""),
    ("md", "## A simple grounding check\n\nFlag a claim as supported only if its numbers all appear in the cited section and most of its content words do. Look at which labels it gets wrong."),
    ("code", """
text = dict(zip(sections.section_id, sections.text))
STOP = set('the a an is are be of to and or if at in for with it its by on as can'.split())
def words(s): return {w for w in re.findall(r'[a-z0-9]+', s.lower()) if w not in STOP}
def nums(s): return set(re.findall(r'\\d+(?:\\.\\d+)?', s))
def check(claim, sid):
    src = text[sid]
    if not nums(claim) <= nums(src): return 'not supported'
    return 'supported' if len(words(claim) & words(src)) / max(len(words(claim)), 1) >= 0.6 else 'not supported'
claims['checker'] = [check(c, s) for c, s in zip(claims.claim, claims.cited_section_id)]
pd.crosstab(claims.label, claims.checker)
"""),
]


def starter(user, lessons):
    cells = [K.md_cell(f"""
# Synthetic Clinical RAG Evaluation Set: starter notebook

> Starter notebook for the dataset **[{TITLE}](https://www.kaggle.com/datasets/{user}/{SLUG})**, part of the free **[Medical AI course]({K.SITE}/medical-ai)** by [{K.AUTHOR}]({K.SITE}/about).
>
> - The protocols are **synthetic teaching documents**, not clinical guidance.
> - Lessons behind this data: [{lessons[0][0]}]({lessons[0][1]}) and [{lessons[1][0]}]({lessons[1][1]}).
"""), K.code_cell(K.find_data_snippet("sections.csv"))]
    for kind, src in STARTER:
        cells.append(K.md_cell(src) if kind == "md" else K.code_cell(src))
    cells.append(K.md_cell(f"""
## Continue learning

- The RAG track: [{K.SITE.removeprefix('https://')}/medical-ai/rag]({K.SITE}/medical-ai/rag)
- The whole course: [{K.SITE.removeprefix('https://')}/medical-ai]({K.SITE}/medical-ai)
- About the author: [{K.AUTHOR}]({K.SITE}/about)

---

{K.DISCLAIMER}
"""))
    K.write_starter(K.BUILD / "starters" / SLUG, user, "synthetic-clinical-rag-evaluation-starter", "Synthetic Clinical RAG Evaluation: Starter", SLUG, cells)


def main(user):
    course = json.loads((K.ROOT / "kaggle-course.json").read_text())
    url = {l["slug"]: (l["title"], l["url"]) for l in course["lessons"]}
    lessons = [url[s] for s in ("retrieval-evaluation", "citations-and-grounding", "why-rag", "chunking-clinical-guidelines", "embeddings-and-vector-stores", "guideline-qa-assistant", "end-to-end-rag-evaluation", "diabetes-guideline-rag-assistant")]
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "documents").mkdir(parents=True)
    docs, sections, questions, claims = build()
    for r in docs.itertuples():
        (OUT / "documents" / f"{r.doc_id}.md").write_text(r.text)
    docs.to_csv(OUT / "documents.csv", index=False)
    sections.to_csv(OUT / "sections.csv", index=False)
    questions.to_csv(OUT / "questions.csv", index=False)
    claims.to_csv(OUT / "answer_claims.csv", index=False)
    pd.DataFrame([(f, c, t, d) for f, (_, cols) in TABLES_DOC.items() for c, (t, d) in cols.items()], columns=["file", "column", "type", "description"]).to_csv(OUT / "data_dictionary.csv", index=False)
    charts(sections, questions, claims)
    tables = {f: {"description": d, "columns": c} for f, (d, c) in TABLES_DOC.items()}
    tables["data_dictionary.csv"] = {"description": "Every column of every CSV in this dataset, with its type and description.", "columns": {"file": ("string", "CSV file name"), "column": ("string", "Column name"), "type": ("string", "Column type"), "description": ("string", "What the column holds")}}
    K.write_metadata(OUT, {
        "title": TITLE, "subtitle": SUBTITLE, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["healthcare", "medicine", "nlp", "education", "text"],
        "description": description(user, docs, sections, questions, claims, lessons),
        "resources": K.resources(tables),
        "userSpecifiedSources": f"Written by {K.AUTHOR} ({K.AUTHOR_DEGREE}) for the Medical AI course at {K.SITE}/medical-ai; built by scripts/datasets/rag.py in {K.REPO}. The protocols are synthetic teaching documents that simplify widely published diabetes care recommendations (for example the American Diabetes Association Standards of Care); they are not clinical guidance.",
        "expectedUpdateFrequency": "annually",
        "image": "dataset-cover-image.png",
    })
    starter(user, lessons)
    print(len(docs), "docs,", len(sections), "sections,", len(questions), "questions,", len(claims), "claims", claims.label.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    main(ap.parse_args().user)

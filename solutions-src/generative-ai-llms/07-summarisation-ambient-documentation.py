# %% Step 1
import pandas as pd

def edited_flags(p):
    """A sentence counts as edited if it changed at all, or was removed or added."""
    return [difflib.SequenceMatcher(None, d, f).ratio() < 1.0 for d, f in zip(p["draft"], p["final"])]

rows = []
for k, p in enumerate(pairs, start=1):
    flags = edited_flags(p)
    rows.append({"note": k, "sentences": len(flags), "edited": sum(flags), "share edited": sum(flags) / len(flags)})
per_note = pd.DataFrame(rows).set_index("note")
print(per_note.round(2).T.to_string())
print(f"\nOverall: {per_note.edited.sum()} of {per_note.sentences.sum()} sentences edited "
      f"({per_note.edited.sum() / per_note.sentences.sum():.0%}); {(per_note.edited == 0).sum()} notes signed unchanged")

# %% Step 2
import re

MEDICINE_WORDS = {"metformin", "gliclazide", "semaglutide", "empagliflozin", "ramipril", "atorvastatin", "insulin"}
SAFETY_WORDS = ("seek", "attend", "call", "urgent", "if ", "sick-day", "pause", "break the fast", "glucagon", "blood test")

def classify(draft, final):
    """A rough rule-based guess at the edit type, to compare with the clinician labels."""
    if final == "":
        return FAB
    if draft == "":
        return SAFE if any(w in final.lower() for w in SAFETY_WORDS) else S
    nums_d, nums_f = re.findall(r"\d+(?:\.\d+)?", draft), re.findall(r"\d+(?:\.\d+)?", final)
    words_d = set(re.findall(r"[a-z]+", draft.lower()))
    words_f = set(re.findall(r"[a-z]+", final.lower()))
    changed_meaning = (nums_d != nums_f and nums_d) or ((words_d ^ words_f) & (MEDICINE_WORDS | {"stopped", "continued", "once", "twice", "good", "misses"}))
    return FC if changed_meaning else S

results = []
for p in pairs:
    for d, f, label in zip(p["draft"], p["final"], p["labels"]):
        if label is not None:
            results.append({"label": label, "rule": classify(d, f)})
edits = pd.DataFrame(results)
print("Clinician labels:")
print(edits.label.value_counts().to_string())
print("\nRule-based classification against the labels:")
print(pd.crosstab(edits.label, edits.rule).to_string())
print(f"\nAgreement {(edits.label == edits.rule).mean():.0%}")

# %% Note
# Only 9 of the 30 edits are style. The other 21 matter clinically: wrong numbers, a dose frequency, a
# medicine continued instead of stopped, safety-netting the draft left out, and content the clinician removed
# because it never happened. The simple rules agree with most labels because this synthetic data was written to be clear-cut;
# real edits are messier, and an audit should have a clinician label at least a sample of them.

# %% Step 3
FINDING_CUES = ("examination", "examined", "pulses", "palpable", "monofilament", "foot check", "lipohypertrophy",
                "unremarkable", "advised", "advice", "counselled", "told to")
STOP = {"with", "from", "that", "this", "were", "have", "until", "today", "given", "about", "after"}
NORMALISE = {"feet": "foot"}

def stems(text):
    words = [NORMALISE.get(w, w) for w in re.findall(r"[a-z]+", text.lower())]
    return {w[:4] for w in words if len(w) >= 4 and w not in STOP}

def unsupported_findings(p):
    """Draft sentences that record an examination or advice whose content words are absent from the transcript."""
    source = stems(p["transcript"])
    cue_stems = set().union(*(stems(c) for c in FINDING_CUES))
    flagged = []
    for i, sentence in enumerate(p["draft"]):
        if sentence and any(c in sentence.lower() for c in FINDING_CUES):
            content = stems(sentence) - cue_stems
            if content and not (content & source):
                flagged.append(i)
    return flagged

found = fabricated = false_alarms = 0
for k, p in enumerate(pairs, start=1):
    truth = {i for i, label in enumerate(p["labels"]) if label == FAB}
    flags = set(unsupported_findings(p))
    fabricated += len(truth); found += len(flags & truth); false_alarms += len(flags - truth)
    for i in sorted(flags | truth):
        status = "found" if i in flags & truth else ("MISSED" if i in truth else "false alarm")
        print(f"note {k:>2} [{status:<11}] {p['draft'][i]}")
print(f"\nFound {found} of {fabricated} fabricated sentences; {false_alarms} false alarm(s)")

# %% Note
# A word-overlap check finds most documented examinations and advice that the transcript does not support,
# which are among the most worrying ambient-documentation errors because they look routine. It misses
# "Lipids repeated today" (a test, not an examination or advice) and flags "counselled re side effects",
# which the transcript supports in other words ("explained stomach upset"). Treat its output as a list for a
# human to look at, not a verdict.

# %% Step 4
n_notes = len(pairs)
labels_flat = [l for p in pairs for l in p["labels"] if l is not None]
notes_with_fc_or_fab = sum(any(l in (FC, FAB) for l in p["labels"]) for p in pairs)
notes_with_fab = sum(FAB in p["labels"] for p in pairs)
notes_safety_added = sum(SAFE in p["labels"] for p in pairs)
report = f"""AI documentation audit (synthetic data, {n_notes} notes)

1. Notes needing a clinically important correction (factual correction or fabricated content removed):
   {notes_with_fc_or_fab} of {n_notes} ({notes_with_fc_or_fab / n_notes:.0%}).
2. Notes in which the draft contained content that did not happen (examination, advice or test):
   {notes_with_fab} of {n_notes} ({notes_with_fab / n_notes:.0%}); automated check found {found} of {fabricated} such sentences.
3. Notes where the clinician had to add safety-netting the draft left out:
   {notes_safety_added} of {n_notes} ({notes_safety_added / n_notes:.0%}).

Context: {per_note.edited.sum() / per_note.sentences.sum():.0%} of sentences edited overall;
{(per_note.edited == 0).sum()} notes signed unchanged (check these are being read, not rubber-stamped).
Actions: share examples of fabricated examinations with the vendor; remind users that drafts omit
safety-netting; re-audit next month with a second-clinician review of a random sample."""
print(report)

# %% Note
# Three metrics a governance committee can track month by month: the share of notes needing a clinically
# important correction, the share containing fabricated content, and the share missing safety-netting.
# Edit rates alone are ambiguous (a falling rate can mean better drafts or less careful review), so they are
# reported as context, with a periodic independent review of signed notes to check that review is happening.

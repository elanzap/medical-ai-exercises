# %% Step 1
fixed = fixed_size_chunks(PROTOCOL)
sections_text = [s["text"] for s in section_chunks(PROTOCOL)]
print(f"{len(fixed)} fixed-size chunks, {len(sections_text)} section chunks")

# %% Step 2
def overlapping_chunks(text, size=40, overlap=10):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]

FULL_RULE = ["should not be started if eGFR is below 45", "If eGFR falls below 30 during treatment, stop metformin"]
question = "When should metformin be stopped because of kidney function?"
for name, chunker in [("fixed", fixed_size_chunks), ("fixed + 10-word overlap", overlapping_chunks)]:
    chunk = top_chunk(chunker(PROTOCOL), question)
    print(f"{name:<24} full rule retrieved: {all(part in chunk for part in FULL_RULE)}\n   {chunk}\n")

# %% Note
# The overlap repeats the last 10 words of each chunk at the start of the next, so a rule that straddles a
# boundary is more likely to appear whole in some chunk. Here it still does not help: the chunk that wins the
# retrieval ends at "...at least once a year. Metformin", so neither version returns the stop rule. Overlap
# reduces boundary problems on average; it does not guarantee that the retrieved chunk is complete.

# %% Step 3
conditional = [   # (question, the words that must stay together for the rule to be correct)
    ("Can metformin be started if eGFR is 40?", "Metformin should not be started if eGFR is below 45."),
    ("When does HbA1c move to every 6 months?", "Once targets are met and treatment is stable, check HbA1c every 6 months."),
    ("When should a foot problem be referred urgently?", "Refer urgently if there is an ulcer with spreading redness, a suspected infection of the bone, or a cold, pale foot."),
    ("When should the hypo treatment be repeated?", "Repeat if glucose is still below 70 mg/dL."),
    ("What HbA1c target applies to most adults?", "A common target for many non-pregnant adults is below 7%, adjusted for the individual."),
]
methods = {"fixed": fixed_size_chunks(PROTOCOL), "overlap": overlapping_chunks(PROTOCOL), "section": sections_text}
print(f"{'question':<50}" + "".join(f"{m:>9}" for m in methods))
for q, rule in conditional:
    print(f"{q:<50}" + "".join(f"{('whole' if rule in top_chunk(c, q) else 'missing'):>9}" for c in methods.values()))

# %% Note
# "missing" means the retrieved chunk does not contain the whole rule. Fixed-size chunks lose the urgent-referral
# rule (it crosses a 40-word boundary) and, without overlap, the HbA1c target. The condition ("if", "below") is
# often the part that gets cut off, which turns a correct rule into a dangerous one.
#
# Section chunks keep every rule whole because the rules never cross a heading. The one "missing" result for
# section chunks is different: keyword retrieval picked the wrong section, because nothing in the protocol says
# "hypo" and "repeated" does not match "Repeat". That is a retrieval problem, the subject of the next lesson.

# %% Step 4
TABLE = """## 6. Glucose thresholds (synthetic teaching values)
| Measurement | Low | Target | High |
|---|---|---|---|
| Fasting glucose (mg/dL) | below 70 | 80-130 | above 180 |
| Glucose 2 h after a meal (mg/dL) | below 70 | below 180 | above 250 |
| HbA1c (%) | not applicable | below 7 | above 9 |
"""

def table_row_chunks(table_text):
    """One chunk per row, each carrying the section title and the column headers."""
    lines = [l for l in table_text.strip().splitlines() if l.strip()]
    title, header = lines[0].lstrip("# "), [h.strip() for h in lines[1].strip("|").split("|")]
    chunks = []
    for row in lines[3:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        chunks.append(f"{title}: {cells[0]}: " + "; ".join(f"{h} {v}" for h, v in zip(header[1:], cells[1:])))
    return chunks

row_chunks = table_row_chunks(TABLE)
for c in row_chunks:
    print(c)
q = "What fasting glucose counts as high?"
print(f"\nQ: {q}\nA chunk: {top_chunk(sections_text + row_chunks, q)}")

# %% Note
# A table split into fixed-size word chunks loses its headers, leaving "below 70 | 80-130 | above 180" with no
# way to tell which number is which. Turning each row into a sentence that names its columns keeps every chunk
# self-explanatory.

# %% Step 5
# %% Note
# Optional. Take a short excerpt from an openly licensed guideline (check the licence: many WHO publications are
# CC BY-NC-SA 3.0 IGO, which allows non-commercial reuse with attribution). If it is a PDF, extract the text
# (for example with `pypdf`), then chunk it by headings as above and *read the extracted text*. Typical problems:
# headers and footers repeated inside paragraphs, words split across lines with hyphens, tables flattened into
# a stream of numbers, footnote markers attached to doses ("500 mg1"), and two-column layouts read across
# columns. Each one can change the meaning of a rule, so extraction errors must be fixed before indexing.

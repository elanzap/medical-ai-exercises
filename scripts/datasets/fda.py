"""Dataset: US FDA AI-Enabled Medical Devices, cleaned snapshot (fda-ai-enabled-medical-devices).

Cleans the US FDA's public "Artificial Intelligence-Enabled Medical Devices" list (a US Government work): parsed
dates, decision year, regulatory pathway from the submission number, tidy text, plus summary tables.

Download the CSV from the FDA page (button "Download a CSV File") to kaggle-build/fda/fda_ai_devices.csv first:
https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices

Usage: uv run --python 3.12 --with numpy --with pandas --with matplotlib scripts/datasets/fda.py --user rakuelli --snapshot 2026-11-15
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_common as K  # noqa: E402
from build_covers import make_cover  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SLUG = "fda-ai-enabled-medical-devices"
TITLE = "US FDA AI-Enabled Medical Devices"
SUBTITLE = "Cleaned snapshot of the FDA's public list of AI-enabled devices, 1995-2026"
OUT = K.BUILD / "datasets" / SLUG
ASSETS = K.ROOT / "assets" / "kaggle" / SLUG
SRC = K.BUILD / "fda" / "fda_ai_devices.csv"
FDA_PAGE = "https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices"


def pathway(sub: str) -> str:
    s = sub.upper()
    return "De Novo" if s.startswith("DEN") else "PMA" if s.startswith("P") else "510(k)" if s.startswith("K") else "other"


def clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def build(snapshot: str):
    raw = pd.read_csv(SRC, dtype=str, encoding="utf-8-sig")
    df = pd.DataFrame({
        "decision_date": pd.to_datetime(raw["Date of Final Decision"].map(clean), format="%m/%d/%Y").dt.strftime("%Y-%m-%d"),
        "submission_number": raw["Submission Number"].map(clean),
        "device": raw["Device"].map(clean),
        "company": raw["Company"].map(clean),
        "panel": raw["Panel (Lead)"].map(clean).replace({"Clinical Toxcicology": "Clinical Toxicology"}),
        "product_code": raw["Primary Product Code"].map(clean),
    })
    df["decision_year"] = df.decision_date.str[:4].astype(int)
    df["pathway"] = df.submission_number.map(pathway)
    df["is_pma_supplement"] = df.submission_number.str.contains("/S", regex=False).astype(int)
    df["device_name_count"] = df.device.str.count(";") + 1
    df["snapshot_date"] = snapshot
    df = df.sort_values(["decision_date", "submission_number"], ascending=[False, True]).reset_index(drop=True)
    by_year = df.pivot_table(index="decision_year", columns="pathway", values="submission_number", aggfunc="count", fill_value=0)
    by_year["total"] = by_year.sum(axis=1)
    by_year = by_year.reset_index()
    by_panel = df.groupby("panel").agg(devices=("submission_number", "size"), first_year=("decision_year", "min"), latest_year=("decision_year", "max")).sort_values("devices", ascending=False).reset_index()
    by_panel["share_percent"] = (100 * by_panel.devices / len(df)).round(1)
    return raw, df, by_year, by_panel


TABLES_DOC = {
    "fda_ai_devices.csv": ("One row per FDA decision on an AI-enabled device: cleaned text, ISO dates, decision year and regulatory pathway.", {
        "decision_date": ("datetime", "Date of the FDA's final decision (ISO format)"),
        "submission_number": ("id", "FDA submission number: K = 510(k), DEN = De Novo, P = PMA (P.../S... is a PMA supplement)"),
        "device": ("string", "Device name as listed (several names are separated by semicolons)"),
        "company": ("string", "Company as listed"),
        "panel": ("string", "Lead FDA review panel (medical specialty)"),
        "product_code": ("string", "Primary FDA product code"),
        "decision_year": ("integer", "Year of the decision"),
        "pathway": ("string", "Regulatory pathway derived from the submission number: 510(k), De Novo or PMA"),
        "is_pma_supplement": ("boolean", "1 if the submission is a PMA supplement"),
        "device_name_count": ("integer", "Number of device names in the row"),
        "snapshot_date": ("datetime", "Date this snapshot of the list was taken"),
    }),
    "devices_by_year.csv": ("Number of decisions per year, by regulatory pathway.", {
        "decision_year": ("integer", "Year"), "510(k)": ("integer", "510(k) clearances"), "De Novo": ("integer", "De Novo authorisations"),
        "PMA": ("integer", "PMA approvals and supplements"), "total": ("integer", "All decisions that year")}),
    "devices_by_panel.csv": ("Number of decisions per lead review panel, with first and latest year.", {
        "panel": ("string", "Lead review panel"), "devices": ("integer", "Decisions"), "first_year": ("integer", "Earliest decision year"),
        "latest_year": ("integer", "Latest decision year"), "share_percent": ("numeric", "Share of all decisions, %")}),
    "fda_ai_devices_original.csv": ("The list exactly as downloaded from the FDA, for reference.", None),
}


def charts(df, by_year, by_panel, snapshot):
    ASSETS.mkdir(parents=True, exist_ok=True)
    K.style()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=110, gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.74, bottom=0.14, wspace=0.42, left=0.05, right=0.97)
    yrs = df[df.decision_year >= 2010]
    rad = yrs[yrs.panel == "Radiology"].groupby("decision_year").size()
    oth = yrs[yrs.panel != "Radiology"].groupby("decision_year").size()
    years = range(2010, df.decision_year.max() + 1)
    rad, oth = rad.reindex(years, fill_value=0), oth.reindex(years, fill_value=0)
    ax[0].bar(years, rad.values, width=0.65, color=K.BLUE, label="Radiology")
    ax[0].bar(years, oth.values, bottom=rad.values, width=0.65, color=K.MUTED, edgecolor=K.SURFACE, linewidth=1.5, label="All other panels")
    last = max(years)
    ax[0].text(last, rad[last] + oth[last] + 4, "to June", ha="center", fontsize=8, color=K.INK_2)
    peak = (rad + oth).idxmax()
    ax[0].text(peak, rad[peak] + oth[peak] + 4, f"{rad[peak] + oth[peak]}", ha="center", fontsize=9)
    ax[0].legend(loc="upper left")
    ax[0].set(ylabel="Decisions", xticks=list(range(2010, last + 1, 2)))
    K.titled(ax[0], "AI-enabled device decisions per year, 2010 onward", "Radiology accounts for most of them")
    top = by_panel.head(8).iloc[::-1]
    cols = [K.ORANGE if p == "Radiology" else K.BLUE for p in top.panel]
    ax[1].barh(top.panel, top.devices, height=0.55, color=cols)
    ax[1].grid(axis="y", visible=False)
    ax[1].grid(axis="x", visible=True)
    for i, (v, s) in enumerate(zip(top.devices, top.share_percent)):
        ax[1].text(v + 10, i, f"{v:,} ({s:.0f}%)", va="center", fontsize=8.5)
    ax[1].set(xlabel="Decisions, all years", xlim=(0, top.devices.max() * 1.3))
    K.titled(ax[1], "Leading review panels", f"{len(df):,} decisions on the list")
    fig.text(0.01, 0.95, "US FDA AI-enabled medical devices", fontsize=14, color=K.INK, ha="left")
    fig.text(0.01, 0.9, f"FDA public list, snapshot {snapshot}; latest decision {df.decision_date.max()}", fontsize=9.5, color=K.INK_2, ha="left")
    K.source_note(fig, "Source: US Food and Drug Administration, Artificial Intelligence-Enabled Medical Devices list; cleaned in US FDA AI-Enabled Medical Devices (Kaggle)")
    fig.savefig(ASSETS / "overview.png")
    plt.close(fig)


    make_cover("fda")  # colourful Kaggle cover from the built CSVs (scripts/build_covers.py)


def description(user, df, by_year, by_panel, snapshot, lesson):
    pw = df.pathway.value_counts()
    panels = "\n".join(f"| {r.panel} | {r.devices:,} | {r.share_percent:.0f}% |" for r in by_panel.head(8).itertuples())
    recent = by_year[by_year.decision_year >= 2020][["decision_year", "total"]]
    yrs = " | ".join(f"{r.decision_year}: {r.total}" for r in recent.itertuples())
    return f"""# {TITLE}

A cleaned, analysis-ready snapshot of the **US Food and Drug Administration's public list of AI-enabled medical devices**: **{len(df):,} FDA decisions** from {df.decision_year.min()} to {df.decision_date.max()}, with ISO dates, decision year, **regulatory pathway** (510(k), De Novo or PMA) derived from the submission number, review panel and product code, plus summary tables by year and by panel. It accompanies the regulation track of the free [Medical AI course]({K.SITE}/medical-ai) by [{K.AUTHOR}]({K.SITE}/about), a physician and diabetologist.

**Source:** [FDA: Artificial Intelligence-Enabled Medical Devices]({FDA_PAGE}), a US Government work, downloaded on {snapshot}. The FDA updates the list periodically and notes that it is not comprehensive; always check the FDA page for the current version. This dataset is not affiliated with or endorsed by the FDA.

![Overview of the FDA AI-enabled device list]({K.RAW}/assets/kaggle/{SLUG}/overview.png)

## What's inside

| File | Rows | What it is |
|---|---|---|
| `fda_ai_devices.csv` | {len(df):,} | Cleaned list: dates, year, pathway, panel, product code |
| `devices_by_year.csv` | {len(by_year)} | Decisions per year by pathway |
| `devices_by_panel.csv` | {len(by_panel)} | Decisions per review panel |
| `fda_ai_devices_original.csv` | {len(df):,} | The list exactly as downloaded |
| `data_dictionary.csv` | | Every column, with type and description |

## At a glance

- **Pathways:** 510(k) {pw.get('510(k)', 0):,} | De Novo {pw.get('De Novo', 0):,} | PMA {pw.get('PMA', 0):,}
- **Decisions per year since 2020:** {yrs} (the latest year is partial)

| Lead review panel | Decisions | Share |
|---|---|---|
{panels}

## Ideas for what to do with it

- **Trends:** how fast are AI-enabled devices growing, and in which specialties?
- **Pathways:** how often do novel devices go through De Novo rather than 510(k)?
- **Text analysis:** classify devices by task (triage, segmentation, detection, reconstruction) from their names.
- **Join it:** product codes link to the FDA product classification database; submission numbers link to the FDA's 510(k), De Novo and PMA databases.

A ready-to-run [starter notebook](https://www.kaggle.com/code/{user}/us-fda-ai-enabled-devices-starter) reproduces the trend charts and searches for diabetes-related devices.

{K.links_block([lesson, ("Software as a Medical Device (SaMD)", f"{K.SITE}/medical-ai/regulation-ethics/software-as-a-medical-device")])}
{K.related_datasets(user, SLUG)}
## Licence and citation

The FDA list is a work of the US Government and in the public domain in the United States; this cleaned version is released under CC0. Company and device names are listed as published by the FDA, for identification only. Please cite the FDA as the source, and optionally: *Ram Kumar G. US FDA AI-Enabled Medical Devices (cleaned snapshot, {snapshot}). Kaggle, 2026.*

{K.DISCLAIMER}
"""


STARTER = [
    ("md", "## Load the list"),
    ("code", """
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(f'{DATA}/fda_ai_devices.csv', parse_dates=['decision_date'])
print(len(df), 'decisions, latest', df.decision_date.max().date())
df.head()
"""),
    ("md", "## Decisions per year, by pathway"),
    ("code", """
t = df[df.decision_year >= 2010].pivot_table(index='decision_year', columns='pathway', values='submission_number', aggfunc='count', fill_value=0)
t.plot.bar(stacked=True, figsize=(10, 4), color={'510(k)': '#2a78d6', 'De Novo': '#eb6834', 'PMA': '#9a9893'}, width=0.7)
plt.title('AI-enabled device decisions per year'); plt.ylabel('Decisions'); plt.show()
"""),
    ("md", "## Which specialties?"),
    ("code", """
df.panel.value_counts().head(10)
"""),
    ("md", "## Diabetes-related devices\n\nA simple keyword search on device names; it will miss some devices and catch a few false matches, so check the results."),
    ("code", """
mask = df.device.str.contains(r'glucose|insulin|diabet|retin', case=False)
df.loc[mask, ['decision_date', 'submission_number', 'device', 'panel', 'pathway']].sort_values('decision_date')
"""),
]


def starter(user, lesson):
    cells = [K.md_cell(f"""
# US FDA AI-Enabled Medical Devices: starter notebook

> Starter notebook for the dataset **[{TITLE}](https://www.kaggle.com/datasets/{user}/{SLUG})**, part of the free **[Medical AI course]({K.SITE}/medical-ai)** by [{K.AUTHOR}]({K.SITE}/about).
>
> - Source: the US FDA's public list of AI-enabled medical devices (snapshot); check the FDA page for the current version.
> - The lesson behind this data: [{lesson[0]}]({lesson[1]}).
"""), K.code_cell(K.find_data_snippet("fda_ai_devices.csv"))]
    for kind, src in STARTER:
        cells.append(K.md_cell(src) if kind == "md" else K.code_cell(src))
    cells.append(K.md_cell(f"""
## Continue learning

- The regulation and ethics track: [{K.SITE.removeprefix('https://')}/medical-ai/regulation-ethics]({K.SITE}/medical-ai/regulation-ethics)
- The whole course: [{K.SITE.removeprefix('https://')}/medical-ai]({K.SITE}/medical-ai)
- About the author: [{K.AUTHOR}]({K.SITE}/about)

---

{K.DISCLAIMER}
"""))
    K.write_starter(K.BUILD / "starters" / SLUG, user, "us-fda-ai-enabled-devices-starter", "US FDA AI-Enabled Devices: Starter", SLUG, cells)


def main(user, snapshot):
    course = json.loads((K.ROOT / "kaggle-course.json").read_text())
    lesson = next((l["title"], l["url"]) for l in course["lessons"] if l["slug"] == "us-fda-ai-devices")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    raw, df, by_year, by_panel = build(snapshot)
    df.to_csv(OUT / "fda_ai_devices.csv", index=False)
    by_year.to_csv(OUT / "devices_by_year.csv", index=False)
    by_panel.to_csv(OUT / "devices_by_panel.csv", index=False)
    shutil.copy(SRC, OUT / "fda_ai_devices_original.csv")
    pd.DataFrame([(f, c, t, d) for f, (_, cols) in TABLES_DOC.items() if cols for c, (t, d) in cols.items()], columns=["file", "column", "type", "description"]).to_csv(OUT / "data_dictionary.csv", index=False)
    charts(df, by_year, by_panel, snapshot)
    tables = {f: {"description": d, "columns": c} for f, (d, c) in TABLES_DOC.items()}
    tables["data_dictionary.csv"] = {"description": "Every column of every CSV in this dataset, with its type and description.", "columns": {"file": ("string", "CSV file name"), "column": ("string", "Column name"), "type": ("string", "Column type"), "description": ("string", "What the column holds")}}
    K.write_metadata(OUT, {
        "title": TITLE, "subtitle": SUBTITLE, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["healthcare", "medicine", "artificial intelligence", "government", "education"],
        "description": description(user, df, by_year, by_panel, snapshot, lesson),
        "resources": K.resources(tables),
        "userSpecifiedSources": f"US Food and Drug Administration, Artificial Intelligence-Enabled Medical Devices list ({FDA_PAGE}), a US Government work, downloaded {snapshot}. Cleaned by {K.AUTHOR} ({K.AUTHOR_DEGREE}) with scripts/datasets/fda.py in {K.REPO}: ISO dates, decision year, pathway from the submission number, whitespace and one spelling fix in a panel name (Toxcicology). Not affiliated with or endorsed by the FDA.",
        "expectedUpdateFrequency": "quarterly",
        "image": "dataset-cover-image.png",
    })
    starter(user, lesson)
    print(len(df), "rows;", df.pathway.value_counts().to_dict(), "| years", df.decision_year.min(), "-", df.decision_year.max(), "| panels", len(by_panel))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--snapshot", required=True, help="date the CSV was downloaded, YYYY-MM-DD")
    a = ap.parse_args()
    main(a.user, a.snapshot)

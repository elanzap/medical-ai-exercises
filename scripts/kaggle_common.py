"""Shared pieces for the Kaggle datasets: links, disclaimer, chart style, cover images, metadata and starter notebooks.

Chart style follows a plain editorial look: light surface, near-black text, recessive hairline grid, one blue for
the data, grey for context and orange only for the point being made. Palette validated with the dataviz skill's
validator (blue/orange/aqua all pass on the light surface; aqua needs direct labels).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle-build"
REPO = "https://github.com/elanzap/medical-ai-exercises"
RAW = "https://raw.githubusercontent.com/elanzap/medical-ai-exercises/main"
SITE = "https://ramkumarmd.com"
AUTHOR = "Dr. Ram Kumar G"
AUTHOR_DEGREE = "MBBS, MD (Internal Medicine), MPH (USA)"

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#b7d3f6"

DISCLAIMER = (
    "**Educational use only.** These data are for learning methods, not for clinical decisions or for building "
    "tools for real patient care. Synthetic records describe no real patients, clinicians or products. Mentions of "
    "medicines, models, companies or regulators are for identification, not endorsement."
)
LICENCE_DISPLAY = {"CC-BY-4.0": "Attribution 4.0 International (CC BY 4.0)"}  # CC0-1.0 is accepted as is


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9.5,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "ytick.left": False,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "text.color": INK,
    })


def titled(ax, title: str, subtitle: str | None = None):
    """Left-aligned plain title with an optional one-line takeaway underneath."""
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def source_note(fig, text: str):
    fig.text(0.01, 0.01, text, fontsize=8, color=MUTED, ha="left", va="bottom")


def cover(path: Path, title: tuple[str, str], subtitle: str, kicker: str, draw):
    """560 x 280 cover. `draw(ax)` paints a small real chart into the lower band; text stays inside the centre
    280 px so Kaggle's square thumbnail still reads."""
    fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    band = fig.add_axes([0.03, 0.04, 0.94, 0.38])
    band.set_facecolor(SURFACE)
    for s in band.spines.values():
        s.set_visible(False)
    band.set_xticks([])
    band.set_yticks([])
    band.grid(False)
    draw(band)
    txt = fig.add_axes([0, 0, 1, 1], facecolor="none")
    txt.axis("off")
    txt.set_xlim(0, 560)
    txt.set_ylim(0, 280)
    txt.text(280, 252, kicker, color=INK_2, fontsize=7.5, ha="center", va="center")
    txt.text(280, 218, title[0], color=INK, fontsize=16, fontweight="bold", ha="center", va="center")
    txt.text(280, 192, title[1], color=INK, fontsize=16, fontweight="bold", ha="center", va="center")
    txt.text(280, 166, subtitle, color=BLUE, fontsize=10.5, ha="center", va="center")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def resources(tables: dict) -> list:
    out = []
    for name, v in tables.items():
        r = {"path": name, "description": v["description"]}
        if v.get("columns"):
            r["schema"] = {"fields": [{"name": c, "type": t, "description": d} for c, (t, d) in v["columns"].items()]}
        out.append(r)
    return out


def write_metadata(out: Path, meta: dict):
    """Writes dataset-metadata.json for create/version, and a sibling '<folder>-update' for metadata --update
    (which wants the licence's display name and the cover image next to it)."""
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    upd = out.parent / f"{out.name}-update"
    shutil.rmtree(upd, ignore_errors=True)
    upd.mkdir(parents=True)
    if (out / "dataset-cover-image.png").exists():
        shutil.copy(out / "dataset-cover-image.png", upd / "dataset-cover-image.png")
    lic = [{"name": LICENCE_DISPLAY.get(l["name"], l["name"])} for l in meta["licenses"]]
    (upd / "dataset-metadata.json").write_text(json.dumps({**meta, "licenses": lic}, indent=2, ensure_ascii=False))


def links_block(lessons: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"- [{t}]({u})" for t, u in lessons)
    return f"""## Learn the methods

These data accompany the free [Medical AI course]({SITE}/medical-ai) by [{AUTHOR}]({SITE}/about), a physician and diabetologist. Lessons that use them:

{lines}

- Course home and all ten tracks: {SITE}/medical-ai
- Start here: {SITE}/medical-ai/start-here
- Exercises and solutions: {SITE}/medical-ai/exercises
- Generator code and notebooks (MIT): {REPO}
- About the author: {SITE}/about
"""


OTHER_DATASETS = [
    ("Synthetic Diabetes Clinic Data for Medical AI", "medical-ai-course-synthetic-exercise-data"),
    ("Synthetic CGM Glucose Traces", "synthetic-cgm-glucose-traces"),
    ("Synthetic Clinical RAG Evaluation Set", "synthetic-clinical-rag-evaluation"),
    ("Synthetic FHIR R4 Bundles for AI Agents", "synthetic-fhir-r4-diabetes-bundles"),
    ("Clinical LLM Red-Team Prompts", "clinical-llm-red-team-prompts"),
    ("US FDA AI-Enabled Medical Devices", "fda-ai-enabled-medical-devices"),
]


def related_datasets(user: str, this_slug: str) -> str:
    rows = "\n".join(f"- [{t}](https://www.kaggle.com/datasets/{user}/{s})" for t, s in OTHER_DATASETS if s != this_slug)
    return f"## Related datasets from the course\n\n{rows}\n"


def notebook(cells: list) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def md_cell(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}


def code_cell(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(keepends=True)}


def write_starter(folder: Path, user: str, slug: str, title: str, dataset_slug: str, cells: list, keywords=("healthcare", "medicine", "beginner"), internet=False):
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "notebook.ipynb").write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False))
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": title, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": False, "enable_gpu": False, "enable_tpu": False, "enable_internet": internet, "keywords": list(keywords),
        "dataset_sources": [f"{user}/{dataset_slug}"], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2))


def find_data_snippet(filename: str) -> str:
    return f"""
import glob, os
hits = glob.glob('/kaggle/input/**/{filename}', recursive=True)
DATA = os.path.dirname(hits[0]) if hits else os.environ.get('DATA_DIR', '.')
print('Data folder:', DATA)
"""

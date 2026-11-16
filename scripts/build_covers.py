"""Colourful Kaggle cover images (1120 x 560) for the six course datasets, drawn from each dataset's built CSVs.

Usage: python scripts/build_covers.py [key ...]     (keys: clinic cgm fhir rag redteam fda; default all)

Each cover: a two-tone diagonal gradient with a faint dot grid, a white title, a warm tagline and a key-numbers
line (all inside the centre 560 px square that Kaggle crops for the thumbnail), and a real chart from the data on a
white card. Writes dataset-cover-image.png into the dataset folder and its '-update' folder, and a copy to
assets/kaggle/<slug>/cover.png.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle-build"
W, H = 1120, 560
INK = "#1f2937"
GREEN, AMBER, RED, ORANGE = "#16a34a", "#f59e0b", "#dc2626", "#f97316"

COVERS = {
    "clinic": dict(slug="medical-ai-course-synthetic-exercise-data", folder=BUILD / "dataset", update=BUILD / "dataset-metadata-update",
                   grad=("#1e1b4b", "#6d28d9"), accent="#fde68a",
                   title=("Synthetic Diabetes", "Clinic Data"), tagline="5,000 patients. Zero privacy risk.",
                   kicker="COHORT  ·  NOTES  ·  LLM PAIRS  ·  FOR MEDICAL AI"),
    "cgm": dict(slug="synthetic-cgm-glucose-traces", grad=("#134e4a", "#1e3a8a"), accent="#fde68a",
                title=("Synthetic CGM", "Glucose Traces"), tagline="403,200 readings. Not one real patient.",
                kicker="100 PEOPLE  ·  14 DAYS  ·  EVERY 5 MINUTES"),
    "fhir": dict(slug="synthetic-fhir-r4-diabetes-bundles", grad=("#0b1739", "#0e7490"), accent="#a5f3fc",
                 title=("Synthetic FHIR R4", "Bundles"), tagline="A practice hospital for your AI agent.",
                 kicker="500 PATIENTS  ·  LOINC  ·  SNOMED CT  ·  RXNORM"),
    "rag": dict(slug="synthetic-clinical-rag-evaluation", grad=("#3b0764", "#a21caf"), accent="#fbcfe8",
                title=("Clinical RAG", "Evaluation Set"), tagline="Does your RAG cite the right page?",
                kicker="7 PROTOCOLS  ·  QUESTIONS  ·  LABELLED CLAIMS"),
    "redteam": dict(slug="clinical-llm-red-team-prompts", grad=("#111827", "#991b1b"), accent="#fecaca",
                    title=("Clinical LLM", "Red-Team Prompts"), tagline="Test your health chatbot before patients do.",
                    kicker="ESCALATE  ·  EMERGENCY  ·  REFUSE  ·  CORRECT"),
    "fda": dict(slug="fda-ai-enabled-medical-devices", grad=("#022c22", "#047857"), accent="#fde68a",
                title=("US FDA AI-Enabled", "Medical Devices"), tagline="{n} FDA decisions, analysis-ready.",
                kicker="510(k)  ·  DE NOVO  ·  PMA  ·  1995 TO 2026"),
}


def folder(c) -> Path:
    return c.get("folder", BUILD / "datasets" / c["slug"])


def update_folder(c) -> Path:
    return c.get("update", BUILD / "datasets" / f"{c['slug']}-update")


def background(fig, c1: str, c2: str):
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    yy, xx = np.mgrid[0:1:200j, 0:1:400j]
    t = np.clip(0.65 * xx + 0.35 * (1 - yy), 0, 1)[..., None]
    img = (1 - t) * np.array(to_rgb(c1)) + t * np.array(to_rgb(c2))
    bg.imshow(img, extent=(0, W, 0, H), origin="lower", aspect="auto")
    gx, gy = np.meshgrid(np.arange(14, W, 28), np.arange(14, H, 28))
    bg.scatter(gx.ravel(), gy.ravel(), s=2.2, color="white", alpha=0.08, lw=0)
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    return bg


def card(fig):
    """White rounded card across the lower band; returns the axes for the chart inside it."""
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.axis("off")
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.add_patch(FancyBboxPatch((40, 26), W - 80, 248, boxstyle="round,pad=0,rounding_size=18", fc="white", ec="none", alpha=0.94))
    ax = fig.add_axes([105 / W, 72 / H, (W - 200) / W, 180 / H], facecolor="none")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#4b5563", labelsize=11, length=0)
    ax.grid(False)
    return ax


def text(fig, c, tagline: str):
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.axis("off")
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.text(W / 2, 520, c["kicker"], color="white", alpha=0.85, fontsize=12.5, ha="center", va="center", fontweight="bold")
    ov.text(W / 2, 465, c["title"][0], color="white", fontsize=36, fontweight="bold", ha="center", va="center")
    ov.text(W / 2, 413, c["title"][1], color="white", fontsize=36, fontweight="bold", ha="center", va="center")
    tg = ov.text(W / 2, 352, tagline, color=c["accent"], fontsize=20, fontweight="bold", ha="center", va="center", style="italic")
    fit(fig, tg, 540)
    for t in ov.texts[:3]:
        fit(fig, t, 540)


def fit(fig, t, max_px: float):
    """Shrink a text until it fits the centre square Kaggle keeps for the thumbnail."""
    r = fig.canvas.get_renderer()
    while t.get_window_extent(r).width > max_px and t.get_fontsize() > 10:
        t.set_fontsize(t.get_fontsize() - 0.5)


# ---- charts -------------------------------------------------------------------------------------------------------

def chart_clinic(ax, d: Path):
    a = pd.read_csv(d / "diabetes_clinic_cohort.csv").hba1c_percent.dropna()
    edges = np.arange(5.0, 12.01, 0.25)
    counts, _ = np.histogram(a, bins=edges)
    cols = [GREEN if e < 7 else AMBER if e < 9 else RED for e in edges[:-1]]
    ax.bar(edges[:-1], counts, width=0.22, align="edge", color=cols)
    top = counts.max()
    for x, lab, col in [(6.0, "under 7%", GREEN), (8.0, "7 to 9%", AMBER), (10.0, "above 9%", RED)]:
        ax.text(x, top * 1.02, lab, color=col, fontsize=13, fontweight="bold", ha="center", va="bottom")
    ax.set_ylim(0, top * 1.22)
    ax.set_yticks([])
    ax.set_xlim(4.9, 12.1)
    ax.set_xticks(range(5, 13))
    ax.set_xticklabels([f"{x}%" for x in range(5, 13)])
    ax.text(12.05, top * 0.5, "HbA1c,\n5,000 synthetic\npatients", color=INK, fontsize=12, ha="right", va="center")


def chart_cgm(ax, d: Path):
    r = pd.read_csv(d / "cgm_readings.csv", usecols=["person_id", "glucose_mg_dl"])
    best = None
    for pid, g in r.groupby("person_id", sort=True):
        v = g.glucose_mg_dl.values
        for day in range(min(14, len(v) // 288)):
            seg = pd.Series(v[day * 288:(day + 1) * 288]).interpolate().bfill().ffill().values
            if seg.min() < 68 and seg.max() > 215:
                best = seg
                break
        if best is not None:
            break
    seg = best if best is not None else pd.Series(r.glucose_mg_dl.values[:288]).interpolate().bfill().values
    x = np.arange(len(seg)) / 12
    ax.axhspan(70, 180, color=GREEN, alpha=0.10, lw=0)
    pts = np.column_stack([x, seg])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    mid = (seg[:-1] + seg[1:]) / 2
    cols = [RED if m < 70 else ORANGE if m > 180 else GREEN for m in mid]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=4, capstyle="round"))
    ax.set_xlim(0, 24)
    ax.set_ylim(40, max(260, seg.max() + 15))
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
    ax.set_yticks([70, 180])
    ax.set_yticklabels(["70", "180"])
    for y0, lab, col in [(55, "low", RED), (125, "in range", GREEN), (225, "high", ORANGE)]:
        ax.text(23.8, y0, lab, color=col, fontsize=13, fontweight="bold", ha="right", va="center")


def chart_fhir(ax, d: Path):
    files = [("Observations", "observations.csv"), ("Encounters", "encounters.csv"), ("Medications", "medications.csv"),
             ("Conditions", "conditions.csv"), ("Agent tasks", "agent_tasks.csv"), ("Patients", "patients.csv")]
    n = [sum(1 for _ in open(d / f)) - 1 for _, f in files]
    cols = ["#0891b2", "#2563eb", "#7c3aed", "#db2777", "#f97316", "#16a34a"]
    x = np.arange(len(files))
    ax.bar(x, n, color=cols, width=0.62)
    for i, v in enumerate(n):
        ax.text(i, v * 1.03 + max(n) * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=13, fontweight="bold", color=cols[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f for f, _ in files], fontsize=12)
    ax.set_yticks([])
    ax.set_ylim(0, max(n) * 1.25)


def chart_rag(ax, d: Path):
    lab = pd.read_csv(d / "answer_claims.csv").label.value_counts()
    cols = ["#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"][: len(lab)]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    pie = ax.inset_axes([0.06, 0.0, 0.32, 1.0])
    pie.pie(lab.values, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    pie.text(0, 0, f"{lab.sum()}\nclaims", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    pie.set_aspect("equal")
    for i, (k, v) in enumerate(lab.items()):
        y = 3.3 - i * (2.6 / max(1, len(lab) - 1))
        ax.add_patch(FancyBboxPatch((4.2, y - 0.18), 0.36, 0.36, boxstyle="round,pad=0,rounding_size=0.08", fc=cols[i], ec="none"))
        ax.text(4.8, y, f"{str(k).replace('_', ' ')}", fontsize=14, va="center", color=INK, fontweight="bold")
        ax.text(9.6, y, f"{v}", fontsize=14, va="center", ha="right", color=cols[i], fontweight="bold")


def chart_redteam(ax, d: Path):
    c = pd.read_csv(d / "red_team_prompts.csv").category.value_counts()
    cols = ["#dc2626", "#f97316", "#f59e0b", "#16a34a", "#0891b2", "#2563eb", "#7c3aed", "#db2777"]
    cols = (cols * 2)[: len(c)]
    x = np.arange(len(c))
    ax.bar(x, c.values, color=cols, width=0.62)
    for i, v in enumerate(c.values):
        ax.text(i, v + c.max() * 0.03, str(v), ha="center", va="bottom", fontsize=13, fontweight="bold", color=cols[i])
    ax.set_xticks(x)
    ax.set_xticklabels([str(k).replace("_", " ") for k in c.index], fontsize=11.5)
    ax.set_yticks([])
    ax.set_ylim(0, c.max() * 1.25)


def chart_fda(ax, d: Path):
    df = pd.read_csv(d / "fda_ai_devices.csv", usecols=["decision_year", "panel"])
    df = df[df.decision_year >= 2010]
    top = df.panel.value_counts().index[:4].tolist()
    df["grp"] = df.panel.where(df.panel.isin(top), "Other panels")
    t = df.pivot_table(index="decision_year", columns="grp", aggfunc="size", fill_value=0)
    order = top + ["Other panels"]
    cols = ["#16a34a", "#0891b2", "#f59e0b", "#db2777", "#9ca3af"]
    bottom = np.zeros(len(t))
    for g, col in zip(order, cols):
        if g in t:
            ax.bar(t.index, t[g].values, bottom=bottom, color=col, width=0.75, label=g)
            bottom += t[g].values
    ax.set_xticks([y for y in t.index if y % 2 == 0])
    ax.set_yticks([])
    ax.set_ylim(0, bottom.max() * 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5, ncol=1, handlelength=1, borderaxespad=0.2)
    return len(pd.read_csv(d / "fda_ai_devices.csv", usecols=["decision_year"]))


CHARTS = {"clinic": chart_clinic, "cgm": chart_cgm, "fhir": chart_fhir, "rag": chart_rag, "redteam": chart_redteam, "fda": chart_fda}


def make_cover(key: str) -> Path:
    c = COVERS[key]
    d = folder(c)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    plt.rcParams["font.family"] = "DejaVu Sans"
    background(fig, *c["grad"])
    ax = card(fig)
    n = CHARTS[key](ax, d)
    text(fig, c, c["tagline"].format(n=f"{n:,}" if n else ""))
    out = d / "dataset-cover-image.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    upd = update_folder(c)
    if upd.exists():
        shutil.copy(out, upd / "dataset-cover-image.png")
    asset = ROOT / "assets" / "kaggle" / c["slug"]
    asset.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, asset / "cover.png")
    return out


if __name__ == "__main__":
    for k in sys.argv[1:] or list(COVERS):
        print(make_cover(k))

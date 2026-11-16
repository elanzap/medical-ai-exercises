"""Builds (and optionally publishes) the Kaggle versions of the Medical AI course.

Every notebook gets a header and footer that link back to the lesson on ramkumarmd.com, the course,
the author page, GitHub and Colab. The course data (lesson titles, URLs, articles) comes from the
website repo: run `npm run kaggle-export -- <this repo>/kaggle-course.json` there first.

Usage:
  python3 scripts/build_kaggle.py --user <kaggle username>             # build into kaggle-build/
  python3 scripts/build_kaggle.py --user <kaggle username> --publish   # build, then upload everything
  python3 scripts/build_kaggle.py --user <kaggle username> --publish --only <kernel-slug>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from kaggle_cohort_cells import LOADER as COHORT_LOADER, SECTIONS as COHORT_SECTIONS

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle-build"
REPO = "https://github.com/elanzap/medical-ai-exercises"
COLAB = "https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/"
DATASET_SLUG = "medical-ai-course-synthetic-exercise-data"
DATASET_TITLE = "Synthetic Diabetes Clinic Data for Medical AI"

DISCLAIMER = (
    "**Educational content only.** This material teaches methods; it is not clinical advice. Code, models and "
    "examples here are not validated medical devices and must not be used to make decisions about real patients "
    "without proper clinical validation, governance and any required regulatory approval. Examples use synthetic "
    "or openly licensed de-identified data only. Never paste identifiable patient information into public AI tools. "
    "Mentions of models, tools or vendors are for identification, not endorsement."
)
LICENCE = (
    f"Code: MIT licence. Text: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
    f"Source: [{REPO.removeprefix('https://')}]({REPO})."
)


# Short names where the trimmed lesson or article title would read badly (Kaggle titles max 50 characters).
SHORT_NAMES = {
    "prediction-vs-causal-questions": "Prediction or causation?",
    "cohorts-and-data-leakage": "Cohorts and data leakage",
    "unsupervised-phenotyping": "Unsupervised patient phenotypes",
    "multimodal-models": "Multimodal models in medicine",
    "summarisation-ambient-documentation": "Clinical summarisation and scribes",
    "why-rag": "Why RAG? Grounding answers",
    "india-cdsco-ai": "India: CDSCO and AI software",
    "who-guidance-ethics": "WHO guidance on AI ethics",
    "clinical-ner-and-coding": "Clinical NER and coding",
    "foot-ulcer-image-classification": "Foot-ulcer image classification",
    "diabetes-guideline-rag-assistant": "Diabetes guideline RAG assistant",
    "eu-ai-act-clinical-ai-tools": "What the EU AI Act means for clinical AI",
    "llms-public-medical-advice-randomised-study": "Public use of AI chatbots for medical advice",
    "uk-national-commission-ai-healthcare-regulation-2026": "UK's 44 recommendations on AI in healthcare",
    "questions-before-adopting-ai-diabetes-clinic": "12 questions before adopting clinical AI",
}


def kaggle_slug(title: str) -> str:
    """Kaggle derives a kernel's URL slug from its title in the same way."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def short_title(title: str, suffix: str, limit: int = 50) -> str:
    """A title that fits Kaggle's 50-character limit: the full title if it fits, else the part before the colon
    (when it says enough on its own), else the title trimmed by whole words, never ending on a filler word."""
    prefix = title.split(":")[0]
    if len(title + suffix) <= limit:
        base = title
    elif len(prefix) >= 18 and len(prefix + suffix) <= limit:
        base = prefix
    else:
        words = title.split()
        while words and len(" ".join(words) + suffix) > limit:
            words.pop()
        while words and words[-1].lower().strip(",:;") in {"a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "by", "what", "when", "medical"}:
            words.pop()
        base = " ".join(words).rstrip(",:;")
    return base + suffix


def code_cell(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(keepends=True)}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def load_course() -> dict:
    path = ROOT / "kaggle-course.json"
    if not path.exists():
        raise SystemExit("kaggle-course.json missing: run `npm run kaggle-export -- <this repo>/kaggle-course.json` in the website repo")
    return json.loads(path.read_text())


def build_dataset(user: str, site: str, author: str) -> set[str]:
    """Builds the dataset, its charts and starter notebook with scripts/kaggle_dataset.py (needs numpy, pandas and
    matplotlib, run through uv). Returns the lesson keys whose exercises use the course data."""
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    subprocess.run([uv, "run", "-q", "--python", "3.12", "--with", "numpy", "--with", "pandas", "--with", "matplotlib",
                    str(ROOT / "scripts" / "kaggle_dataset.py"), "--user", user], check=True)
    return {f"{p.parent.name}/{p.stem}" for p in (ROOT / "exercise-data").glob("*/*.py")}


def build_kernel(user, slug, title, cells, data_source, keywords):
    folder = BUILD / "kernels" / slug
    folder.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (folder / "notebook.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": keywords,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"] if data_source else [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))


def build(user: str) -> list[dict]:
    course = load_course()
    site, author = course["site"], course["author"]
    shutil.rmtree(BUILD / "kernels", ignore_errors=True)
    data_keys = build_dataset(user, site, author)
    tracks = {t["slug"]: t for t in course["tracks"]}
    by_track: dict[str, list] = {}
    for lesson in sorted(course["lessons"], key=lambda l: (tracks[l["track"]]["order"], l["number"])):
        by_track.setdefault(lesson["track"], []).append(lesson)

    built, used = [], set()
    author_line = f"[{author}]({site}/about), {course['authorDegree']}"
    for track_lessons in by_track.values():
        for i, lesson in enumerate(track_lessons):
            if not lesson.get("notebookPath"):
                continue
            track = tracks[lesson["track"]]
            nb_path = ROOT / lesson["notebookPath"]
            sol_path = nb_path.with_name(nb_path.stem + "-solution.ipynb")
            key = f"{lesson['track']}/{nb_path.stem}"
            level = lesson["level"]
            keywords = ["healthcare", "medicine", level]
            nxt = next((l for l in track_lessons[i + 1:]), None)
            name = SHORT_NAMES.get(lesson["slug"], lesson["title"])
            titles = {kind: short_title(name, f" ({kind})") for kind in ("Exercise", "Solution")}
            slugs = {kind: kaggle_slug(t) for kind, t in titles.items()}
            for kind, path in (("Exercise", nb_path), ("Solution", sol_path)):
                if not path.exists():
                    continue
                if slugs[kind] in used:
                    raise SystemExit(f"Duplicate Kaggle slug {slugs[kind]}")
                used.add(slugs[kind])
                other = "Solution" if kind == "Exercise" else "Exercise"
                counterpart = f"https://www.kaggle.com/code/{user}/{slugs[other]}"
                repo_path = path.relative_to(ROOT).as_posix()
                dataset_url = f"https://www.kaggle.com/datasets/{user}/{DATASET_SLUG}"
                uses_data = key in data_keys or lesson["slug"] in COHORT_SECTIONS
                data_line = f"\n> - Data: the Kaggle dataset [{DATASET_TITLE}]({dataset_url})" if uses_data else ""
                header = f"""
> **{lesson['title']}**: {kind.lower()} notebook, lesson {track['order']}.{lesson['number']:02d} of the free **[Medical AI course]({site}/medical-ai)** by {author_line}.
>
> - **Read the full lesson** (explanation, worked example, FAQs and sources): [{lesson['title']}]({lesson['url']})
> - Track: [{track['title']}]({track['url']}), and the [course start page]({site}/medical-ai/start-here)
> - {"Try the exercise first; the worked solution is in the" if kind == "Exercise" else "This is the worked solution; try the"} [{other.lower()} notebook]({counterpart}) {"when you're done" if kind == "Exercise" else "first"}
> - Also on [GitHub]({REPO}/blob/main/{repo_path}) and [Colab]({COLAB}{repo_path}){data_line}
>
> {LICENCE}
"""
                footer = f"""
## Continue the course

- Lesson page with the full explanation and sources: [{lesson['title']}]({lesson['url']})
""" + (f"- Next lesson: [{nxt['title']}]({nxt['url']})\n" if nxt else f"- Next: browse all tracks of the [Medical AI course]({site}/medical-ai)\n") + f"""- All exercises and solutions: [{site.removeprefix('https://')}/medical-ai/exercises]({site}/medical-ai/exercises)
- About the author: [{author}]({site}/about) ([LinkedIn]({course['linkedinUrl']}))

---

{DISCLAIMER}
"""
                cells = json.loads(path.read_text())["cells"]
                extension = []
                if lesson["slug"] in COHORT_SECTIONS:
                    intro, code_src = COHORT_SECTIONS[lesson["slug"]]
                    extension = [
                        md(f"## Try it on a larger cohort\n\nAn extra section for Kaggle, beyond the lesson's exercise. {intro} "
                           f"The data is `diabetes_clinic_cohort.csv` from the dataset [{DATASET_TITLE}]({dataset_url}): 5,000 synthetic patients, no real people."),
                        code_cell(COHORT_LOADER),
                        code_cell(code_src),
                    ]
                for c in cells:
                    if c["cell_type"] == "code":
                        c.setdefault("outputs", [])
                        c.setdefault("execution_count", None)
                build_kernel(user, slugs[kind], titles[kind], [md(header)] + cells + extension + [md(footer)], uses_data, keywords)
                built.append({"slug": slugs[kind], "title": titles[kind], "kind": kind.lower(), "lesson": lesson["url"]})

    for article in course["articles"]:
        title = short_title(SHORT_NAMES.get(article["slug"], article["title"]), "")
        slug = kaggle_slug(title)
        if slug in used:
            raise SystemExit(f"Duplicate Kaggle slug {slug}")
        used.add(slug)
        key_points = []
        for s in article["sections"]:
            first = (s["paragraphs"] or s["bullets"] or [""])[0]
            sentence = re.split(r"(?<=[.!?])\s", first, maxsplit=1)[0]
            key_points.append(f"- **{s['heading']}:** {sentence}")
        sources = "\n".join(f"- [{s['label']}]({s['href']})" if s.get("href") else f"- {s['label']}" for s in article["sources"])
        cells = [
            md(f"""
# {article['title']}

> A summary of an article from the free **[Medical AI course]({site}/medical-ai)** by {author_line}.
> **Read the full article, with tables, decision guides and FAQs: [{article['title']}]({article['url']})**
"""),
            md(f"## In short\n\n{article['quickAnswer']}"),
            md("## Key points\n\n" + "\n".join(key_points) + f"\n\n**[Continue reading on ramkumarmd.com]({article['url']})**"),
            md(f"## Sources\n\n{sources}"),
            md(f"""
## More from the course

- All articles: [{site.removeprefix('https://')}/medical-ai/articles]({site}/medical-ai/articles)
- The course, from the start: [{site.removeprefix('https://')}/medical-ai/start-here]({site}/medical-ai/start-here)
- About the author: [{author}]({site}/about)

---

{DISCLAIMER}
"""),
        ]
        build_kernel(user, slug, title, cells, False, ["healthcare", "medicine"])
        built.append({"slug": slug, "title": title, "kind": "article", "lesson": article["url"]})

    print(f"Built {len(built)} kernels and the dataset in {BUILD}")
    return built


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def publish(user: str, built: list[dict], only: str | None):
    kaggle = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")  # Kaggle CLI 2.x (uv tool install kaggle)
    log = BUILD / "published.csv"
    done = {}
    if log.exists():
        done = {r["slug"]: r for r in csv.DictReader(open(log))}
    if not only:
        code, out = run([kaggle, "datasets", "status", f"{user}/{DATASET_SLUG}"])
        if code != 0 or "ready" not in out.lower():
            code, out = run([kaggle, "datasets", "create", "-p", str(BUILD / "dataset"), "--dir-mode", "zip", "--public"])
            print("dataset:", out.splitlines()[-1] if out else code)
            time.sleep(60)  # let the dataset finish processing before kernels attach it
    # Updates to existing notebooks first: Kaggle limits creating new notebooks far more than updating them.
    built = sorted(built, key=lambda i: done.get(i["slug"], {}).get("result") != "stale")
    for item in built:
        if only and item["slug"] != only:
            continue
        if not only and done.get(item["slug"], {}).get("result") == "pushed":
            continue
        code, out = run([kaggle, "kernels", "push", "-p", str(BUILD / "kernels" / item["slug"])])
        if "rate limit" in out.lower():
            print(f"rate limited on {item['slug']}; stopping (resume later; already-published notebooks are skipped)", flush=True)
            break
        ok = code == 0 and "error" not in out.lower() and "rate limit" not in out.lower()
        done[item["slug"]] = {**item, "url": f"https://www.kaggle.com/code/{user}/{item['slug']}", "result": "pushed" if ok else "failed", "message": out.splitlines()[-1] if out else ""}
        print(("OK  " if ok else "FAIL"), item["slug"], "-", done[item["slug"]]["message"])
        with open(log, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["slug", "title", "kind", "lesson", "url", "result", "message"])
            w.writeheader()
            w.writerows(done.values())
        time.sleep(180)  # publish slowly: Kaggle rate-limits making notebooks public


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--only", help="publish just this kernel slug (for a test push)")
    args = ap.parse_args()
    items = build(args.user)
    if args.publish:
        publish(args.user, items, args.only)

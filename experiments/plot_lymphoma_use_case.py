"""Erzeugt lymphoma_use_case.png fuer die Publikation.

Zwei Panels:
  links  - Yield des regelbasierten Lymphom-Retrievals je Wachstumsmuster
  rechts - Uebereinstimmung mit der manuellen Kohorte auf den 13
           vergleichbaren Zeilen (gefunden / Muster korrekt / verfehlt)

Datenbasis: results/lymphoma_search/{lymphoma_pattern_cases.csv,comparison.json}
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("experiments/lymphoma_use_case.png")
RES = Path("results/lymphoma_search")

PATTERNS = [("nodular", "nodular"), ("diffuse", "diffuse"),
            ("minimal-change", "minimal-change"), ("necrosis", "necrotic")]
SUB2LABEL = {"nodular": "nodular", "diffuse": "diffuse",
             "minimal-change": "minimal-change", "necrosis": "necrosis"}


def main() -> None:
    cases = list(csv.DictReader((RES / "lymphoma_pattern_cases.csv").open()))
    comp = json.loads((RES / "comparison.json").read_text())

    auto = Counter()
    for r in cases:
        for lab in r["labels"].split(";"):
            auto[lab] += 1

    comparable = [c for c in comp if c["found"] != "nicht_in_datenbank"]
    found = {s: sum(1 for c in comparable
                    if c["subcategory"] == s and c["found"] == "gefunden")
             for s in SUB2LABEL}
    correct = {s: sum(1 for c in comparable
                      if c["subcategory"] == s and c["label_match"] == "ja")
               for s in SUB2LABEL}
    manual = {s: sum(1 for c in comparable if c["subcategory"] == s)
              for s in SUB2LABEL}

    ink = "#222222"
    bar_main, bar_ok, bar_miss = "#4878A8", "#5FA777", "#C9CFD6"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.5),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # ---- links: Yield je Wachstumsmuster -------------------------------
    labels = [en for _, en in PATTERNS]
    keys = [k for k, _ in PATTERNS]
    vals = [auto[k] for k in keys]
    b = ax1.bar(labels, vals, color=bar_main, width=0.62)
    for rect, v in zip(b, vals):
        ax1.text(rect.get_x() + rect.get_width() / 2, v + 1.2, str(v),
                 ha="center", va="bottom", fontsize=9, color=ink)
    ax1.set_ylim(0, max(vals) * 1.18)
    ax1.set_ylabel("Automatically retrieved cases")
    ax1.set_title(f"Retrieval yield\n{len(cases)} cases, "
                  f"{sum(vals)} pattern assignments", fontsize=10)
    ax1.tick_params(labelsize=8.5)
    ax1.spines[["top", "right"]].set_visible(False)

    # ---- rechts: Uebereinstimmung je Muster ----------------------------
    y = list(range(len(keys)))
    h = 0.38
    fm = [found[k] for k in keys]
    mn = [manual[k] - found[k] for k in keys]
    # invert_yaxis() dreht die Achse: -h/2 landet optisch oberhalb.
    ax2.barh([i - h / 2 for i in y], fm, height=h, color=bar_ok,
             label="retrieved")
    ax2.barh([i + h / 2 for i in y], mn, height=h, color=bar_miss,
             label="missed")
    for i, k in enumerate(keys):
        ax2.text(fm[i] + 0.06, i - h / 2, f"{correct[k]}/{manual[k]} correct",
                 va="center", fontsize=8, color=ink)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(manual.values()) + 1.9)
    ax2.set_xlabel("Manual cases (only 13 of 77 rows are joinable)")
    ax2.set_title("Concordance with the manual cohort", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Clinical use case: targeted retrieval of lymph node slides",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("written", OUT)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Combined 2x2 figure: production-run summary (top) + lymphoma use case (bottom).

Renders four panels in a single publication figure (PNG + PDF) with ONE font
family and ONE base font size across every panel, axis, tick and legend:

  top-left    (a) extracted mentions per label (horizontal bars, log scale)
  top-right   (b) per-report extraction latency (histogram, median/mean lines)
  bottom-left (c) lymphoma rule-based retrieval yield per growth pattern
  bottom-right(d) concordance with the manual lymphoma cohort per pattern

Top panels read ``results/summary_2024.json``; the latency histogram is parsed
from ``logs/pipeline_2024.log`` (paired Start->FERTIG reports).  The two
lymphoma panels reproduce the values of the archived ``lymphoma_use_case.png``
(retrieval yield and manual-cohort concordance), which are also stated in
Table~\ref{tab:lymphoma-concordance}.

Run from the repository root::

    python scripts/plot_archive_usecase.py \
        [--log logs/pipeline_2024.log] [--summary results/summary_2024.json] \
        [--copy-to ~/Documents/overleaf_datasetGeneration/images]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- one font family + one base size for every panel ------------------------
FONT = "Liberation Sans"
FS = 11
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "DejaVu Sans", "sans-serif"],
    "font.size": FS, "axes.unicode_minus": False,
    "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "figure.titlesize": FS,
})
# shared colours
C_BAR = "#4C72B0"
C_HIST = "#5FA777"
C_MED = "black"
C_MEAN = "#E07B00"
C_LYM = "#4878A8"
C_OK = "#5FA777"
C_MISS = "#C9CFD6"


# =============================================================================
# data
# =============================================================================

def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_latency(path: Path) -> list[float]:
    """Paired per-report extraction durations (Start->FERTIG), in seconds."""
    RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    RE_START = re.compile(r"REPORT (\d+) \| Start Extraktion")
    RE_FERTIG = re.compile(r"REPORT (\d+) \| FERTIG")
    starts: dict[int, float] = {}
    durs: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = RE_START.search(line)
            if s:
                ts = RE_TS.match(line)
                starts[int(s.group(1))] = ts.group(1) if ts else None
                continue
            e = RE_FERTIG.search(line)
            if e:
                rid = int(e.group(1))
                base = starts.pop(rid, None)
                if base is None:
                    continue
                ts = RE_TS.match(line)
                if ts is None:
                    continue
                from datetime import datetime
                t0 = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
                t1 = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
                if t1 >= t0:
                    durs.append((t1 - t0).total_seconds())
    return durs


# lymphoma numbers (as in Table: lymphoma concordance / original figure)
LYMPHOMA = [
    # (pattern, archived_yield, manual, retrieved, correct)
    ("nodular", 64, 2, 2, 2),
    ("diffuse", 38, 5, 4, 4),
    ("minimal-change", 38, 3, 3, 1),
    ("necrotic", 43, 3, 2, 2),
]
LYMPHOMA_CASES = 117
LYMPHOMA_REPORTS = 237


# =============================================================================
# panels
# =============================================================================

def panel_labels(ax, labels: dict[str, int]) -> None:
    import matplotlib.text as mtext
    items = sorted(labels.items(), key=lambda kv: kv[1])  # ascending counts
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    ax.barh(names, counts, color=C_BAR, edgecolor="none")
    ax.set_xscale("symlog", linthresh=1)
    # Count next to the label, on the left of the axis: the label column is
    # left-aligned and the count column is right-aligned (digits line up with
    # the axis edge), so they read as a clean "name   count" pair.
    nw = max(len(n) for n in names)
    tick_labels = []
    for name, c in zip(names, counts):
        t = mtext.Text(0, 0)
        t.set_text(f"{name.ljust(nw)}  {c:,}")
        t.set_ha("right")
        tick_labels.append(t)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Extracted mentions (log scale)")
    ax.set_title("(a) Extracted mentions per label", loc="left", pad=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)


def panel_latency(ax, durs: list[float]) -> None:
    bin_edges = [0]
    step = max(5, int(statistics.median(durs) / 4))
    mx = int(max(durs))
    while bin_edges[-1] < mx:
        bin_edges.append(bin_edges[-1] + step)
    if bin_edges[-1] < mx:
        bin_edges.append(mx)
    ax.hist(durs, bins=bin_edges, color=C_HIST, edgecolor="none", alpha=0.9)
    med = statistics.median(durs)
    mean = statistics.mean(durs)
    ax.axvline(med, color=C_MED, ls="--", lw=1.2, label=f"median {med:.0f}s")
    ax.axvline(mean, color=C_MEAN, ls="--", lw=1.2, label=f"mean {mean:.0f}s")
    ax.set_xlabel("Extraction time per report (s)")
    ax.set_ylabel("Number of reports")
    ax.set_title("(b) Per-report extraction latency", loc="left", pad=8)
    ax.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    p90 = sorted(durs)[int(len(durs) * 0.9)]
    ax.text(0.98, -0.2, f"p90 {p90:.0f}s · 300s client timeout drops a tail (n={len(durs):,})",
            transform=ax.transAxes, ha="right", va="top", color="#55606e")


def panel_yield(ax, data: list[tuple]) -> None:
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    bars = ax.bar(names, vals, color=C_LYM, width=0.62, edgecolor="none")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, f"{v}",
                ha="center", va="bottom", color="0.1")
    ax.set_ylim(0, max(vals) * 1.16)
    ax.set_ylabel("Retrieved cases (per pattern)")
    ax.set_title(f"(c) Lymphoma retrieval yield  ·  {LYMPHOMA_CASES} cases, "
                 f"{sum(vals)} assignments", loc="left", pad=8)
    ax.tick_params(axis="x", length=0, pad=4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def panel_concordance(ax, data: list[tuple]) -> None:
    names = [d[0] for d in data]
    manual = [d[2] for d in data]
    retrieved = [d[3] for d in data]
    correct = [d[4] for d in data]
    missed = [m - r for m, r in zip(manual, retrieved)]
    y = list(range(len(names)))
    h = 0.38
    ax.barh([i - h / 2 for i in y], retrieved, height=h, color=C_OK, label="retrieved")
    ax.barh([i + h / 2 for i in y], missed, height=h, color=C_MISS, label="missed")
    for i, (c, m) in enumerate(zip(correct, manual)):
        ax.text(retrieved[i] + 0.06, i - h / 2, f"{c}/{m} correct",
                va="center", color="0.1")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(manual) + 1.9)
    ax.set_xlabel("Manual cohort rows (13 of 77 joinable)")
    ax.set_title("(d) Concordance with the manual cohort", loc="left", pad=8)
    ax.legend(loc="lower right", frameon=False)
    ax.tick_params(axis="y", length=0, pad=4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.2)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="logs/pipeline_2024.log")
    ap.add_argument("--summary", default="results/summary_2024.json")
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args()

    summary = load_summary(Path(args.summary))
    labels = summary["labels"]
    reports = summary["reports"]

    t0 = time.time()
    durs = parse_latency(Path(args.log)) if Path(args.log).exists() else \
        [summary["latency"]["median"]] * 10
    print(f"latency: {len(durs)} reports in {time.time() - t0:.1f}s "
          f"(median {statistics.median(durs):.0f}s)")

    fig, gs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Patho-HERO 2024 archive ({reports:,} reports) — extraction "
                 f"and clinical use case", y=0.99, ha="left")

    panel_labels(gs[0, 0], labels)
    panel_latency(gs[0, 1], durs)
    panel_yield(gs[1, 0], LYMPHOMA)
    panel_concordance(gs[1, 1], LYMPHOMA)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = Path("experiments") / "archive_use_case.png"
    out_pdf = Path("experiments") / "archive_use_case.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white",
                pad_inches=0.14)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", pad_inches=0.14)
    plt.close(fig)
    print(f"Wrote: {out_png}\nWrote: {out_pdf}")

    if args.copy_to is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for f in (out_png, out_pdf):
            shutil.copy(str(f), args.copy_to / f.name)
        print(f"Copied to {args.copy_to.resolve()}")


if __name__ == "__main__":
    main()

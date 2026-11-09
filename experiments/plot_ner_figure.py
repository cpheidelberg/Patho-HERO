#!/usr/bin/env python3
"""NER figure: comparison of Hybrid-Pipeline, MedCAT and OpenMed."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
except Exception:  # pragma: no cover
    sns = None

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"

HERE = Path(__file__).resolve().parent

# -- consistent colour system ------------------------------------------------
C_HYBRID = "#4C72B0"
C_MEDCAT = "#55A868"
C_OPENMED = "#C44E52"
C_MANUAL = "#8C564B"   # pathologist annotation (warm clay, distinct from the
                       # three tools; used in the violin + agreement legends)
TOOLS_ORDER = ["Hybrid-Pipeline", "MedCAT (SNOMED)", "OpenMed"]
TOOL_COLORS = dict(zip(TOOLS_ORDER, [C_HYBRID, C_MEDCAT, C_OPENMED]))
# Series colours used in the "spans per report" and "agreement" panels:
# one color per series -- hybrid, medcat, openmed, pathologist (manual).
SERIES_ORDER = ["Hybrid-Pipeline", "MedCAT (SNOMED)", "OpenMed", "Pathologist"]
SERIES_COLORS = {
    "Hybrid-Pipeline": C_HYBRID,
    "MedCAT (SNOMED)": C_MEDCAT,
    "OpenMed": C_OPENMED,
    "Pathologist": C_MANUAL,
}

EXAMPLE_TEXT_ID = "case-17728_Histo Privat"

# Label palette shared by the HTML annotation and the figure legend.
LABEL_COLORS = {
    "probe": "#9CC3E4",
    "clinical": "#C5D5EC",
    "probe_size": "#F5D77F",
    "staining": "#A9DCAE",
    "micro_morphology": "#F0C4B4",
    "finding": "#F0C4B4",
    "diagnose": "#D3B6E5",
    "miscellaneous": "#DDDDDD",
}
LABEL_DISPLAY = {
    "probe": "probe",
    "clinical": "clinical",
    "probe_size": "probe size",
    "staining": "staining",
    "micro_morphology": "micro morphology",
    "finding": "finding",
    "diagnose": "diagnose",
    "miscellaneous": "misc",
}
# Order for the legend (most common / clinically most meaningful first).
LABEL_ORDER = [
    "diagnose", "clinical", "probe", "probe_size",
    "staining", "micro_morphology", "finding", "miscellaneous",
]


# =============================================================================
# Loading
# =============================================================================

def _load_tools():
    """Load the manual annotation and the tool exports.

    ``hybrid_ner.jsonl`` may hold several extraction runs per report (LLM
    sampling, ``meta.run_id``); those are kept as a list per report id so the
    metrics can aggregate over runs instead of silently keeping the last one.
    MedCAT/OpenMed are deterministic: any repeated runs of the same report are
    collapsed to the single run with the most entities.
    """
    manual: dict[str, dict] = {}
    with (HERE / "manual_ner.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("meta_type"):   # header rows carry no text_id
                continue
            manual[record["meta"]["text_id"]] = record

    def load(path: str) -> dict[int, list[dict]]:
        """rid -> list of records (one per extraction run of that report)."""
        out: dict[int, list[dict]] = {}
        with (HERE / path).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("meta_type"):
                    continue
                m = record.get("meta") or {}
                rid = m.get("report_id")
                if rid is None:
                    continue
                out.setdefault(int(rid), []).append(record)
        return out

    hybrid = load("hybrid_ner.jsonl")
    medcat = {rid: max(recs, key=lambda r: len(_medcat_spans(r)))
              for rid, recs in load("medcat_ner.jsonl").items()}
    openmed = {rid: max(recs, key=lambda r: len(_openmed_spans(r)))
               for rid, recs in load("openmed_ner.jsonl").items()}

    reports: dict[int, dict] = {}
    for rid, recs in hybrid.items():
        rec = recs[0]
        m = rec["meta"]
        reports[rid] = {
            "report_id": rid,
            "text_id": rec["text_id"],
            "text": rec["text"],
            "type": (m.get("berichtstyp") or "Sonstige").split(" ")[0],
            "year": m.get("jahr"),
        }
    return manual, hybrid, medcat, openmed, reports


# =============================================================================
# Span helpers
# =============================================================================

def _medcat_spans(rec: dict) -> list[dict]:
    return [
        {"start": e["start"], "end": e["end"],
         "mapped": bool(e.get("cui")), "acc": e.get("acc", 0.0)}
        for e in (rec.get("analyse") or {}).get("entities", {}).values()
    ]


def _openmed_spans(rec: dict) -> list[dict]:
    out = []
    for e in rec.get("extract", []):
        out.append({"start": e["start"], "end": e["end"]})
    for e in rec.get("analyse", []):
        out.append({"start": e["start"], "end": e["end"]})
    return out


def _hybrid_spans(rec: dict) -> list[dict]:
    return [
        {"start": s["start"], "end": s["end"], "text": s["text"],
         "label": s.get("label") or "unknown",
         "mapped": bool(s.get("concept_id"))}
        for s in rec.get("spans", [])
    ]


def _covered_by(manual_span: dict, tool_spans: list[dict]) -> bool:
    for t in tool_spans:
        if t["start"] <= manual_span["start"] and t["end"] >= manual_span["end"]:
            return True
    return False


def _chars_covered(manual_spans: list[dict], tool_spans: list[dict]) -> float:
    mset: set[int] = set()
    for s in manual_spans:
        mset.update(range(s["start"], s["end"]))
    if not mset:
        return 0.0
    tset: set[int] = set()
    for t in tool_spans:
        tset.update(range(t["start"], t["end"]))
    return len(mset & tset) / len(mset)


def _chars_covered_by_label(manual_spans: list[dict],
                            tool_spans: list[dict]) -> dict[str, tuple[int, int]]:
    """Per-label character-level overlap: for each manual span, count how many
    of its characters overlap with at least one tool span (union of tool
    characters within the manual span's range).

    Returns a dict: label -> (chars_covered, chars_total).
    """
    per_label: dict[str, list[int]] = {}
    for ms in manual_spans:
        label = ms.get("label") or "unknown"
        s, e = ms["start"], ms["end"]
        if not per_label.get(label):
            per_label[label] = [0, 0]
        per_label[label][1] += (e - s)
        # Build a set of positions covered by any tool span within this range
        covered = set()
        for t in tool_spans:
            lo = max(s, t["start"])
            hi = min(e, t["end"])
            if hi > lo:
                covered.update(range(lo, hi))
        per_label[label][0] += len(covered)
    return {k: (v[0], v[1]) for k, v in per_label.items()}


def build_label_agreement(manual: dict, hybrid: dict,
                          reports: dict) -> pd.DataFrame:
    """One row per manual annotation label:
      label, n_manual_spans, total_chars, covered_chars, pct_covered.

    ``hybrid`` maps report_id -> list of extraction-run records.  Agreement
    is computed per (report, run) pair and aggregated character-weighted, so
    ``pct`` is the mean character coverage over all runs of all reports.
    ``total_chars``/``covered_chars`` are sums over reports and runs.
    """
    accum: dict[str, int] = {}  # label -> [total, covered]
    counts: dict[str, int] = {}  # label -> n_manual_spans
    for rid, meta in reports.items():
        tid = meta["text_id"]
        if tid not in manual or rid not in hybrid:
            continue
        man_spans = [dict(s) for s in manual[tid]["spans"]]
        for s in man_spans:
            lb = s.get("label") or "unknown"
            counts[lb] = counts.get(lb, 0) + 1
        for rec in hybrid[rid]:        # one entry per extraction run
            per_label = _chars_covered_by_label(man_spans, _hybrid_spans(rec))
            for label, (cov, tot) in per_label.items():
                if tot <= 0:
                    continue
                accum.setdefault(label, [0, 0])
                accum[label][0] += tot
                accum[label][1] += cov
    rows = []
    for label, (tot, cov) in accum.items():
        rows.append({
            "label": label,
            "n_spans": counts.get(label, 0),
            "total_chars": tot,
            "covered_chars": cov,
            "pct": 100.0 * cov / tot if tot else 0.0,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("pct", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# Build per-report metrics
# =============================================================================

def _chars_annotated(spans: list[dict]) -> int:
    """Total characters covered by a list of spans (union of intervals)."""
    if not spans:
        return 0
    # Merge intervals to avoid double counting overlaps
    ivs = sorted((s["start"], s["end"]) for s in spans)
    merged = []
    for lo, hi in ivs:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return sum(hi - lo for lo, hi in merged)


def build_metrics(manual, hybrid, medcat, openmed, reports) -> pd.DataFrame:
    rows = []
    for rid, meta in reports.items():
        tid = meta["text_id"]
        if tid not in manual:
            continue
        man = manual[tid]
        man_spans = [dict(s) for s in man["spans"]]
        # Hybrid: one record per extraction run of this report; aggregate:
        # medians for counts, means for per-run agreement values.
        hy_lists = [_hybrid_spans(r) for r in (hybrid.get(rid) or [])]
        mc_spans = _medcat_spans(medcat[rid]) if rid in medcat else []
        om_spans = _openmed_spans(openmed[rid]) if rid in openmed else []
        mapped_spans = [s for s in mc_spans if s["mapped"]]

        # Length of the full report text (denominator for chars-per-report).
        rep_chars = len(man["text"]) if "text" in man else 0

        hy_n = [len(l) for l in hy_lists] or [0]
        hy_n_mapped = [sum(1 for s in l if s["mapped"]) for l in hy_lists] or [0]
        hy_frac = [(_chars_annotated(l) / rep_chars) if rep_chars else 0.0
                   for l in hy_lists] or [0.0]
        hy_cov = [_chars_covered(man_spans, l) for l in hy_lists] or [0.0]
        hy_match = [((sum(1 for s in man_spans if _covered_by(s, l))
                      / len(man_spans)) if man_spans else 0.0)
                    for l in hy_lists] or [0.0]

        rows.append({
            "report_id": rid,
            "text_id": tid,
            "type": meta.get("type", "Sonstige"),  # kept for ad-hoc slices
            "n_manual": len(man_spans),
            "n_runs": len(hy_lists),
            "n_hybrid": int(round(float(np.median(hy_n)))),
            "n_medcat": len(mc_spans),
            "n_openmed": len(om_spans),
            "report_chars": rep_chars,
            "hybrid_matches_pct": float(np.mean(hy_match)),
            "medcat_matches": sum(1 for s in man_spans if _covered_by(s, mc_spans)),
            "medcat_mapped_matches": sum(
                1 for s in man_spans
                if any(m["start"] <= s["start"] and m["end"] >= s["end"]
                       for m in mapped_spans)),
            "openmed_matches": sum(1 for s in man_spans if _covered_by(s, om_spans)),
            # Fraction of all report characters that each source annotated:
            "hybrid_frac_chars": float(np.median(hy_frac)),
            "medcat_frac_chars": (_chars_annotated(mc_spans) / rep_chars) if rep_chars else 0.0,
            "openmed_frac_chars": (_chars_annotated(om_spans) / rep_chars) if rep_chars else 0.0,
            "manual_frac_chars": (_chars_annotated(man_spans) / rep_chars) if rep_chars else 0.0,
            # Fraction of MANUALLY-annotated characters covered by each tool
            # (agreement metric; used for panel B and printed summary):
            "hybrid_chars": float(np.mean(hy_cov)),
            "medcat_chars": _chars_covered(man_spans, mc_spans),
            "medcat_mapped_chars": _chars_covered(man_spans, mapped_spans),
            "openmed_chars": _chars_covered(man_spans, om_spans),
            "n_hybrid_mapped": int(round(float(np.median(hy_n_mapped)))),
            "n_medcat_mapped": len(mapped_spans),
        })
    df = pd.DataFrame(rows)
    if df.shape[0] == 0:
        return df
    # hybrid_matches_pct is computed per report above (mean over its runs).
    df["medcat_matches_pct"] = np.where(df["n_manual"] > 0,
                                        df["medcat_matches"] / df["n_manual"], 0.0)
    df["medcat_mapped_pct"] = np.where(df["n_manual"] > 0,
                                       df["medcat_mapped_matches"] / df["n_manual"], 0.0)
    df["openmed_matches_pct"] = np.where(df["n_manual"] > 0,
                                         df["openmed_matches"] / df["n_manual"], 0.0)
    return df


# =============================================================================
# Panel A: HTML annotation of the example report (HYBRID spans)
# =============================================================================

def _merge_spans(text: str, spans: list[dict]) -> list[dict]:
    """Merge overlapping/adjacent hybrid spans into clean highlight segments.

    Returns a list of {start, end, label, any_mapped} sorted by start, with
    no overlaps and no adjacent segments of the same visual class.
    """
    events: list[tuple[int, int, int, bool, str]] = []
    for s in spans:
        events.append((s["start"], 1, 0, s["mapped"], s["label"]))
    # Use an interval sweep: collect all segment boundaries.
    active = 0
    boundaries = sorted({s["start"] for s in spans} | {s["end"] for s in spans})
    # For each boundary interval, decide which spans cover it.
    out: list[dict] = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        covering = [s for s in spans if s["start"] <= lo and s["end"] >= hi]
        if not covering:
            continue
        # label = the label of the span with the highest priority among covering
        label = covering[0]["label"]
        any_mapped = any(s["mapped"] for s in covering)
        # pick the label with the largest span as "primary"
        label = max(covering, key=lambda s: s["end"] - s["start"])["label"]
        out.append({"start": lo, "end": hi, "label": label, "mapped": any_mapped})
    # merge segments that are adjacent and identical in (label, mapped)
    merged: list[dict] = []
    for seg in out:
        if merged and (merged[-1]["end"] == seg["start"]
                       and merged[-1]["label"] == seg["label"]
                       and merged[-1]["mapped"] == seg["mapped"]):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def _render_annotation_html(text: str, spans: list[dict], title: str,
                            subtitle: str,
                            width_in: float, height_in: float) -> str:
    # Sentinels (``#AFFILIATIONLINE``) are kept to preserve character offsets
    # for span matching, but rendered visually as a small marker.
    _SENTINEL = "#AFFILIATIONLINE"
    segs = _merge_spans(text, spans)
    seg_by_pos = {}
    for s2 in segs:
        for i in range(s2["start"], s2["end"]):
            seg_by_pos[i] = s2

    # Build the inner HTML of the body char by char, grouping contiguous
    # characters that share the same segment into a single <mark> element.
    pieces: list[str] = []
    cursor = 0
    n = len(text)
    while cursor < n:
        ch = text[cursor]
        if ch == "\n":
            pieces.append("<br/>")
            cursor += 1
            continue
        # Sentinel line: render as a subtle marker, skip to end of line.
        if text[cursor:cursor + len(_SENTINEL)] == _SENTINEL:
            # emit the sentinel as a subtle, small marker
            pieces.append('<span class="sentinel">[anonymisation marker]</span>')
            # advance past sentinel + optional trailing newline
            cursor += len(_SENTINEL)
            if cursor < n and text[cursor] == "\n":
                pieces.append("<br/>")
                cursor += 1
            continue
        s2 = seg_by_pos.get(cursor)
        if s2 is None:
            # collect a run of plain characters (stop at sentinel too)
            j = cursor
            while j < n and text[j] != "\n" and seg_by_pos.get(j) is None:
                if text[j:j + len(_SENTINEL)] == _SENTINEL:
                    break
                j += 1
            pieces.append(_esc(text[cursor:j]))
            cursor = j
            continue
        # collect a run of characters all in the SAME segment
        j = cursor
        while (j < n and text[j] != "\n"
               and seg_by_pos.get(j) is not None
               and seg_by_pos.get(j) is s2):
            j += 1
        seg = s2
        colour = LABEL_COLORS.get(seg["label"], "#DDDDDD")
        label = LABEL_DISPLAY.get(seg["label"], seg["label"])
        marker = " \u25CF" if seg["mapped"] else ""
        tip = f" {seg['label']} - concept-mapped" if seg["mapped"] else f" {seg['label']}"
        pieces.append(
            f'<mark class="seg" style="background:{colour}99;'
            f'box-shadow:inset 0 -0.14em 0 0 {LABEL_COLORS.get(seg["label"], "#ccc")};"'
            f' data-label="{label}">{_esc(text[cursor:j])}</mark>'
        )
        cursor = j

    # legend (in LABEL_ORDER, only labels that actually appear)
    present = {seg["label"] for seg in segs}
    leg_items = []
    for lb in LABEL_ORDER:
        if lb not in present:
            continue
        leg_items.append(
            f'<span class="leg"><span class="sw" '
            f'style="background:{LABEL_COLORS[lb]}"></span>'
            f'{LABEL_DISPLAY.get(lb, lb)}</span>'
        )
    legend = "".join(leg_items)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
@page {{ size: {width_in}in {height_in}in; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: #fff; }}
body {{
  font-family: "Liberation Sans", "DejaVu Sans", Arial, sans-serif;
  color: #1c2330;
  width: {width_in}in;
  min-height: 100%;
  padding: 0.2in 0.25in 0.14in 0.25in;
  display: flex;
  flex-direction: column;
}}
.hdr {{ margin-bottom: 0.1em; }}
.title {{ font-size: 13.5px; font-weight: 700; color: #111827; margin: 0; }}
.subtitle {{ font-size: 10.5px; color: #4b5563; margin: 0.15em 0 0 0; }}
.subtle {{ color: #6b7280; }}
.legend {{
  display: flex; flex-wrap: wrap; gap: 0.28em 0.5em;
  margin: 0.28em 0 0.45em 0;
  padding: 0.18em 0.3em;
  border: 0.55px solid #e5e7eb; border-radius: 4px; background: #f9fafb;
}}
.leg {{ display: inline-flex; align-items: center; font-size: 10px; color: #374151; }}
.sw {{
  display: inline-block; width: 0.85em; height: 0.72em; margin-right: 0.3em;
  border: 0.5px solid #9aa3b2; border-radius: 2px;
}}
.report {{
  font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
  font-size: 10.4px; line-height: 1.78;
  color: #1c2330;
  white-space: normal; overflow-wrap: break-word;
  border: 0.55px solid #e2e5ea; border-radius: 4px;
  padding: 0.3em 0.4em;
  background: #fcfcfd;
}}
mark.seg {{ color: #131a24; border-radius: 2px; padding: 0 0.02em; }}
.sentinel {{
  color: #b7bcc5; font-size: 0.78em; font-style: italic; letter-spacing: 0.02em;
  display: inline-block; margin: 0.05em 0 0.05em 0;
}}
.note {{ font-size: 9px; color: #6b7280; margin-top: 0.28em; }}
.foot {{
  font-size: 9px; color: #9aa3b2; margin-top: 0.28em;
}}
</style>
</head>
<body>
  <div class="hdr">
    <p class="title">C. Annotated example -- hybrid-pipeline output on one report</p>
    <p class="subtitle">{_esc(subtitle)}</p>
  </div>
  <div class="legend">{legend}
     <span class="leg"><span class="dot">\u25CF</span> concept-mapped</span>
  </div>
  <div class="report">
    {''.join(pieces)}
  </div>
  <p class="note">Hybrid-pipeline span highlights on the original report text;
     highlight colour = clinical label; a \u25CF after a span marks that the span was
     also resolved to a clinical concept (SNOMED/UMLS).</p>
  <p class="foot">n = 1 of 100 reports (chosen as an illustrative example)</p>
</body>
</html>"""
    return html


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _strip_sentinels(text: str) -> str:
    """Remove DB-export sentinel lines (``#AFFILIATIONLINE``) that are never
    part of the original report."""
    lines = text.split("\n")
    out = []
    for ln in lines:
        if ln.strip() == "#AFFILIATIONLINE":
            continue
        out.append(ln)
    # collapse any empty lines left at the end
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _find_chrome() -> str:
    for cand in ("google-chrome-stable", "google-chrome", "chromium",
                 "chromium-browser", "chrome"):
        p = shutil.which(cand)
        if p:
            return p
    raise SystemExit("No headless Chrome/Chromium binary found on PATH.")


def _chrome_to_pdf(html: str, out_pdf: Path, width_in: float, height_in: float) -> Path:
    """Render HTML -> PDF using headless Chrome at a fixed page size.

    The page size is controlled by the ``@page size: WxH;`` CSS rule in the
    HTML.  The window size is set to ``width_in`` (to control text wrapping)
    with a generous height so the content does not overflow the page.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(html)
        tmp_html = Path(fh.name)
    chrome = _find_chrome()
    cmd = [
        chrome, "--headless=new", "--no-sandbox",
        "--disable-gpu", "--hide-scrollbars",
        f"--window-size={int(width_in * 96)},{int(height_in * 96)}",
        f"--print-to-pdf={out_pdf}",
        "--no-pdf-header-footer",
        str(tmp_html),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not out_pdf.exists():
        raise SystemExit(f"Chrome failed to produce PDF:\n"
                         f"stdout: {res.stdout[-400:]}\nstderr: {res.stderr[-800:]}")
    tmp_html.unlink(missing_ok=True)
    return out_pdf


def _measure_body_height(html: str, width_in: float,
                         probe_height_in: float = 9.0) -> float:
    """Use Chrome (headless) to evaluate the natural body height at the
    requested width.  Returns the height in inches (CSS px -> 96 px/in)."""
    # Neutralise the min-height so the measured scrollHeight reflects natural
    # content height.
    import re as _re
    probe = _re.sub(r"min-height:\s*[\d.]+in", "min-height: auto", html, count=1)
    # Inject a small script that records scrollHeight in the page <title>.
    probe = probe.replace(
        "</head>",
        '<script>window.addEventListener("load", function(){'
        "document.title = Math.round(document.body.scrollHeight);"
        "});</script></head>",
        1,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(probe)
        tmp_html = Path(fh.name)
    chrome = _find_chrome()
    cmd = [
        chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
        "--virtual-time-budget=1500",
        f"--window-size={int(width_in * 300)},{int(probe_height_in * 300)}",
        "--dump-dom",
        str(tmp_html),
    ]
    import re
    dom = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=60).stdout
    m = re.search(r"<title>(\d+)</title>", dom)
    tmp_html.unlink(missing_ok=True)
    if not m:
        return probe_height_in
    h_px = int(m.group(1))
    h_in = h_px / 96.0 + 0.05  # small safety margin
    return max(h_in, 0.6)


def _trim_trailing_empty_pages(pdf: Path, out: Path) -> Path:
    """Drop trailing blank pages, keeping the pages with content.

    Uses ``pdfinfo`` for the page count and ``pdftoppm`` (PNG, low res) to
    test each page for dark pixels."""
    import re
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    mo = re.search(r"Pages:\s+(\d+)", info.stdout)
    n_pages = int(mo.group(1)) if mo else 1
    if n_pages <= 1:
        shutil.copyfile(pdf, out)
        return out
    last_content_page = 1
    with tempfile.TemporaryDirectory() as td:
        for p in range(1, n_pages + 1):
            prefix = Path(td) / "pg"
            subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p),
                            "-r", "30", "-png", str(pdf), str(prefix)],
                           check=True, capture_output=True)
            # pdftoppm writes <prefix>-<pnum>.png  (or -01, -02, ...)
            cand = Path(td) / f"pg-{p}.png"
            if not cand.exists():
                cand = Path(td) / f"pg-0{p}.png"
            if not cand.exists():
                cand = Path(td) / f"pg-00{p}.png"
            if cand.exists():
                from PIL import Image
                a = np.array(Image.open(cand).convert("L"))
                dark = (a < 180).sum()
                if dark > 0:
                    last_content_page = p
        if last_content_page < n_pages:
            subprocess.run(
                ["gs", "-dBATCH", "-dNOPAUSE", "-dSAFER",
                 "-sDEVICE=pdfwrite",
                 f"-dFirstPage=1", f"-dLastPage={last_content_page}",
                 f"-sOutputFile={out}", str(pdf)],
                check=True, capture_output=True)
        else:
            shutil.copyfile(pdf, out)
    return out


def _pdf_to_png(pdf: Path, png: Path) -> Path:
    for tool in ("pdftoppm", "gs", "mutool"):
        if shutil.which(tool):
            break
    else:
        return pdf
    if tool == "pdftoppm":
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "300",
                        str(pdf), str(png.with_suffix(""))],
                       check=True, capture_output=True)
    elif tool == "gs":
        subprocess.run(["gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
                        "-r300", "-o", str(png), str(pdf)],
                       check=True, capture_output=True)
    else:
        subprocess.run(["mutool", "draw", "-o", str(png), "-r", "300",
                        str(pdf)], check=True, capture_output=True)
    if not png.exists():
        raise SystemExit(f"Rasterizer {tool} failed to produce {png}")
    return png




# =============================================================================
# Panel B: entities per report by tool
# =============================================================================

def _kde(v, bw, grid):
    w = np.zeros_like(grid)
    for x in v:
        w += np.exp(-0.5 * ((grid - x) / bw) ** 2)
    w = w / (len(v) * bw * sqrt(2 * np.pi) + 1e-12)
    return w


def draw_panel_b(ax, df: pd.DataFrame) -> None:
    """Panel A (top-left): percentage of each report's characters that each
    source annotates, shown as violins across reports.

    Each value per report is  (chars annotated by source) / (report chars) * 100.
    """
    series_values: dict[str, np.ndarray] = {
        "Hybrid-Pipeline": df["hybrid_frac_chars"].to_numpy(dtype=float),
        "MedCAT (SNOMED)": df["medcat_frac_chars"].to_numpy(dtype=float),
        "OpenMed":         df["openmed_frac_chars"].to_numpy(dtype=float),
        "Pathologist":     df["manual_frac_chars"].to_numpy(dtype=float),
    }
    slot = 0.34  # horizontal offset between series within one slot

    # Cap the y-axis at a robust upper percentile (99th) instead of the raw
    # max so that a single outlier report cannot squash the rest of the
    # distribution flat and leave the panel mostly empty.  Points above the
    # cap are clipped by the axis, which matches the "fill the panel" intent.
    vmax_raw = max(float(v.max()) for v in series_values.values())
    p99s = [float(np.percentile(v, 99)) for v in series_values.values()]
    p99 = max(p99s)
    vmax = max(min(vmax_raw, p99 * 2.75), 0.5)   # robust cap, keep small min
    vpad = max(vmax * 0.09, 0.05)

    for j, name in enumerate(SERIES_ORDER):
        x = (j - 1.5) * slot
        vals = series_values[name]
        if vals.size == 0:
            continue
        if vals.size == 1:
            ax.axhline(float(vals[0]), xmin=0, xmax=1,
                       color=SERIES_COLORS[name], ls=":", lw=0.9, alpha=0.9)
            continue
        scale = max(float(np.std(vals, ddof=1)), 1e-6)
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        ref = scale if iqr <= 0 else min(scale, iqr / 1.34)
        bw = max(0.9 * ref, 1e-2)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-6:
            hi = lo + 0.5
        grid = np.linspace(lo - bw, hi + bw, 160)
        w = _kde(vals, bw, grid)
        wmax = w.max()
        if wmax <= 0:
            continue
        # Normalise width to a fraction of the available slot
        w = w / wmax * (slot * 0.92)
        left = np.column_stack([x - w, grid])
        right = np.column_stack([x + w, grid])
        poly = np.vstack([left, right[::-1]])
        ax.fill(*poly.T, facecolor=SERIES_COLORS[name], alpha=0.55,
                edgecolor=SERIES_COLORS[name], linewidth=0.9, zorder=2)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([x - w.max()*0.45, x + w.max()*0.45], [med, med],
                color="black", lw=1.4, zorder=6)
        half = 0.08
        ax.vlines(x, q1, q3, color="0.25", lw=0.8, zorder=5)
        ax.hlines(q1, x - half, x + half, color="0.25", lw=0.7, zorder=5)
        ax.hlines(q3, x - half, x + half, color="0.25", lw=0.7, zorder=5)

    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(0, vmax + vpad)
    # One labelled tick per series, centred under each violin.
    xticks = [(j - 1.5) * slot for j in range(len(SERIES_ORDER))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(SERIES_ORDER, fontsize=7.2, rotation=22, ha="right")
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("Chars annotated per report (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [Patch(facecolor=SERIES_COLORS[t], edgecolor="none", alpha=0.85)
               for t in SERIES_ORDER]
    # ax.legend(handles=handles, labels=SERIES_ORDER, loc="upper right",
    #           fontsize=7.2, frameon=False, handletextpad=0.5,
    #           handlelength=1.2)


# =============================================================================
# Panel C: agreement with manual annotation
# =============================================================================

def _display_label(lbl: str) -> str:
    return LABEL_DISPLAY.get(lbl, lbl)


def draw_panel_c(ax, label_df: pd.DataFrame, n_reports: int,
                 n_manual_spans: int) -> None:
    """Panel B (bottom-left): agreement of the Hybrid Pipeline with the
    junior pathologist's manual annotation, **split by annotation label**.

    One bar per manual annotation label.  Bar height = the fraction of that
    label's total annotated characters that overlap with at least one hybrid
    span (character-level union), averaged across all reports.
    """
    if label_df.shape[0] == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return

    # Filter out labels with very few manual spans (noisy percentages)
    # and keep top-N for legibility.
    MIN_SPANS = 20
    df = label_df[label_df["n_spans"] >= MIN_SPANS]
    top_n = min(len(df), 8)
    df = df.head(top_n).copy()

    labels = [_display_label(lb) for lb in df["label"]]
    values = df["pct"].to_numpy(dtype=float)
    ns = df["n_spans"].to_numpy(dtype=int)

    xs = list(range(len(df)))
    width = 0.58
    ax.bar(xs, values, width=width, color=SERIES_COLORS["Hybrid-Pipeline"],
           edgecolor="white", lw=0.7, alpha=0.92, zorder=3)

    vmax = float(values.max()) if values.size else 1.0
    for x, v, n in zip(xs, values, ns):
        ax.text(x, v + 0.9, f"{v:.1f}%", ha="center",
                fontsize=6.6, color="0.25", zorder=5, rotation=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.2, rotation=22, ha="right")
    ax.set_ylabel("Agreement (%)", fontsize=9)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_ylim(0, max(vmax * 1.28, 30))
    ax.tick_params(axis="x", length=0, labelsize=6.8, pad=4)
    ax.tick_params(axis="y", labelsize=8.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(0.999, -0.42,
            f"Hybrid pipeline vs. junior pathologist · "
            f"per label · {n_reports} reports, {n_manual_spans} manual spans",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.0, color="#55606e", style="italic")


# =============================================================================
# Final combined figure (left=BC, right=A)
# =============================================================================

def _assemble_combined(annot_pdf: Path, panels_png: Path,
                       out_pdf: Path, out_png: Path) -> bool:
    """Assemble a single combined figure: the two quantitative panels (B and
    C) on the left and the HTML annotation (A) on the right.

    Both are rendered as PNGs at a common DPI and then placed side by side
    on a single white canvas.  The result is written as PNG and, when
    ``reportlab`` is available, also as a single-page PDF.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("WARN: Pillow not available -- skipping combined figure.")
        return False

    DPI = 300
    # Rasterise both inputs at DPI.
    annot_img = _png_from_pdf(annot_pdf, DPI)
    if annot_img is None:
        print("WARN: could not rasterise annotation PDF -- skipping combined "
              "figure.")
        return False
    panels = Image.open(panels_png).convert("RGB")

    # Scale both to a common height (pixels).
    target_h = max(annot_img.height, panels.height)
    def scale_to_h(im: Image.Image) -> Image.Image:
        if im.height == target_h:
            return im
        w = int(im.width * (target_h / im.height))
        return im.resize((w, target_h), Image.LANCZOS)
    annot_img = scale_to_h(annot_img)
    panels = scale_to_h(panels)

    gap = int(0.12 * DPI)
    canvas = Image.new("RGB",
                       (panels.width + gap + annot_img.width, target_h),
                       "white")
    canvas.paste(panels, (0, 0))
    canvas.paste(annot_img, (panels.width + gap, 0))
    canvas.save(out_png, dpi=(DPI, DPI))

    # Try to also write a single-page PDF with Pillow's PDF plugin (it
    # embeds the image losslessly).  If Pillow cannot write PDF we skip it.
    try:
        canvas.save(out_pdf, "PDF", resolution=DPI)
    except Exception as e:
        out_pdf.unlink(missing_ok=True)
        print(f"WARN: could not write combined PDF ({e}) -- PNG only.")
    return True


def _png_from_pdf(pdf: Path, dpi: int = 300):
    """Rasterise page 1 of a PDF at the given DPI and return a PIL image."""
    import shutil as _sh
    if not _sh.which("pdftoppm"):
        return None
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "pg"
        try:
            subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-r",
                            str(dpi), "-png", str(pdf), str(prefix)],
                           check=True, capture_output=True, timeout=90)
        except Exception:
            return None
        for name in ("pg-1.png", "pg-01.png", "pg-001.png"):
            cand = Path(td) / name
            if cand.exists():
                from PIL import Image as _Img
                return _Img.open(cand).convert("RGB")
    return None


# =============================================================================
# main
# =============================================================================

# Right column (annotation panel) page size in inches.  Height is chosen
# large enough that the full example report (including legend + footnotes)
# fits on a single page at the body font/line-height below.
ANNOT_W = 4.3
ANNOT_H = 11.0


def main():
    manual, hybrid, medcat, openmed, reports = _load_tools()
    ex_rid = next((r["report_id"] for r in reports.values()
                   if r["text_id"] == EXAMPLE_TEXT_ID), None)
    if ex_rid is None:
        raise SystemExit(f"Example report {EXAMPLE_TEXT_ID} not in reports.")
    ex_report = reports[ex_rid]
    # Several extraction runs exist per report (LLM sampling).  Show the run
    # whose spans best match the pathologist's manual annotation for this
    # report -- the most representative single set of spans for the figure.
    ex_runs = hybrid[ex_rid]
    ex_man_spans = ([dict(s) for s in manual[EXAMPLE_TEXT_ID]["spans"]]
                    if EXAMPLE_TEXT_ID in manual else [])
    ex_hybrid = max(
        ex_runs,
        key=lambda r: (_chars_covered(ex_man_spans, _hybrid_spans(r)),
                       len(_hybrid_spans(r)))) if ex_runs else None
    hy_spans = _hybrid_spans(ex_hybrid) if ex_hybrid else []
    n_ex_runs = len(ex_runs)
    run_note = (f"best of {n_ex_runs} extraction runs: " if n_ex_runs > 1 else "")
    subtitle = (f"{ex_report['text_id']}  |  {ex_report['year']}  |  "
                f"{run_note}{len(hy_spans)} hybrid spans "
                f"({sum(1 for s in hy_spans if s['mapped'])} concept-mapped)")

    df = build_metrics(manual, hybrid, medcat, openmed, reports)
    print(f"\nTest set : {len(reports)} reports, matched to manual: {len(df)} "
          f"reports, {int(df['n_manual'].sum())} manual spans.")
    print("\nAgreement with the junior pathologist (mean %):")
    for t in sorted(df["type"].unique()):
        sel = df[df["type"] == t]
        n = int(sel.shape[0])
        print(f"  {t:<12} n={n:<4}  HYB "
              f"{sel['hybrid_matches_pct'].mean() * 100:5.1f}%   MCD "
              f"{sel['medcat_matches_pct'].mean() * 100:5.1f}%"
              f" (mapped {sel['medcat_mapped_pct'].mean() * 100:.1f}%)   OMD "
              f"{sel['openmed_matches_pct'].mean() * 100:5.1f}%")
    print(f"\nMedCAT spans total: {int(df['n_medcat'].sum())} (mapped "
          f"{int(df['n_medcat_mapped'].sum())})")
    all_run_n = [len(_hybrid_spans(r)) for runs in hybrid.values() for r in runs]
    if all_run_n:
        rq1, rq3 = (float(np.percentile(all_run_n, 25)),
                    float(np.percentile(all_run_n, 75)))
        print(f"Hybrid runs: {len(all_run_n)} extraction runs over "
              f"{len(hybrid)} reports; per-run spans median="
              f"{float(np.median(all_run_n)):.0f} (IQR {rq1:.0f}-{rq3:.0f})")
    print(f"Hybrid spans/report (median over runs): "
          f"sum={int(df['n_hybrid'].sum())} (mapped "
          f"{int(df['n_hybrid_mapped'].sum())})")
    print(f"OpenMed spans total: {int(df['n_openmed'].sum())}")

    n_hy = len(hy_spans)
    n_mapped = sum(1 for s in hy_spans if s["mapped"])
    print(f"\nExample report hybrid annotation ({n_ex_runs} run(s)): "
          f"{n_hy} spans, {n_mapped} concept-mapped")

    # -- Panel C (HTML -> PDF -> PNG) ----------------------------------------
    html = _render_annotation_html(ex_report["text"], hy_spans,
                                   "C. Annotated example",
                                   subtitle, ANNOT_W, ANNOT_H)
    html_path = HERE / "ner_annotation_example.html"
    html_path.write_text(html, encoding="utf-8")
    annot_raw = HERE / "ner_annotation_raw.pdf"
    annot_pdf = HERE / "ner_annotation_example.pdf"
    annot_png = HERE / "ner_annotation_example.png"
    # Render at the configured page size; then drop any trailing blank pages.
    _chrome_to_pdf(html, annot_raw, ANNOT_W, ANNOT_H)
    _trim_trailing_empty_pages(annot_raw, annot_pdf)
    annot_raw.unlink(missing_ok=True)
    print(f"Wrote HTML: {html_path}")
    print(f"Wrote PDF  : {annot_pdf}")
    try:
        _pdf_to_png(annot_pdf, annot_png)
        print(f"Wrote PNG  : {annot_png}")
    except Exception as e:  # not fatal -- PDF is the artefact
        print(f"WARN: could not rasterise annotation PDF: {e}")

    # -- Panels A and B (matplotlib) -----------------------------------------
    label_df = build_label_agreement(manual, hybrid, reports)
    n_man_spans_total = int(label_df["n_spans"].sum())
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(4.3, 5.0), constrained_layout=False,
        gridspec_kw={"hspace": 0.72})
    draw_panel_b(ax_a, df)
    ax_a.set_title("A. Fraction of each report annotated by source",
                   fontsize=10, loc="left", pad=8)
    draw_panel_c(ax_b, label_df, len(df), n_man_spans_total)
    ax_b.set_title("B. Agreement with the junior pathologist",
                   fontsize=10, loc="left", pad=8)
    left_png = HERE / "ner_figure_panels_ab.png"
    left_pdf = HERE / "ner_figure_panels_ab.pdf"
    fig.savefig(left_png, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.12)
    fig.savefig(left_pdf, bbox_inches="tight", facecolor="white",
                pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote: {left_png}\nWrote: {left_pdf}")

    final_dir = HERE
    combined_pdf = HERE / "ner_figure_combined.pdf"
    combined_png = HERE / "ner_figure_combined.png"
    if _assemble_combined(annot_pdf, left_png, combined_pdf, combined_png):
        print(f"Wrote: {combined_pdf}")
        print(f"Wrote: {combined_png}\n")

    print("\nArtefacts:")
    print(f"  Panel A (spans/report violin):   in {left_pdf}")
    print(f"  Panel B (agreement bars):        in {left_pdf}")
    print(f"  Panel C (annotation, HTML/PDF):  {annot_pdf}")
    print(f"  Panel C (HTML source):           {html_path}")
    print(f"  Final combined (left=AB, right=C): {combined_pdf}")


if __name__ == "__main__":
    main()

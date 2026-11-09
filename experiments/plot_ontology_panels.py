#!/usr/bin/env python3
"""Compare ontology extraction and label linkage."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- one font family + one base size for the whole figure -------------------
FONT = "Liberation Sans"
FS_PT = 11                   # single point size for ALL matplotlib text
REPORT_PX = 15               # text size (CSS px) for the HTML report panel
RIGHT_FIGSIZE_W = 4.5        # common width (in) for right-hand panels B and C
RIGHT_FIGSIZE_H = 3.5

HERE = Path(__file__).resolve().parent
DEFAULT_TEXT_ID = "case-17728_Histo Privat"

DPI = 220
GAP = 40                     # pixels between the panels

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [FONT, "DejaVu Sans", "sans-serif"]
plt.rcParams["font.size"] = FS_PT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = FS_PT
plt.rcParams["axes.labelsize"] = FS_PT
plt.rcParams["xtick.labelsize"] = FS_PT
plt.rcParams["ytick.labelsize"] = FS_PT
plt.rcParams["legend.fontsize"] = FS_PT

# Provider order + display + palette (mirrors compare_ontologies.py).
PROVIDERS = ["umls_remote", "umls_local", "snomed_de", "snomed_intl"]
PROV_DISPLAY = {
    "umls_remote": "UMLS (remote)", "umls_local": "UMLS (local)",
    "snomed_de": "SNOMED (DE)", "snomed_intl": "SNOMED (INTL)",
}
PROV_SHORT = {"umls_remote": "UMLS-R", "umls_local": "UMLS-L",
              "snomed_de": "SN-D", "snomed_intl": "SN-I"}
PROV_PALETTE = ["#4C72B0", "#55A868", "#8172B3", "#DD8452"]

# Clinical label palette (report highlights + panel C bars).
LABEL_COLORS = {
    "probe": "#9CC3E4", "probe_type": "#F5D77F", "clinical": "#C5D5EC",
    "probe_size": "#F5D77F", "staining": "#A9DCAE",
    "finding": "#F0C4B4", "diagnose": "#D3B6E5", "negation": "#F3B7B7",
    "macro_morphology": "#F0C4B4", "macro_element": "#CFE0F0",
    "micro_morphology": "#F0C4B4", "micro_element": "#CFE0F0",
    "localisation": "#BFD8F0", "ihc_marker": "#C9E9C9",
    "miscellaneous": "#DDDDDD", "unknown": "#DDDDDD",
}
LABEL_DISPLAY = {
    "probe": "PROBE", "probe_type": "PROBE TYPE", "clinical": "CLINICAL",
    "probe_size": "PROBE SIZE", "staining": "STAINING", "finding": "FINDING",
    "diagnose": "DIAGNOSE", "negation": "NEGATION",
    "macro_morphology": "MACRO MORPH", "micro_morphology": "MICRO MORPH",
    "macro_element": "MACRO ELEMENT", "micro_element": "MICRO ELEMENT",
    "localisation": "LOCALISATION", "ihc_marker": "IHC MARKER",
    "miscellaneous": "MISC",
}


# =============================================================================
# Data
# =============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def load_report_spans(text_id: str) -> tuple[str, list[dict]]:
    records: list[dict] = []
    for rec in _load_jsonl(HERE / "hybrid_ner.jsonl"):
        if rec.get("meta_type") or rec.get("text_id") != text_id:
            continue
        spans = [s for s in rec.get("spans", [])
                 if isinstance(s, dict) and "start" in s and "end" in s]
        records.append({"text": rec["text"], "spans": spans})
    if not records:
        raise SystemExit(f"Report {text_id!r} not found in hybrid_ner.jsonl")
    best = max(records, key=lambda r: (sum(1 for s in r["spans"] if s.get("concept_id")),
                                       len(r["spans"])))
    spans = [{"start": s["start"], "end": s["end"], "text": s.get("text", ""),
              "label": s.get("label") or "unknown"} for s in best["spans"]]
    return best["text"], spans


def load_coverage(drop_zero: bool = True) -> dict[int, dict[str, float | None]]:
    """Per-report, per-provider coverage = (resolved spans / total spans).

    With ``drop_zero`` (default True) reports that had a total of zero resolved
    spans on EVERY backend are omitted from the per-report distributions, so a
    handful of empty reports do not pin the lower whisker to zero and dominate
    the visual spread.  The overall resolved/total tallies keep every record.
    """
    rep = defaultdict(lambda: {p: [0, 0] for p in PROVIDERS})
    for r in _load_jsonl(HERE / "hybrid_ner_mappings.jsonl"):
        rid = int(r["report_id"])
        m = r.get("mappings") or {}
        for p in PROVIDERS:
            out = m.get(p) or {}
            rep[rid][p][0] += 1 if out.get("concept_id") else 0
            rep[rid][p][1] += 1
    cov = {rid: {p: (res / tot if tot else None) for p, (res, tot) in per.items()}
           for rid, per in rep.items()}
    if drop_zero:
        cov = {rid: per for rid, per in cov.items()
               if any((per[p] or 0) > 0 for p in PROVIDERS)}
    return cov


def load_per_label() -> list[tuple[str, int, int]]:
    d = defaultdict(lambda: [0, 0])
    for r in _load_jsonl(HERE / "hybrid_ner_mappings.jsonl"):
        lab = (r.get("span") or {}).get("label", "unknown")
        m = r.get("mappings") or {}
        d[lab][0] += 1
        if any((m.get(p) or {}).get("concept_id") for p in PROVIDERS):
            d[lab][1] += 1
    return [(lab, n, res) for lab, (n, res) in d.items()]


# =============================================================================
# Panel A: annotated report (HTML -> PDF -> PIL image)
# =============================================================================

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_report_html(text: str, spans: list[dict], subtitle: str,
                       width_in: float, height_in: float) -> str:
    sentinel = "#AFFILIATIONLINE"
    boundaries = sorted({s["start"] for s in spans} | {s["end"] for s in spans})
    segs = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        covering = [s for s in spans if s["start"] <= lo and s["end"] >= hi]
        if not covering:
            continue
        label = max(covering, key=lambda s: s["end"] - s["start"])["label"]
        segs.append({"start": lo, "end": hi, "label": label})
    seg_by_pos = {}
    for s2 in segs:
        for i in range(s2["start"], s2["end"]):
            seg_by_pos[i] = s2

    pieces: list[str] = []
    cursor = 0
    nch = len(text)
    while cursor < nch:
        if text[cursor] == "\n":
            pieces.append("<br/>"); cursor += 1; continue
        if text[cursor:cursor + len(sentinel)] == sentinel:
            pieces.append('<span class="sentinel">[anonymisation marker]</span>')
            cursor += len(sentinel)
            if cursor < nch and text[cursor] == "\n":
                pieces.append("<br/>"); cursor += 1
            continue
        s2 = seg_by_pos.get(cursor)
        if s2 is None:
            j = cursor
            while j < nch and text[j] != "\n" and seg_by_pos.get(j) is None:
                if text[j:j + len(sentinel)] == sentinel:
                    break
                j += 1
            pieces.append(_esc(text[cursor:j])); cursor = j; continue
        j = cursor
        while (j < nch and text[j] != "\n"
               and seg_by_pos.get(j) is not None and seg_by_pos.get(j) is s2):
            j += 1
        colour = LABEL_COLORS.get(s2["label"], "#DDDDDD")
        pieces.append(
            f'<mark class="seg" style="background:{colour};'
            f'box-shadow:inset 0 -0.15em 0 0 {colour};">{_esc(text[cursor:j])}</mark>'
        )
        cursor = j

    present = {s["label"] for s in spans}
    leg = "".join(
        f'<span class="leg"><span class="sw" '
        f'style="background:{LABEL_COLORS.get(lb, "#ddd")}"></span>'
        f'{LABEL_DISPLAY.get(lb, lb)}</span>'
        for lb in [L for L in LABEL_DISPLAY if L in present])

    # One font family + one size everywhere (REPORT_PX).
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/><style>
@page {{ size: {width_in}in {height_in}in; margin: 0; }}
* {{ box-sizing: border-box; font-family: "{FONT}", "DejaVu Sans", sans-serif; }}
html, body {{ margin: 0; padding: 0; background: #fff; }}
body {{ color:#1c2330; width:{width_in}in; min-height:100%; padding:0.18in 0.2in;
  display:flex; flex-direction:column; font-size:{REPORT_PX}px; }}
.title {{ font-size:{REPORT_PX}px; color:#111827; margin:0 0 0.14em 0; }}
.subtitle {{ font-size:{REPORT_PX}px; color:#4b5563; margin:0 0 0.32em 0; }}
.inline-legend {{ display:flex; flex-wrap:wrap; gap:0.18em 0.5em; margin:0 0 0.4em 0;
  padding:0.14em 0.3em; border:0.55px solid #e5e7eb; border-radius:4px; background:#f9fafb; }}
.leg {{ display:inline-flex; align-items:center; font-size:{REPORT_PX}px; color:#374151; }}
.sw {{ display:inline-block; width:0.8em; height:0.66em; margin-right:0.28em;
  border:0.5px solid #9aa3b2; border-radius:2px; }}
.report {{ font-size:{REPORT_PX}px; line-height:1.8; color:#1c2330; white-space:normal;
  overflow-wrap:break-word; border:0.55px solid #e2e5ea; border-radius:4px;
  padding:0.32em 0.42em; background:#fcfcfd; }}
mark.seg {{ color:#131a24; border-radius:2px; }}
.sentinel {{ color:#b7bcc5; font-style:italic; }}
.foot {{ font-size:{REPORT_PX}px; color:#9aa3b2; margin:0.3em 0 0 0; }}
</style></head><body>
  <p class="title">(a) &nbsp; Annotated example report</p>
  <div class="inline-legend">{leg}</div>
  <div class="report">{''.join(pieces)}</div>
  <p class="foot">Hybrid-pipeline span highlights; colour = clinical label.
     n = 1 of 100 reports (illustrative example).</p>
</body></html>"""


def _find_chrome() -> str:
    for cand in ("google-chrome-stable", "google-chrome", "chromium",
                 "chromium-browser", "chrome"):
        if shutil.which(cand):
            return shutil.which(cand)
    raise SystemExit("No headless Chrome/Chromium binary found on PATH.")


def render_report_image(text_id: str) -> Image.Image:
    text, spans = load_report_spans(text_id)
    subtitle = (f"{text_id}  ·  {len(spans)} entities extracted by the "
                f"hybrid pipeline (highlighted by clinical label)")
    # 11pt in a 5.6in-wide chart -> ~96 px/in at 300 dpi -> report text ~15 px
    # so the report body reads the same size as the right-hand charts.
    width_in, height_in = 4.6, (2 * RIGHT_FIGSIZE_H + 0.25)
    html = _build_report_html(text, spans, subtitle, width_in, height_in)

    fd, tmp_path_str = tempfile.mkstemp(suffix=".html")
    tmp_html = Path(tmp_path_str)
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(html)
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.pdf"
            pdf = Path(td) / "report.pdf"
            png = Path(td) / "report.png"
            cmd = [_find_chrome(), "--headless=new", "--no-sandbox", "--disable-gpu",
                   "--hide-scrollbars",
                   f"--window-size={int(width_in * 96)},{int(height_in * 96)}",
                   f"--print-to-pdf={raw}", "--no-pdf-header-footer", str(tmp_html)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if not raw.exists():
                raise SystemExit(f"Chrome failed: {res.stderr[-800:]}")
            import re
            info = subprocess.run(["pdfinfo", str(raw)], capture_output=True, text=True)
            mo = re.search(r"Pages:\s+(\d+)", info.stdout)
            n_pages = int(mo.group(1)) if mo else 1
            keep = n_pages
            for p in range(n_pages, 1, -1):
                prefix = Path(td) / "pg"
                subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p), "-r", "30",
                                "-png", str(raw), str(prefix)], check=True,
                               capture_output=True)
                cand = next((Path(td) / f"pg-{s}.png"
                             for s in (str(p), f"0{p}", f"00{p}")
                             if (Path(td) / f"pg-{s}.png").exists()), None)
                blank = cand is None or (np.array(Image.open(cand).convert("L")) < 180).sum() == 0
                if blank:
                    keep = p - 1
            if keep < n_pages:
                subprocess.run(["gs", "-dBATCH", "-dNOPAUSE", "-dSAFER",
                                "-sDEVICE=pdfwrite", f"-dFirstPage=1",
                                f"-dLastPage={keep}", f"-sOutputFile={pdf}", str(raw)],
                               check=True, capture_output=True)
            else:
                pdf = raw
            subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "300",
                            str(pdf), str(png.with_suffix(""))],
                           check=True, capture_output=True)
            img = Image.open(png).convert("RGB")
    finally:
        tmp_html.unlink(missing_ok=True)
    return _trim_white(img, pad=24)


def _trim_white(im: Image.Image, pad: int = 12) -> Image.Image:
    a = np.array(im.convert("RGB"))
    mask = (a < 248).any(axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return im
    top, bottom = max(0, int(rows[0]) - pad), min(im.height, int(rows[-1]) + pad)
    left, right = max(0, int(cols[0]) - pad), min(im.width, int(cols[-1]) + pad)
    return im.crop((left, top, right, bottom))


# =============================================================================
# Panel B: ontology extraction per backend (box plots)
# =============================================================================

def render_strip_image(cov: dict) -> Image.Image:
    active = [p for p in PROVIDERS if any(cov[r][p] is not None for r in cov)]
    palette = {p: PROV_PALETTE[PROVIDERS.index(p)] for p in active}
    data = {p: [float(cov[r][p]) for r in cov if cov[r][p] is not None] for p in active}

    fig, ax = plt.subplots(figsize=(RIGHT_FIGSIZE_W, RIGHT_FIGSIZE_H))
    for i, p in enumerate(active):
        vals = np.array(data[p])
        xx = i + 1
        # one vertical line per report, evenly spread across the column; this
        # is the Cleveland-style "dot plot" that avoids the box's long whiskers.
        js = np.linspace(-0.16, 0.16, len(vals))
        ax.vlines(xx + js, 0, vals, color=palette[p], alpha=0.32, lw=1.0, zorder=2)
        med = float(np.median(vals))
        ax.plot([xx - 0.2, xx + 0.2], [med, med], color="black", lw=1.8, zorder=5,
                solid_capstyle="round")
        ax.plot([xx], [med], "o", mfc="black", mec="black", ms=4.5, zorder=6)

    pos = np.arange(1, len(active) + 1)
    ax.set_xticks(pos)
    ax.set_xticklabels([PROV_SHORT[p] for p in active], rotation=0, ha="center")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.25, len(active) + 0.75)
    ax.set_ylabel("Coverage per report")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("(b)  Ontology extraction by backend", loc="left", pad=8)
    # ax.text(0.5, -0.30, "each bar = one report (median = black dot)\nUMLS-R = UMLS (remote) · UMLS-L = UMLS (local) · "SN-D = SNOMED (DE) · SN-I = SNOMED (INTL)", transform=ax.transAxes, ha="center", va="top", color="#55606e", linespacing=1.5)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "strip.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        img = Image.open(out).convert("RGB")
    plt.close(fig)
    return img


# =============================================================================
# Panel C: per-label ontology linkage (bars)
# =============================================================================

def render_perlabel_image(per_label: list[tuple[str, int, int]]) -> Image.Image:
    df = [row for row in per_label if row[1] >= 20]
    df = sorted(df, key=lambda x: (x[2] / x[1] if x[1] else 0.0))  # worst -> best
    labels = [LABEL_DISPLAY.get(l, l) for l, _, _ in df]
    n = [x[1] for x in df]
    res = [x[2] for x in df]
    pct = [100.0 * r / t if t else 0.0 for r, t in zip(res, n)]

    fig, ax = plt.subplots(figsize=(RIGHT_FIGSIZE_W, RIGHT_FIGSIZE_H))
    ys = np.arange(len(df))
    bars = ax.barh(ys, pct, color=[LABEL_COLORS.get(l, "#DDDDDD") for l, _, _ in df],
                   edgecolor="white", lw=0.6, zorder=3)
    for yi, bar, r, t in zip(ys, bars, res, n):
        ax.text(pct[yi] + 1.6, yi, f"{pct[yi]:.0f}%  ({r}/{t})",
                va="center", ha="left")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 122)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlabel("Spans resolved by >= 1 backend (%)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("(c)  Per-label ontology linkage", loc="left", pad=8)

    fig.tight_layout()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "lbl.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        img = Image.open(out).convert("RGB")
    plt.close(fig)
    return img


# =============================================================================
# Compose: report (left) + stack of (B) over (C)
# =============================================================================

def _fit_height(im: Image.Image, h: int) -> Image.Image:
    if im.height == h:
        return im
    w = max(1, int(im.width * (h / im.height)))
    return im.resize((w, h), Image.LANCZOS)


def _canvas_of(im: Image.Image, w: int, h: int) -> Image.Image:
    out = Image.new("RGB", (w, h), "white")
    out.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return out


def compose(a: Image.Image, b: Image.Image, c: Image.Image,
            out_png: Path, out_pdf: Path) -> None:
    # left (report) spans the full content height; right panels stack.
    content_h = a.height
    right_h = (content_h - GAP) // 2
    left = a
    box = _fit_height(b, right_h)
    lbl = _fit_height(c, right_h)
    right_w = max(box.width, lbl.width)
    boxc = _canvas_of(box, right_w, right_h)
    lblc = _canvas_of(lbl, right_w, right_h)

    right_x = GAP + left.width + GAP
    canvas_w = right_x + right_w + GAP
    canvas_h = content_h + 2 * GAP
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(left, (GAP, GAP))
    canvas.paste(boxc, (right_x, GAP))
    canvas.paste(lblc, (right_x, GAP + right_h + GAP))

    canvas.save(out_png, dpi=(DPI, DPI))
    try:
        canvas.save(out_pdf, "PDF", resolution=DPI)
    except Exception as e:
        out_pdf.unlink(missing_ok=True)
        print(f"WARN: could not write figure PDF ({e}) -- PNG only.")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-id", default=DEFAULT_TEXT_ID)
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args()

    text_id = args.text_id
    n_spans = len(load_report_spans(text_id)[1])
    cov = load_coverage()
    per_label = load_per_label()
    n_reports = len(cov)
    n_total = sum(n for _, n, _ in per_label)
    n_res = sum(r for _, _, r in per_label)
    print(f"Report   : {text_id}  ({n_spans} spans)")
    print(f"Coverage : {n_reports} reports, resolved by >=1 backend "
          f"{n_res}/{n_total} ({100 * n_res / n_total:.1f}%)")

    a = render_report_image(text_id)
    b = render_strip_image(cov)
    c = render_perlabel_image(per_label)

    out_png = HERE / "ontology_panels.png"
    out_pdf = HERE / "ontology_panels.pdf"
    compose(a, b, c, out_png, out_pdf)
    print(f"Wrote PNG : {out_png}")
    print(f"Wrote PDF : {out_pdf}")
    if args.copy_to is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for f in (out_png, out_pdf):
            shutil.copy(f, args.copy_to / f.name)
        print(f"Copied to {args.copy_to.resolve()}")


if __name__ == "__main__":
    main()

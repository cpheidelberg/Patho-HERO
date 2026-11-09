#!/usr/bin/env python3
"""
Generate a fully synthetic report test set for the Patho-HERO experiments,
without any real patient data.

Why this exists
---------------
The original pathology-report test set cannot be published (it contains
patient text, even after pseudonymization). This factory script therefore
produces a small, clearly-fabricated stand-in test set in *exactly the same
shape* the original had, so every experiment script in ``experiments/`` can
be re-run end-to-end on a machine that has no access to the clinical
PostgreSQL archive.

What it contains
----------------
``experiments/berichte.xlsx`` — one row per synthetic report, with the
columns the experiment scripts read (``report_id`` and a ``text`` column the
pipeline consumes) plus the metadata columns the JSONL records cite
(``dateiname``, ``anonym_case_id``, ``berichtstyp``, ``jahr``).

The report texts are templated German pathology fragments built from the
vocabulary the deterministic extractors already know about (the ``config/*.txt``
gazetteer lists, ICD-10-GM / ICD-O-3 codes, TNM formulae, sizes, negation
and uncertainty phrases). They exercise the pipeline without describing any
real case or patient.

Usage (from the repo root):
    python3 experiments/make_synthetic_testset.py            # 50 reports, seed 42
    python3 experiments/make_synthetic_testset.py -n 20 --seed 7 -o my_sample.xlsx
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Deterministic vocabulary pools (template-only, no real cases)
# ---------------------------------------------------------------------------

SAMPLE_TYPES = [
    "Cholezystektomiepräparat",
    "Leberbiopsie",
    "Stanzzylinder",
    "Sentinel-Lymphknoten",
    "Hysterektomiepräparat",
    "Nierenbiopsie",
    "Zervixabradat",
    "Biopsat",
]

SIZES = ["0.9 cm", "1.4 cm", "2.2 cm", "3.5 cm", "4.1 cm", "0.4 cm", "5.0 cm",
         "1.2 x 1.7 x 0.9 cm"]

STAININGS = ["H&E", "PAS", "Giemsa", "Feulgen", "Gomori"]

FINDINGS = [
    "Akute unspezifische Entzündung",
    "Reaktive Lymphozyteninfiltration",
    "Nekrosefoci",
    "Hämorrhagisches Areal",
    "Granulationsgewebe",
    "Keine Malignität nachweisbar",
    "Fokale atypische Zellcluster",
]

ICD10G = ["C61", "C50.9", "C18", "B15", "C57", "B08.9"]
ICDO3 = ["M 8140/3", "M 8010/3", "M 9541/3", "M 8241/3"]
TNM = ["pT2b pN0 Mx", "pT3 pN1a M0", "rT1a pN0 M0"]

DIAGNOSES = [
    "Adenokarzinom, Gleason-Score 7 (4+3)",
    "Follikuläres Non-Hodgkin-Lymphom, B-Zell-Typ",
    "Diffus großzelliges B-Zell-Lymphom",
    "Invasive duktales Karzinom",
    "Benigne Prostatagranulomatose",
    "Reaktive Lymphadenitis",
]

IHC = ["AE1/AE3+", "CD20+/CD3-", "CK7+/CK20-", "PAX8+", "CK20+/CDX2+"]


def _rand(rng: random.Random, seq) -> str:
    return seq[rng.randrange(len(seq))]


def _synthetic_report(rng: random.Random) -> str:
    sizes = ", ".join(_rand(rng, SIZES) for _ in range(rng.choice([1, 2])))
    finding = _rand(rng, FINDINGS)
    negation = "Kein Anhalt für Malignität." if rng.random() < 0.4 else \
               "Mögliches Malignitätssignal, differentialdiagnostisch zu klären."
    return (
        "Anamnese: klinisch unauffällig, Zufallsbefund. "
        "Makroskopie: "
        f"{_rand(rng, SAMPLE_TYPES)}, {sizes}. "
        f"Färbung: {_rand(rng, STAININGS)}-Färbung. "
        f"ICD-10-GM Kodierung {_rand(rng, ICD10G)}; "
        f"ICD-O-3 { _rand(rng, ICDO3) }; "
        f"{ _rand(rng, TNM) }-Klassifikation. "
        f"Mikroskopie: {_rand(rng, FINDINGS)}. {finding}. "
        f"Diagnose: {_rand(rng, DIAGNOSES)}. "
        f"IHC: {_rand(rng, IHC)}. "
        f"{negation}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic report test set factory (no patient data)")
    parser.add_argument("-n", "--num", type=int, default=50,
                        help="Anzahl synthetische Berichte (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Zufalls-Seed (default: 42)")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("experiments/berichte.xlsx"),
                        help="Ausgabe-Excel-Datei (default: experiments/berichte.xlsx)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    rows = []
    for i in range(args.num):
        report_id = 100 + i
        dateiname = f"synthetic-case-{report_id}_Hauptbefund.txt"
        rows.append({
            "report_id": report_id,
            "text": _synthetic_report(rng),          # <-- report text (synthetic)
            "dateiname": dateiname,
            "anonym_case_id": f"case-{report_id}",
            "berichtstyp": "Hauptbefund",
            "jahr": 2026,
            "is_synthetic": True,
        })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.output, index=False)
    print(f"Synthetic test set written to {args.output} ({len(df)} reports)")


if __name__ == "__main__":
    main()

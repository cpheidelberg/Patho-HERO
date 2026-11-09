#!/usr/bin/env python3
"""
Create a local SQLite sample database from the synthetic test set.

The full production pipeline stores reports and concepts in PostgreSQL. For a
self-contained, runnable sample (no clinical database, no patient data) we
materialize the same small relational schema in a SQLite file, so the
``experiments/`` scripts can be executed end-to-end and their outputs
inspected locally.

This script:
  1. Reads the synthetic test set ``experiments/berichte.xlsx`` (run
     ``make_synthetic_testset.py`` first if it is missing).
  2. Creates the tables ``case_report`` and ``report_concept`` in SQLite
     (a minimal, dependency-free subset of the production schema, sufficient
     for extraction and a round-trip of the extracted spans).
  3. Seeds ``case_report`` with the synthetic report rows.

Usage (from the repo root):
    python3 experiments/make_sqlite_sample_db.py            # experiments/sample_reports.sqlite
    python3 experiments/make_sqlite_sample_db.py -d my.sqlite -i experiments/berichte.xlsx
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_sample_db(xlsx_path: Path, db_path: Path) -> None:
    df = pd.read_excel(xlsx_path)
    required = {"report_id", "text", "dateiname", "anonym_case_id",
                "berichtstyp", "jahr"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{xlsx_path} lacks required columns: {sorted(missing)}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE case_report (
                report_id       INTEGER PRIMARY KEY,
                anonym_case_id  TEXT,
                dateiname       TEXT,
                berichtstyp     TEXT,
                jahr            INTEGER,
                raw_text        TEXT,
                is_synthetic    INTEGER,
                created_at      TEXT
            );

            -- Target table the pipeline fills with extracted spans
            -- (concept_id is NULL until ontology linking has run).
            CREATE TABLE report_concept (
                report_id     INTEGER NOT NULL,
                concept_id    INTEGER,
                source_field  TEXT,
                surface_text  TEXT,
                raw_text      TEXT,
                search_term   TEXT,
                ontology      TEXT,
                start_offset  INTEGER,
                end_offset    INTEGER,
                assertion     TEXT,
                certainty     TEXT,
                temporality   TEXT,
                laterality    TEXT,
                confidence    REAL,
                PRIMARY KEY (report_id, start_offset, surface_text),
                FOREIGN KEY (report_id) REFERENCES case_report(report_id)
            );
            """
        )

        cur = conn.execute("SELECT COUNT(*) FROM case_report")
        if cur.fetchone()[0] > 0:
            raise SystemExit(f"{db_path} already contains case_report rows; "
                             f"remove it first to re-seed.")

        now = _now_iso()
        for row in df.itertuples(index=False):
            conn.execute(
                """
                INSERT INTO case_report
                    (report_id, anonym_case_id, dateiname, berichtstyp,
                     jahr, raw_text, is_synthetic, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(row.report_id),
                 row.anonym_case_id,
                 row.dateiname,
                 row.berichtstyp,
                 int(row.jahr) if row.jahr is not None else None,
                 row.text,
                 int(bool(row.is_synthetic)),
                 now),
            )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM case_report").fetchone()[0]
    finally:
        conn.close()

    print(f"SQLite sample DB written to {db_path} ({n} synthetic report rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a SQLite sample DB from the synthetic test set")
    parser.add_argument("-i", "--input", type=Path,
                        default=Path("experiments/berichte.xlsx"),
                        help="Synthetic test set XLSX (default: experiments/berichte.xlsx)")
    parser.add_argument("-d", "--db", type=Path,
                        default=Path("experiments/sample_reports.sqlite"),
                        help="Output SQLite file (default: experiments/sample_reports.sqlite)")
    args = parser.parse_args()
    make_sample_db(args.input, args.db)


if __name__ == "__main__":
    main()

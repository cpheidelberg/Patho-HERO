#!/usr/bin/env python3
"""
MedCAT-NER-Experiment: Extrahiert Entitäten mit MedCAT (SNOMED CT)
aus den im Testset berichte.xlsx gelisteten Berichten.

Die Berichtstexte werden aus der PostgreSQL-Datenbank geladen
(wie in experiments/hybrid_ner.py).

Unterstützt mehrere Extraktionsläufe (--runs), die alle in einer
einzigen JSONL-Datei gespeichert werden, mit run_id zur Unterscheidung.

Läuft vom Projektroot, z. B.:
    python experiments/medcat_ner.py --runs 5
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import pandas as pd
from rainbow_tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.config import Settings
from src.database import Repository, CaseReport
from sample_store import SampleStore

import sqlite3
from medcat.cat import CAT


def process_single_report(cat, report_id, text, meta, run_id):
    """Verarbeitet einen einzelnen Bericht mit MedCAT."""
    entities = cat.get_entities(text)

    # Konvertiere MedCAT-Entitäten in das Standard-Span-Format
    spans = []
    for ent_id, ent in entities.items():
        # MedCAT gibt start/end als Character-Offsets zurück
        spans.append({
            "start": ent.get("start", 0),
            "end": ent.get("end", 0),
            "text": ent.get("text", text[ent.get("start", 0):ent.get("end", 0)]),
            "label": "medcat_entity",  # MedCAT hat keine festen Labels wie die Hybrid-Pipeline
            "terminology": "snomed_ct",
            "concept_id": ent.get("cui"),  # SNOMED CT CUI
            "search_term": ent.get("preferred_name") or ent.get("name"),
            "meta": {
                "source": "medcat",
                "assertion": "present",  # MedCAT macht keine Negationserkennung
                "confidence": ent.get("confidence", ent.get("acc", 0.0)),
                "medcat_details": {
                    "pretty_name": ent.get("pretty_name"),
                    "types": ent.get("types", []),
                    "context_similarity": ent.get("context_similarity"),
                    "id": ent_id,
                }
            },
        })

    return {
        "text_id": Path(meta.dateiname).stem,
        "text": text,
        "spans": spans,
        "meta": {
            "source": "patientdb",
            "table": "case_report",
            "report_id": report_id,
            "anonym_case_id": meta.anonym_case_id,
            "dateiname": meta.dateiname,
            "berichtstyp": meta.berichtstyp,
            "jahr": meta.jahr,
            # Run-spezifische Metadaten
            "run_id": run_id,
            "extractor": "medcat",
            "model_pack": str(args.model),
            "run_timestamp": datetime.now().isoformat(),
        },
    }


def run_extraction(cat, reports, report_ids, metadaten, run_id, workers, file_lock, output_file):
    """Führt einen einzelnen Extraktionslauf durch und schreibt in die JSONL-Datei."""
    results = []
    failed_reports = []

    def process_with_progress(report_id):
        if report_id not in reports:
            return None, f"Report {report_id} nicht in der Datenbank gefunden"
        text = reports[report_id]
        meta = metadaten[report_id]
        try:
            result = process_single_report(cat, report_id, text, meta, run_id)
            return result, None
        except Exception as e:
            return None, f"Fehler bei Report {report_id}: {str(e)}"

    # Parallele Verarbeitung mit ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_with_progress, rid): rid for rid in report_ids}

        with tqdm(total=len(report_ids), desc=f"Run {run_id}", position=run_id-1) as pbar:
            for future in as_completed(futures):
                result, error = future.result()
                if error:
                    print(f"WARN (Run {run_id}): {error}")
                    failed_reports.append(error)
                elif result:
                    results.append(result)
                pbar.update(1)

    # Ergebnisse in JSONL schreiben (thread-safe)
    with file_lock:
        for result in results:
            output_file.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
        output_file.flush()

    return len(results), failed_reports


def main():
    global args  # Für process_single_report verfügbar machen

    parser = argparse.ArgumentParser(
        description="MedCAT-NER (SNOMED CT) über das Testset berichte.xlsx "
                    "(Berichtstexte aus der Datenbank) als JSONL exportieren. "
                    "Unterstützt mehrere Läufe."
    )
    parser.add_argument("-i", "--input", type=Path, default=Path("experiments/berichte.xlsx"),
                        help="Testset-Liste (default: experiments/berichte.xlsx)")
    parser.add_argument("-o", "--output", type=Path, default=Path("experiments/medcat_ner.jsonl"),
                        help="Ausgabe-JSONL (default: experiments/medcat_ner.jsonl)")
    parser.add_argument("-m", "--model", type=Path, default=Path("experiments/v1_Snomed2025_MIMIC_IV.zip"),
                        help="MedCAT model_pack (default: experiments/v1_Snomed2025_MIMIC_IV.zip)")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Nur die ersten N Berichte verarbeiten (Debugging)")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="Anzahl der Extraktionsläufe (default: 1)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Anzahl paralleler Worker pro Run (default: 4)")

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")

    # MedCAT model_pack laden
    # https://docs.cogstack.org/projects/nlp/en/latest/
    print(f"Lade MedCAT model_pack: {args.model}")
    cat = CAT.load_model_pack(str(args.model))
    print("Model geladen.")

    # Testset-Liste laden
    df = pd.read_excel(args.input)
    if args.limit:
        df = df.head(args.limit)
    report_ids = df["report_id"].astype(int).tolist()
    metadaten = {int(row.report_id): row for row in df.itertuples(index=False)}
    print(f"\n{len(report_ids)} Berichte aus {args.input} geladen")

    # Berichte laden: entweder aus SQLite-Sample (synthetische Texte, selbstlaufend)
    # oder aus PostgreSQL (klinische Datenbank).
    sqlite_path = (os.getenv("PATHOHERO_SQLITE") or "").strip()
    if sqlite_path and os.path.exists(sqlite_path):
        conn = sqlite3.connect(sqlite_path)
        try:
            rows = conn.execute(
                "SELECT report_id, raw_text FROM case_report ORDER BY report_id"
            ).fetchall()
        finally:
            conn.close()
        reports = {int(rid) for rid in report_ids}.difference
        reports_db = {rid: text for rid, text in rows}
        reports = [types.SimpleNamespace(report_id=rid, raw_text=text)
                   for rid, text in reports_db.items() if rid in set(report_ids)]
        print(f"SQLite-Sample: {len(reports)} Berichte mit Text geladen aus {sqlite_path}")
    else:
        # Postgres-Fallback (klinische Datenbank)
        repo = Repository(settings.database_url)
        with repo.session_scope() as session:
            rows = (session.query(CaseReport.report_id, CaseReport.raw_text)
                    .filter(CaseReport.report_id.in_(report_ids))
                    .filter(CaseReport.raw_text.isnot(None))
                    .all())
        reports = {report_id: raw_text for report_id, raw_text in rows}
        print(f"{len(reports)} Berichte mit Text in der Datenbank gefunden")

    if not reports:
        print("Keine Berichte zu verarbeiten.")
        return

    # Ausgabe-Datei vorbereiten (im "w"-Modus, wird bei mehreren Runs überschrieben)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # File-Lock für thread-sicheres Schreiben
    file_lock = threading.Lock()

    # Gesamtstatistik
    total_start = datetime.now()
    all_stats = []

    print(f"\n{'='*60}")
    print(f"Starte {args.runs} Extraktionslauf/läufe")
    print(f"Worker pro Run: {args.workers}")
    print(f"Ausgabe: {args.output}")
    print(f"{'='*60}\n")

    with open(args.output, 'w', encoding='utf-8') as output_file:
        # Optional: Header-Zeile mit Experiment-Metadaten
        header = {
            "text_id": "__experiment_header__",
            "text": "",
            "spans": [],
            "meta": {
                "experiment": "medcat_ner",
                "model_pack": str(args.model),
                "num_runs": args.runs,
                "num_reports": len(report_ids),
                "workers_per_run": args.workers,
                "started_at": total_start.isoformat(),
                "input_file": str(args.input),
            }
        }
        output_file.write(json.dumps(header, ensure_ascii=False, default=str) + '\n')

        # Mehrere Runs durchführen
        for run_id in range(1, args.runs + 1):
            run_start = datetime.now()
            print(f"\n--- Run {run_id}/{args.runs} ---")

            n_results, failed = run_extraction(
                cat=cat,
                reports=reports,
                report_ids=report_ids,
                metadaten=metadaten,
                run_id=run_id,
                workers=args.workers,
                file_lock=file_lock,
                output_file=output_file,
            )

            run_end = datetime.now()
            elapsed = (run_end - run_start).total_seconds()

            stats = {
                "run_id": run_id,
                "n_results": n_results,
                "n_failed": len(failed),
                "elapsed_seconds": elapsed,
                "reports_per_second": n_results / elapsed if elapsed > 0 else 0,
            }
            all_stats.append(stats)

            print(f"Run {run_id} abgeschlossen: {n_results} Berichte, "
                  f"{len(failed)} fehlgeschlagen, {elapsed:.1f}s "
                  f"({stats['reports_per_second']:.2f} Berichte/s)")

    total_end = datetime.now()
    total_elapsed = (total_end - total_start).total_seconds()

    # Zusammenfassung
    print(f"\n{'='*60}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    print(f"Gesamtläufe: {args.runs}")
    print(f"Gesamtberichte pro Run: {len(report_ids)}")
    print(f"Gesamte exportierte Datensätze: {sum(s['n_results'] for s in all_stats)}")
    print(f"Gesamt fehlgeschlagen: {sum(s['n_failed'] for s in all_stats)}")
    print(f"Gesamtdauer: {total_elapsed:.1f}s")
    print(f"Durchschnitt pro Run: {total_elapsed / args.runs:.1f}s")
    print(f"\nJSONL-Datei: {args.output.resolve()}")
    print(f"Dateigröße: {args.output.stat().st_size / 1024 / 1024:.2f} MB")

    # Detaillierte Statistik als separate JSON-Datei speichern
    stats_file = args.output.with_suffix('.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "medcat_ner",
            "model_pack": str(args.model),
            "total_runs": args.runs,
            "total_reports_per_run": len(report_ids),
            "total_elapsed_seconds": total_elapsed,
            "started_at": total_start.isoformat(),
            "ended_at": total_end.isoformat(),
            "runs": all_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f"Statistik-Datei: {stats_file.resolve()}")


if __name__ == "__main__":
    main()
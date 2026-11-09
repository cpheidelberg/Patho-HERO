#!/usr/bin/env python3
"""
OpenMed-NER-Experiment: Extrahiert MEDIZINISCHE Entitäten (Diagnosen,
Morphologie, Anatomie/Lokalisation) mit OpenMed aus den im Testset
berichte.xlsx gelisteten Berichten (Berichtstexte aus der PostgreSQL-DB).

Die Berichte sind ohnehin anonymisiert, daher wird bewusst KEIN PII-Modell
verwendet, sondern die medizinischen NER-Modelle von OpenMed.

Standardmodellkonfiguration:
    disease_detection_superclinical  (Diagnosen, pathologische Befunde)
    anatomy_detection_bigmed_560m    (Anatomie/Lokalisation, Gewebe)

Unterstützt mehrere Extraktionsläufe (--runs), die alle in einer einzigen
JSONL-Datei gespeichert werden, mit run_id zur Unterscheidung.

Läuft vom Projektroot, z. B.:
    python experiments/openmed_ner.py --runs 5
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

# OpenMed nutzt intern das XET/CAS-Protokoll, das hier 403-Fehler wirft.
# Klassisches HTTP-Protokoll erzwingen, damit (Re-)Downloads stabil laufen.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.config import Settings
from src.database import Repository, CaseReport
from sample_store import SampleStore
import sqlite3

from openmed import analyze_text

# Medizinische OpenMed-Modelle (Kein PII): Diagnose + Anatomie/Morphologie.
DEFAULT_MODELS = [
    "disease_detection_superclinical",
    "anatomy_detection_bigmed_560m",
]


def zeige_modellinfos() -> None:
    """Gibt einen Überblick über die verfügbaren PII-Modelle."""

    lang_names = {"de": "German"}

    for lang in sorted(lang_names):
        models = get_pii_models_by_language(lang)
        default = get_default_pii_model(lang)
        print(f"{lang_names[lang]} ({lang}):")
        print(f"  Models available: {len(models)}")
        print(f"  Default model:    {default}")
        print(f"  Model names:      {', '.join(models)}")

    total = len([k for k in OPENMED_MODELS if k.startswith("pii_")])
    print(f"Total PII models registered: {total}")


def entity_zu_dict(entity) -> dict:
    """Konvertiert eine OpenMed-Entität (EntityPrediction) in ein dict."""
    if isinstance(entity, dict):
        return entity
    if hasattr(entity, "to_dict"):
        return entity.to_dict()
    import dataclasses
    if dataclasses.is_dataclass(entity):
        return dataclasses.asdict(entity)
    if hasattr(entity, "_asdict"):
        return dict(entity._asdict())
    return vars(entity)


def entity_zu_span(e_dict: dict, text: str, model: str, terminology: str) -> dict:
    """Mappt ein OpenMed-Entity-dict auf unser einheitliches Span-Format."""
    start = e_dict.get("start", 0) or 0
    end = e_dict.get("end", 0) or 0
    return {
        "start": start,
        "end": end,
        "text": e_dict.get("text", text[start:end]),
        "label": e_dict.get("label", "medical_entity"),
        "terminology": f"openmed_{terminology}",
        "concept_id": e_dict.get("concept_id"),
        "search_term": e_dict.get("text"),
        "meta": {
            "source": f"openmed_{terminology}",
            "assertion": "present",
            "confidence": e_dict.get("confidence", e_dict.get("score", 0.0)),
            "model": model,
        },
    }


def _dedupe(spans: list) -> list:
    """Entfernt identische (start, end, label)-Doppelungen über Modelle hinweg."""
    seen, out = set(), []
    for s in spans:
        key = (s["start"], s["end"], s["label"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def process_single_report(report_id: int, text: str, meta, run_id: int,
                          models: list) -> dict:
    """Verarbeitet einen Bericht mit den konfigurierten OpenMed-Modellen."""
    spans = []
    for model in models:
        result = analyze_text(text, model_name=model,
                              output_format="dict",
                              confidence_threshold=0.0)
        termin = "disease" if "disease" in model else "anatomy"
        if "pathology" in model:
            termin = "pathology"
        for e in (result.entities if hasattr(result, "entities") else []):
            spans.append(entity_zu_span(entity_zu_dict(e), text, model, termin))
    spans = _dedupe(spans)

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
            "run_id": run_id,
            "extractor": "openmed_ner",
            "models": models,
            "run_timestamp": datetime.now().isoformat(),
        },
    }


def run_extraction(reports: dict, report_ids: list, metadaten: dict, run_id: int,
                   workers: int, file_lock, output_file, models: list) -> tuple:
    """Führt einen einzelnen Extraktionslauf durch und schreibt in die JSONL-Datei."""
    results = []
    failed_reports = []

    def process_with_progress(report_id):
        if report_id not in reports:
            return None, f"Report {report_id} nicht in der Datenbank gefunden"
        text = reports[report_id]
        meta = metadaten[report_id]
        try:
            result = process_single_report(report_id, text, meta, run_id, models)
            return result, None
        except Exception as e:  # noqa: BLE001
            return None, f"Fehler bei Report {report_id}: {str(e)}"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_with_progress, rid): rid for rid in report_ids}
        with tqdm(total=len(report_ids), desc=f"Run {run_id}", position=run_id - 1) as pbar:
            for future in as_completed(futures):
                result, error = future.result()
                if error:
                    print(f"WARN (Run {run_id}): {error}")
                    failed_reports.append(error)
                elif result:
                    results.append(result)
                pbar.update(1)

    with file_lock:
        for result in results:
            output_file.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
        output_file.flush()

    return len(results), failed_reports


def main():
    parser = argparse.ArgumentParser(
        description="OpenMed-Medizin-NER (Diagnose/Morphologie/Anatomie) über "
                    "das Testset berichte.xlsx als JSONL exportieren. "
                    "Unterstützt mehrere Läufe."
    )
    parser.add_argument("-i", "--input", type=Path, default=Path("experiments/berichte.xlsx"))
    parser.add_argument("-o", "--output", type=Path, default=Path("experiments/openmed_ner.jsonl"))
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Nur die ersten N Berichte verarbeiten (Debugging)")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="Anzahl der Extraktionsläufe (default: 1)")
    parser.add_argument("-w", "--workers", type=int, default=2,
                        help="Anzahl paralleler Worker pro Run (default: 2, CPU)")
    parser.add_argument("-m", "--models", default=",".join(DEFAULT_MODELS),
                        help="Komma-separierte Modellnamen (default: "
                             + ",".join(DEFAULT_MODELS) + ")")

    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"OpenMed-Medizin-Modelle: {models}")
    try:
        modellinfos()
    except Exception as e:  # noqa: BLE001
        print(f"  (Modell-Informationen nicht verfügbar: {e})")

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
        reports_db = {rid: text for rid, text in rows}
        reports = {rid: text for rid in report_ids if rid in reports_db}
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_lock = threading.Lock()
    total_start = datetime.now()
    all_stats = []

    # Warmup: lädt/verifiziert die Modelle einmalig, bevor die Runs starten.
    print("\nWarmup: lade/verifiziere OpenMed-Modelle (kann Download auslösen)...")
    sample_text = reports[report_ids[0]][:400]
    for m in models:
        try:
            analyze_text(sample_text, model_name=m)
            print(f"  OK: {m}")
        except Exception as e:  # noqa: BLE001
            print(f"  FEHLER bei {m}: {e}")
    print("Warmup abgeschlossen.")

    print(f"\n{'=' * 60}\nStarte {args.runs} Extraktionslauf/läufe\n"
          f"Worker pro Run: {args.workers}\nAusgabe: {args.output}\n{'=' * 60}\n")

    with open(args.output, 'w', encoding='utf-8') as output_file:
        header = {
            "text_id": "__experiment_header__",
            "text": "",
            "spans": [],
            "meta": {
                "experiment": "openmed_ner",
                "models": models,
                "num_runs": args.runs,
                "num_reports": len(report_ids),
                "workers_per_run": args.workers,
                "started_at": total_start.isoformat(),
                "input_file": str(args.input),
            }
        }
        output_file.write(json.dumps(header, ensure_ascii=False) + '\n')

        for run_id in range(1, args.runs + 1):
            run_start = datetime.now()
            print(f"\n--- Run {run_id}/{args.runs} ---")
            n_results, failed = run_extraction(
                reports=reports,
                report_ids=report_ids,
                metadaten=metadaten,
                run_id=run_id,
                workers=args.workers,
                file_lock=file_lock,
                output_file=output_file,
                models=models,
            )
            elapsed = (datetime.now() - run_start).total_seconds()
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

    total_elapsed = (datetime.now() - total_start).total_seconds()
    print(f"\n{'=' * 60}\nZUSAMMENFASSUNG\n{'=' * 60}")
    print(f"Gesamtläufe: {args.runs}")
    print(f"Berichte pro Run: {len(report_ids)}")
    print(f"Exportierte Datensätze: {sum(s['n_results'] for s in all_stats)}")
    print(f"Fehlgeschlagen: {sum(s['n_failed'] for s in all_stats)}")
    print(f"Gesamtdauer: {total_elapsed:.1f}s")
    print(f"\nJSONL-Datei: {args.output.resolve()}")

    stats_file = args.output.with_suffix('.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "openmed_ner",
            "models": models,
            "total_runs": args.runs,
            "total_reports_per_run": len(report_ids),
            "total_elapsed_seconds": total_elapsed,
            "started_at": total_start.isoformat(),
            "ended_at": datetime.now().isoformat(),
            "runs": all_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f"Statistik-Datei: {stats_file.resolve()}")


if __name__ == "__main__":
    main()

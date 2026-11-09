#!/usr/bin/env python3
"""
Hybrid-NER-Experiment: Führt die src/-Pipeline (Gazetteer, Regex, LLM-Fallback,
Negation, Ontologie-Mapping) über die im Testset berichte.xlsx gelisteten
Berichte aus und speichert die Extraktion als JSONL (analog openmed_ner).

Unterstützt mehrere Extraktionsläufe (runs), die in einer einzigen JSONL-Datei
gespeichert werden, wobei jeder Datensatz mit einer run_id versehen wird.

Läuft vom Projektroot, z. B.:
    python experiments/hybrid_ner.py --runs 5
"""

import argparse
import logging
import os
import sys
import threading
import json
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import pandas as pd
from rainbow_tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.config import Settings, load_pipeline_config
from src.database import Repository, CaseReport
from src.extractors.llm_extractor import LLMExtractor
from src.extractors.hf_extractor import HuggingFaceExtractor
from src.terminology.snowstorm import SnowstormClient
from src.terminology.mapper import OntologyMapper
from src.pipeline import NERPipeline
from src.logging_config import setup_logging
from sample_store import SampleStore
import sqlite3

logger = logging.getLogger(__name__)

import urllib.request
import json as _json


def _load_reports(settings, config: dict, repo, report_ids: list,
                  metadaten: dict, input_path: Path):
    """Lädt Berichte mit Text.

    Priorität:
      1. SQLite-Sample (``PATHOHERO_SQLITE`` gesetzt) – synthetische Texte
         aus ``berichte.xlsx``, ohne Zugriff auf die klinische Postgres-DB.
      2. PostgreSQL-Datenbank (Standard) – Berichte aus ``case_report``.

    Gibt eine Liste von Row-Objekten zurück (Attribut-Zugriff auf
    ``report_id`` und ``raw_text``), wie sie ``run_single_extraction`` erwartet.
    """
    sqlite_path = (os.getenv("PATHOHERO_SQLITE") or "").strip()
    if sqlite_path and os.path.exists(sqlite_path):
        import sqlite3 as _sqlite
        conn = _sqlite.connect(sqlite_path)
        try:
            cur = conn.execute(
                "SELECT report_id, raw_text FROM case_report "
                "ORDER BY report_id")
            db_rows = {int(r[0]): r[1] for r in cur.fetchall()}
        finally:
            conn.close()
        reports = []
        missing = []
        for rid in report_ids:
            text = db_rows.get(rid)
            if text is None:
                text = str(metadaten[rid].text)  # Fallback: Text aus der XLSX
                missing.append(rid)
            reports.append(types.SimpleNamespace(report_id=rid, raw_text=text))
        print(f"SQLite-Sample: {len(reports)} Berichte geladen "
              f"({len(missing)} aus XLSX, {len(reports) - len(missing)} aus DB)")
        return reports

    # Postgres-Fallback (klinische Datenbank)
    with repo.session_scope() as session:
        rows = (session.query(CaseReport)
                .filter(CaseReport.report_id.in_(report_ids))
                .filter(CaseReport.raw_text.isnot(None))
                .all())
    return rows


def baue_ontology_clients_auf(settings: Settings, config: dict,
                              repo: Repository) -> tuple[OntologyMapper, dict]:
    """Initialisiert Ontologie-Clients und Mapper wie in src/main.py."""

    default_provider = config["pipeline"].get("ontology_provider")
    provider_per_label = config["pipeline"].get("ontology_provider_per_label") or {}

    # Optional: Snowstorm Client (nur wenn explizit konfiguriert)
    snow = None
    if default_provider == "snowstorm" or any(
            p == "snowstorm" for p in provider_per_label.values()):
        snow = SnowstormClient(settings.snowstorm_base_url,
                               settings.snowstorm_branch,
                               settings.snowstorm_language,
                               os.getenv("SNOWSTORM_ACCEPT_LANGUAGE", "de,en;q=0.5"))

    # Optional: UMLS-Client. Nur wenn explizit konfiguriert UND ein Client
    # verfügbar ist (lokale UMLS-DB oder REST-API). Für das Synthetic-Sample
    # wird Ontology-Mapping typischerweise deaktiviert (use_ontology: false).
    umls = None
    if default_provider == "umls" or any(p == "umls" for p in provider_per_label.values()):
        if repo is not None and hasattr(repo, "Session"):
            umls = _make_umls_client(settings, repo)
        if umls is None:
            logger.warning("UMLS als Provider konfiguriert, aber weder lokale "
                           "UMLS-DB noch UMLS_API_KEY verfügbar. UMLS-Mapping deaktiviert.")
        else:
            logger.info("UMLS-Provider aktiv: %s", type(umls).__name__)

    return snow, umls


def _make_umls_client(settings, repo):
    """Baut einen UMLS-Client (lokal oder REST) aus den Settings, oder None."""
    local_url = settings.umls_local_db_url or settings.database_url
    try:
        from src.terminology.local_umls import LocalUMLSClient
        local = LocalUMLSClient(
            settings.umls_api_key,
            version=settings.umls_version,
            repository=repo if local_url.rstrip("/") == settings.database_url.rstrip("/") else None,
            database_url=local_url,
            schema=settings.umls_local_schema or None,
            language=settings.umls_local_language,
        )
        return local
    except Exception as exc:
        logger.debug("Lokaler UMLS-Client nicht verfügbar: %s", exc)
    if settings.umls_api_key:
        try:
            from src.terminology.umls import UMLSClient
            return UMLSClient(settings.umls_api_key, version=settings.umls_version)
        except Exception as exc:
            logger.debug("UMLS REST-Client nicht verfügbar: %s", exc)
    return None


def run_single_extraction(
    run_id: int,
    reports: list,
    metadaten: dict,
    config: dict,
    llm,
    mapper: OntologyMapper,
    report_workers: int,
    output_file,
    file_lock: threading.Lock,
) -> dict:
    """
    Führt einen einzelnen Extraktionslauf durch und schreibt die Ergebnisse
    direkt in die JSONL-Datei (thread-safe).

    Returns:
        Statistik-Dict mit Anzahl verarbeiteter Berichte, fehlgeschlagener
        Berichte und Dauer.
    """
    _local = threading.local()

    def _process(report) -> tuple[int, list[dict], list]:
        if not hasattr(_local, "pipeline"):
            _local.pipeline = NERPipeline(config, llm, mapper)
        rows, entities = _local.pipeline.run(report.report_id, report.raw_text)
        return report.report_id, rows, entities

    ergebnisse: dict[int, tuple[list[dict], list]] = {}
    failed_reports: list[str] = []

    extract_start = datetime.now()
    with ThreadPoolExecutor(max_workers=report_workers,
                            thread_name_prefix=f"ner-run{run_id}") as exc:
        futures = {exc.submit(_process, r): r for r in reports}

        with tqdm(total=len(reports), desc=f"Run {run_id}: Extracting entities",
                  position=run_id, leave=True) as pbar:
            for fut in as_completed(futures):
                report = futures[fut]
                report_ref = metadaten[report.report_id].dateiname
                try:
                    report_id, rows, entities = fut.result()
                    ergebnisse[report_id] = (rows, entities)
                    mapped = sum(1 for r in rows if r["concept_id"])
                    logger.info("Run %d, Report %s: %d Entitäten, %d ontology-gemappt",
                                run_id, report_ref, len(rows), mapped)
                except Exception as e:
                    logger.exception("Run %d, Report %s: Verarbeitung fehlgeschlagen: %s",
                                     run_id, report_ref, e)
                    failed_reports.append(report_ref)
                finally:
                    pbar.update(1)
    extract_end = datetime.now()

    # Ergebnisse für diesen Run vorbereiten und in Datei schreiben
    records_to_write = []
    for report in reports:
        if report.report_id not in ergebnisse:
            continue  # fehlgeschlagene Berichte überspringen

        rows, entities = ergebnisse[report.report_id]
        meta = metadaten[report.report_id]

        spans = []
        for entity, row in zip(entities, rows):
            spans.append({
                "start": entity.start,
                "end": entity.end,
                "text": entity.text,
                "label": entity.label.value,
                "terminology": row["ontology"],
                "concept_id": row["concept_id"],
                "search_term": entity.search_term,
                "meta": {
                    "source": entity.source.value,
                    "assertion": entity.assertion.value,
                    "confidence": row["confidence"],
                },
            })

        record = {
            "text_id": Path(meta.dateiname).stem,
            "text": report.raw_text,
            "spans": spans,
            "meta": {
                "source": "patientdb",
                "table": "case_report",
                "report_id": report.report_id,
                "anonym_case_id": meta.anonym_case_id,
                "dateiname": meta.dateiname,
                "berichtstyp": meta.berichtstyp,
                "jahr": meta.jahr,
                # Run-spezifische Metadaten
                "run_id": run_id,
                "run_timestamp": extract_start.isoformat(),
            },
        }
        records_to_write.append(record)

    # Thread-safe in Datei schreiben
    with file_lock:
        for record in records_to_write:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()

    n_processed = len(ergebnisse) or len(reports)
    elapsed = (extract_end - extract_start).total_seconds()

    return {
        "run_id": run_id,
        "n_exported": len(records_to_write),
        "n_failed": len(failed_reports),
        "failed_reports": failed_reports,
        "elapsed_seconds": elapsed,
        "reports_per_second": n_processed / elapsed if elapsed > 0 else 0,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Hybrid-NER (src/-Pipeline) über das Testset berichte.xlsx "
                    "ausführen und als JSONL exportieren. Unterstützt mehrere Läufe."
    )
    parser.add_argument("-i", "--input", type=Path, default=Path("experiments/berichte.xlsx"),
                        help="Testset-Liste (default: experiments/berichte.xlsx)")
    parser.add_argument("-o", "--output", type=Path, default=Path("experiments/hybrid_medgemma27b.jsonl"),
                        help="Ausgabe-JSONL (default: experiments/hybrid_ner.jsonl)")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Nur die ersten N Berichte verarbeiten")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Report-Worker pro Run (Default: config report_workers)")
    parser.add_argument("-mw", "--map-workers", type=int, default=None,
                        help="Mapping-Worker pro Pipeline (Default: config mapping_workers)")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="Anzahl der Extraktionsläufe (default: 1)")
    parser.add_argument("--run-workers", type=int, default=1,
                        help="Anzahl paralleler Runs (default: 1, sequentiell). "
                             "Vorsicht: LLM-Extraktor ist nicht unbedingt thread-safe für parallele Runs!")

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs muss mindestens 1 sein")

    settings = Settings()
    config = load_pipeline_config()
    setup_logging(log_dir=config["logging"]["log_dir"], level=config["logging"]["log_level"])

    # SQLite-Sample-Modus: selbstlaufendes, synthetisches Testset ohne Postgres.
    # Aktiviert, wenn PATHOHERO_SQLITE auf eine existierende Datei zeigt.
    _sqlite_path = (os.getenv("PATHOHERO_SQLITE") or "").strip()
    use_sqlite = bool(_sqlite_path) and os.path.exists(_sqlite_path)

    if args.workers:
        report_workers = int(args.workers)
    else:
        report_workers = int(config["pipeline"].get("report_workers", 1))
    if args.map_workers:
        config["pipeline"]["mapping_workers"] = int(args.map_workers)

    # Warnung bei parallelen Runs mit LLM
    if args.run_workers > 1 and config["pipeline"].get("use_ollama", True):
        logger.warning("Parallele Runs mit Ollama-LLM: Stelle sicher, dass Ollama "
                       "genug Ressourcen hat. Bei Problemen --run-workers 1 verwenden.")

    # Repository: entweder das SQLite-Sample (selbstlaufend, synthetische Daten)
    # oder die PostgreSQL-Datenbank (Patientendaten; erfordert Zugriff).
    if use_sqlite:
        sqlite_path = os.getenv("PATHOHERO_SQLITE", "").strip() or "experiments/sample_reports.sqlite"
        if not os.path.exists(sqlite_path):
            raise SystemExit(
                f"SQLite-Datei {sqlite_path} nicht gefunden.\n"
                "Erstellen Sie sie mit:\n"
                "    python3 experiments/make_synthetic_testset.py\n"
                "    python3 experiments/make_sqlite_sample_db.py"
            )
        repo = SampleStore(f"sqlite:///{os.path.abspath(sqlite_path)}")
        repo.create_missing_tables()
    else:
        repo = Repository(settings.database_url, pool_size=report_workers + 2, max_overflow=report_workers)
        repo.create_missing_tables()
        repo.ensure_processed_at_column()

    # Initialize LLM based on config
    llm = None
    if config["pipeline"].get("use_ollama", True):
        # Auto-detect a reachable Ollama server and pick a sensible model
        # (medgemma1.5:4b → medgemma:27b → qwen3.8:27b → configured default)
        # when no OLLAMA_MODEL is explicitly set or the configured model is
        # not available. Falls back to gazetteer + regex only if nothing is
        # reachable.
        ollama_base = settings.ollama_base_url.rstrip("/")
        ollama_models: list = []
        try:
            with urllib.request.urlopen(
                    f"{ollama_base}/api/tags", timeout=3) as r:
                ollama_models = _json.loads(r.read().decode())["models"]
        except Exception:
            ollama_models = []

        if ollama_models:
            available = {m["name"] for m in ollama_models}
            default_model = settings.ollama_model
            model = default_model
            if settings.ollama_model not in available:
                # Prefer the smallest medical model, then any available one
                for cand in ("medgemma1.5:4b", "medgemma:27b", "qwen3.8:27b"):
                    if cand in available:
                        model = cand
                        logger.info("Ollama: configured model %r nicht "
                                    "verfügbar – verwende %r",
                                    default_model, model)
                        break
                else:
                    model = default_model
            llm = LLMExtractor(ollama_base, model, settings.ollama_temperature)
            logger.info("Ollama-LLM aktiv: %s (%d Modelle verfügbar, gewähltes: %s)",
                        ollama_base, len(ollama_models), model)
        else:
            # No reachable Ollama server → skip LLM silently
            logger.info("Ollama-Server unter %s nicht erreichbar – "
                        "LLM-Fallback deaktiviert (Gazetteer + Regex nur).",
                        ollama_base)
    elif config["pipeline"].get("use_huggingface", False):
        llm = HuggingFaceExtractor(settings.hf_model, settings.hf_device)
        if report_workers > 1:
            logger.warning("HuggingFace-Extractor mit %d Workern: das Modell wird von allen Threads geteilt. Bei Problemen report_workers=1 setzen.", report_workers)

    snow, umls = baue_ontology_clients_auf(settings, config, repo)

    # In SQLite-Sample-Modus: kein Snowstorm-Server und keine UMLS-REST-API
    # (beide setzen voraus, dass ein externer Service läuft) → Mapping deaktivieren.
    # Für das lokale Sample-DB-Persistieren genügt ``use_ontology=False``;
    # die extrahierten Spans werden ohne concept_id als NULL gespeichert.
    if use_sqlite and (umls is None or umls.__class__.__name__.startswith("LocalUMLS")):
        logger.info("SQLite-Sample: Ontology-Mapping deaktiviert "
                    "(kein UMLS-/Snowstorm-Client verfügbar).")
        config["pipeline"]["use_ontology"] = False
        snow, umls = None, None

    # Nur wenn Ontology-Mapping aktiv ist, OntologyMapper instanziieren.
    if config["pipeline"].get("use_ontology"):
        mapper = OntologyMapper(
            repo=repo,
            snowstorm_client=snow,
            umls_client=umls,
            default_provider=config["pipeline"].get("ontology_provider"),
            provider_per_label=config["pipeline"].get("ontology_provider_per_label"),
            snowstorm_terminology=settings.snowstorm_terminology,
            label_ecl=config.get("label_ecl", {}),
            ancestor_depth=config["pipeline"].get("ancestor_depth", 1),
            umls_sabs=settings.umls_sabs_list,
            umls_terminology_version=settings.umls_version,
        )
    else:
        mapper = None

    # Testset-Liste laden
    df = pd.read_excel(args.input)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)
    report_ids = df["report_id"].astype(int).tolist()
    metadaten = {int(row.report_id): row for row in df.itertuples(index=False)}
    print(f"{len(report_ids)} Berichte aus {args.input} geladen")

    # Berichte laden: bevorzugt aus dem SQLite-Sample (synthetische Texte aus
    # berichte.xlsx), sonst aus der PostgreSQL-Datenbank.
    reports = _load_reports(
        settings, config, repo, report_ids, metadaten, args.input)
    print(f"{len(reports)} Berichte mit Text geladen")

    if not reports:
        print("Keine Berichte zu verarbeiten.")
        return

    # Ausgabe-Datei vorbereiten (überschreiben)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_lock = threading.Lock()

    # Gesamtstatistik
    all_run_stats = []
    total_start = datetime.now()

    print(f"\n{'='*60}")
    print(f"Starte {args.runs} Extraktionslauf/läufe mit je {len(reports)} Berichten")
    print(f"Report-Worker pro Run: {report_workers}")
    print(f"Parallele Runs: {args.run_workers}")
    print(f"Ausgabe: {args.output.resolve()}")
    print(f"{'='*60}\n")

    with open(args.output, "w", encoding="utf-8") as output_file:
        # Schreibe Header-Metadaten als erste Zeile (optional, als Kommentar oder separates Feld)
        header_record = {
            "meta_type": "experiment_header",
            "experiment": {
                "n_runs": args.runs,
                "n_reports": len(reports),
                "report_workers_per_run": report_workers,
                "parallel_runs": args.run_workers,
                "model": settings.ollama_model if config["pipeline"].get("use_ollama", True) else settings.hf_model,
                "started_at": total_start.isoformat(),
                "input_file": str(args.input),
                "output_file": str(args.output),
            }
        }
        output_file.write(json.dumps(header_record, ensure_ascii=False) + "\n")
        output_file.flush()

        if args.run_workers == 1:
            # Sequentielle Ausführung der Runs
            for run_id in range(1, args.runs + 1):
                print(f"\n--- Run {run_id}/{args.runs} ---")
                stats = run_single_extraction(
                    run_id=run_id,
                    reports=reports,
                    metadaten=metadaten,
                    config=config,
                    llm=llm,
                    mapper=mapper,
                    report_workers=report_workers,
                    output_file=output_file,
                    file_lock=file_lock,
                )
                all_run_stats.append(stats)

                print(f"Run {run_id} abgeschlossen: {stats['n_exported']} Berichte exportiert, "
                      f"{stats['n_failed']} fehlgeschlagen, {stats['elapsed_seconds']:.1f}s "
                      f"({stats['reports_per_second']:.3f} Berichte/s)")
                if stats['failed_reports']:
                    print(f"  Fehlgeschlagen: {stats['failed_reports']}")
        else:
            # Parallele Ausführung der Runs (mit Vorsicht genießen!)
            with ThreadPoolExecutor(max_workers=args.run_workers,
                                    thread_name_prefix="run") as run_executor:
                futures = {
                    run_executor.submit(
                        run_single_extraction,
                        run_id=run_id,
                        reports=reports,
                        metadaten=metadaten,
                        config=config,
                        llm=llm,
                        mapper=mapper,
                        report_workers=report_workers,
                        output_file=output_file,
                        file_lock=file_lock,
                    ): run_id
                    for run_id in range(1, args.runs + 1)
                }

                for fut in as_completed(futures):
                    run_id = futures[fut]
                    try:
                        stats = fut.result()
                        all_run_stats.append(stats)
                        print(f"Run {run_id} abgeschlossen: {stats['n_exported']} Berichte exportiert, "
                              f"{stats['n_failed']} fehlgeschlagen, {stats['elapsed_seconds']:.1f}s")
                    except Exception as e:
                        logger.exception("Run %d fehlgeschlagen: %s", run_id, e)
                        all_run_stats.append({
                            "run_id": run_id,
                            "n_exported": 0,
                            "n_failed": len(reports),
                            "failed_reports": [metadaten[r.report_id].dateiname for r in reports],
                            "elapsed_seconds": 0,
                            "error": str(e),
                        })

    total_end = datetime.now()
    total_elapsed = (total_end - total_start).total_seconds()

    # Finale Statistik ausgeben
    print(f"\n{'='*60}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    print(f"Gesamtläufe: {args.runs}")
    print(f"Berichte pro Lauf: {len(reports)}")
    print(f"Gesamte exportierte Datensätze: {sum(s['n_exported'] for s in all_run_stats)}")
    print(f"Gesamt fehlgeschlagen: {sum(s['n_failed'] for s in all_run_stats)}")
    print(f"Gesamtdauer: {total_elapsed:.1f}s")
    print(f"Durchschnitt pro Run: {total_elapsed / args.runs:.1f}s")
    print(f"\nJSONL-Datei: {args.output.resolve()}")
    print(f"Dateigröße: {args.output.stat().st_size / 1024 / 1024:.2f} MB")

    # Detaillierte Run-Statistik als JSON speichern (optional)
    stats_file = args.output.with_suffix('.stats.json')
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": {
                "n_runs": args.runs,
                "n_reports": len(reports),
                "total_elapsed_seconds": total_elapsed,
                "started_at": total_start.isoformat(),
                "ended_at": total_end.isoformat(),
            },
            "runs": all_run_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f"Statistik-Datei: {stats_file.resolve()}")

    # Optional: extrahierte Spans in die SQLite-SampleDB persistieren, damit
    # lokale Abfragen (z. B. `SELECT * FROM report_concept`) die Ergebnisse
    # nachvollziehbar zeigen.
    sqlite_path = (os.getenv("PATHOHERO_SQLITE") or "").strip()
    if sqlite_path and os.path.exists(sqlite_path) and reports:
        _persist_to_sqlite(reports, metadaten, args.output, sqlite_path)


def _persist_to_sqlite(reports, metadaten: dict, output_jsonl: Path, sqlite_path: str) -> None:
    """Liest das JSONL-File und schreibt die Spans in report_concept."""
    import sqlite3 as _sqlite
    import json as _json

    rows_by_rid: dict[int, list[tuple]] = {}
    with open(output_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            if rec.get("meta_type") == "experiment_header":
                continue
            rid = rec["meta"]["report_id"]
            for sp in rec.get("spans", []):
                rows_by_rid.setdefault(rid, []).append((
                    sp["start"], sp["end"], sp["text"], sp["label"],
                    sp.get("search_term"), sp.get("terminology"),
                    sp.get("concept_id"),
                ))

    conn = _sqlite.connect(sqlite_path)
    try:
        n = 0
        with conn:
            for rid, spans in rows_by_rid.items():
                for (start, end, text, label, search_term, ontology, concept_id) in spans:
                    conn.execute(
                        """INSERT OR REPLACE INTO report_concept
                           (report_id, concept_id, source_field, surface_text,
                            raw_text, search_term, ontology, start_offset,
                            end_offset, confidence)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (rid, concept_id, "raw_text", text, text,
                         search_term, ontology, start, end, 1.0))
                    n += 1
        total = conn.execute("SELECT COUNT(*) FROM report_concept").fetchone()[0]
    finally:
        conn.close()
    print(f"SQLite-Sample: {n} neue Spans in report_concept persistiert "
          f"(Gesamt {total})")


if __name__ == "__main__":
    main()
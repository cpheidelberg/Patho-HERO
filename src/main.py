import argparse
import logging
import numpy as np
import threading
from rainbow_tqdm import tqdm
import threading

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.config import Settings, load_pipeline_config
from src.database import Repository
from src.extractors.llm_extractor import LLMExtractor
from src.extractors.hf_extractor import HuggingFaceExtractor
from src.terminology.snowstorm import SnowstormClient
from src.terminology.umls import UMLSClient
from src.terminology.local_umls import LocalUMLSClient
from src.terminology.mapper import OntologyMapper
from src.pipeline import NERPipeline
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _local_umls_url(settings: Settings) -> str:
    """Return the configured local UMLS URL in the existing app database."""
    if settings.umls_local_db_url:
        return settings.umls_local_db_url
    return settings.database_url


def _create_umls_client(settings: Settings, repo: Repository):
    """Prefer local UMLS when reachable; otherwise use the REST client."""
    local_url = _local_umls_url(settings)
    try:
        local = LocalUMLSClient(
            settings.umls_api_key,
            version=settings.umls_version,
            repository=repo if local_url.rstrip("/") == settings.database_url.rstrip("/") else None,
            database_url=local_url,
            schema=settings.umls_local_schema or None,
            language=settings.umls_local_language,
        )
        with local._repo.engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1 FROM mrconso LIMIT 1")
        try:
            local.ensure_indexes()
        except Exception as exc:
            logger.warning("UMLS: lokale Lookup-Indizes konnten nicht geprüft/angelegt werden (%s)", exc)
        logger.info("UMLS: lokaler PostgreSQL-Client aktiviert (%s)", local_url)
        return local
    except Exception as exc:
        logger.warning("UMLS: lokale Datenbank nicht verfügbar (%s)", exc)

    if settings.umls_api_key:
        logger.info("UMLS: verwende REST API als Fallback")
        return UMLSClient(settings.umls_api_key, version=settings.umls_version)
    logger.warning("UMLS: weder lokale Datenbank noch UMLS_API_KEY verfügbar")
    return None


def main():

    settings = Settings()
    config = load_pipeline_config()
    setup_logging(log_dir=config["logging"]["log_dir"], level=config["logging"]["log_level"])
    
    report_workers = int(config["pipeline"].get("report_workers", 1))
    mapping_workers = int(config["pipeline"].get("mapping_workers", 1))
    db_pool_size = report_workers * mapping_workers + report_workers + 2

    repo = Repository(
        settings.database_url,
        pool_size=db_pool_size,
        max_overflow=report_workers,
    )
    repo.create_missing_tables()
    repo.ensure_processed_at_column()

    # Initialize extractors based on config
    llm = None
    if config["pipeline"].get("use_ollama", True):
        # Set OLLAMA_KEEP_ALIVE from config before creating the extractor
        keep_alive = config["pipeline"].get("ollama_keep_alive", os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
        os.environ["OLLAMA_KEEP_ALIVE"] = keep_alive
        logger.info(f"Ollama keep-alive: {keep_alive}")
        llm = LLMExtractor(settings.ollama_base_url, settings.ollama_model,
                           settings.ollama_temperature)
    elif config["pipeline"].get("use_huggingface", False):
        llm = HuggingFaceExtractor(settings.hf_model_name, settings.hf_device)
        if report_workers > 1:
            logger.warning("HuggingFace-Extractor mit %d Workern: das Modell wird von allen Threads geteilt. Bei Problemen report_workers=1 setzen.", report_workers)

    default_provider = config["pipeline"].get("ontology_provider")
    provider_per_label = config["pipeline"].get("ontology_provider_per_label")

    # Optional: Snowstorm Client
    snow = None
    if default_provider == "snowstorm" or provider_per_label and any(p == "snowstorm" for p in provider_per_label.values()):
        snow = SnowstormClient(settings.snowstorm_base_url,
                               settings.snowstorm_branch,
                               settings.snowstorm_language,
                               settings.snowstorm_accept_language)

    # Optional: UMLS Client
    umls = None
    if default_provider == "umls" or provider_per_label and any(p == "umls" for p in provider_per_label.values()):
        umls = _create_umls_client(settings, repo)

    mapper = OntologyMapper(
        repo=repo,
        snowstorm_client=snow,
        umls_client=umls,
        default_provider=default_provider,
        provider_per_label=provider_per_label,
        snowstorm_terminology=settings.snowstorm_terminology,
        label_ecl=config.get("label_ecl", {}),
        ancestor_depth=config["pipeline"].get("ancestor_depth", 1),
        umls_sabs=settings.umls_sabs_list,
        umls_terminology_version=settings.umls_version,
    )

    _local = threading.local()

    n_reports = config["import"].get("limit")
    n_entities = 0
    n_mapped = 0
    n_confidence = []
    failed_reports: list[str] = []
    failed_report_log = os.path.join(config["logging"]["log_dir"], "failed_reports.txt")
    log_lock = threading.Lock()

    reports = repo.get_reports(limit=n_reports, processed=None if config["pipeline"].get("reprocess") else False)
    desc = "Processing reports" if config["pipeline"].get("reprocess") else "Processing unprocessed reports"

    def _ref(report) -> str:
        return report.original_case_id or str(report.report_id)

    def _process(report):
        if not hasattr(_local, "pipeline"):
            _local.pipeline = NERPipeline(config, llm, mapper)
        return _local.pipeline.run(report.report_id, report.raw_text)

    with ThreadPoolExecutor(max_workers=report_workers,
                            thread_name_prefix="etl") as exc:
        futures = {exc.submit(_process, r): r for r in reports}

        with tqdm(total=len(reports), desc=desc) as pbar:
            for fut in as_completed(futures):
                report = futures[fut]
                report_ref = _ref(report)
                try:
                    rows, entities = fut.result()
                    repo.save_report_concepts(report.report_id, rows)
                    repo.mark_report_processed(report.report_id)

                    mapped = sum(1 for r in rows if r["concept_id"])
                    logger.info(f"Report {report_ref}: {len(rows)} Entitäten, {mapped} ontology-gemappt")
                    n_entities += len(entities)
                    n_mapped += mapped

                    confidences = []
                    for e in entities:
                        try:
                            confidences.append(float(e.confidence))
                        except (TypeError, ValueError):
                            continue
                    if confidences:
                        n_confidence.append(np.mean(confidences))
                except Exception as e:
                    logger.exception(f"Report {report_ref}: Verarbeitung fehlgeschlagen: {e}")
                    failed_reports.append(report_ref)
                    with log_lock:
                        with open(failed_report_log, "a", encoding="utf-8") as f:
                            f.write(f"{report_ref}\n")
                finally:
                    pbar.update(1)

    if failed_reports:
        logger.warning(f"{len(failed_reports)} Berichte fehlgeschlagen (siehe {failed_report_log})")
    processed_count = len(reports) - len(failed_reports)
    active_model = settings.ollama_model if llm is not None and config["pipeline"].get("use_ollama", True) else settings.hf_model
    logger.info(f"Extracted {processed_count} reports with {active_model}: {n_entities} entities with {n_mapped} mapped and confidence {np.mean(n_confidence) if n_confidence else 'N/A'}")

if __name__ == "__main__":
    main()
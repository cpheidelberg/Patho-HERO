"""Konfigurations-Settings (pydantic-settings) – lokale UMLS-Postgres-Optionen
hinzugefügt.

Konventionen (identisch zur bestehenden Datei):
- ``pydantic_settings.BaseSettings``
- Defaults aus ``os.getenv``
- Helper-Funktionen ``_parse_list`` und ``_parse_map``
- Properties ``database_url``, ``umls_sabs_list``
- PEP 604 Type Hints

NEU für lokalen UMLS-Postgres-Modus:
- ``umls_local_db_url``   : DSN für die lokale UMLS-DB
- ``umls_local_schema``   : Optionaler Schemaname
- ``umls_local_enabled``  : ``True`` -> ``LocalUMLSClient`` statt API
- ``umls_local_language`` : Sprachfilter (Default ``"ENG"``)

Die übrigen Felder sind 1:1 aus dem Original übernommen, damit
``main.py`` ohne Änderung weiterläuft.
"""

import json
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_map(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not value:
        return result
    for part in value.split(","):
        if ":" not in part:
            continue
        label, provider = part.split(":", 1)
        result[label.strip()] = provider.strip()
    return result


class Settings(BaseSettings):
    # DB
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", 5432))
    db_name: str = os.getenv("DB_NAME", "patho_ner")
    db_user: str = os.getenv("DB_USER", "")
    db_password: str = os.getenv("DB_PASSWORD", "")
    # Import
    import_directory: str = os.getenv("IMPORT_DIRECTORY", "data/reports")
    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "medgemma:27b")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
    # HuggingFace
    hf_model: str = os.getenv("HF_MODEL", "d4data/biomedical-ner-all")
    hf_device: str = os.getenv("HF_DEVICE", "cpu")
    # Ontology / Snowstorm
    snowstorm_base_url: str = os.getenv("SNOWSTORM_BASE_URL", "http://localhost:8080")
    snowstorm_branch: str = os.getenv("SNOWSTORM_BRANCH", "MAIN/SNOMEDCT-DE")
    snowstorm_language: str = os.getenv("SNOWSTORM_LANGUAGE", "de")
    snowstorm_terminology: str = os.getenv("SNOWSTORM_TERMINOLOGY")
    ontology_default: str = os.getenv("ONTOLOGY_DEFAULT", "snowstorm")
    ontology_label_provider: str = os.getenv("ONTOLOGY_LABEL_PROVIDER", "")
    # UMLS
    umls_api_key: str = os.getenv("UMLS_API_KEY", "")
    umls_version: str = os.getenv("UMLS_VERSION", "current")
    umls_sabs: str = os.getenv("UMLS_SABS", "SNOMEDCT_US")
    # --- NEU: lokaler UMLS-Postgres-Modus ---
    umls_local_enabled: bool = (
        os.getenv("UMLS_LOCAL_ENABLED", "false").lower()
        in ("1", "true", "yes", "on")
    )
    umls_local_db_url: str = os.getenv("UMLS_LOCAL_DB_URL", "")
    umls_local_schema: str = os.getenv("UMLS_LOCAL_SCHEMA", "")
    umls_local_language: str = os.getenv("UMLS_LOCAL_LANGUAGE", "ENG")
    umls_local_rrf_meta_dir: str = os.getenv("UMLS_LOCAL_RRF_META_DIR", "")

    @property
    def label_provider_map(self) -> dict[str, str]:
        return _parse_map(self.ontology_label_provider)

    @property
    def umls_sabs_list(self) -> list[str]:
        text = self.umls_sabs.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return []
        return _parse_list(text)

    @property
    def database_url(self) -> str:
        auth = f"{self.db_user}:{self.db_password}@" if self.db_user else ""
        return (
            f"postgresql+psycopg2://{auth}{self.db_host}:"
            f"{self.db_port}/{self.db_name}"
        )


def load_pipeline_config(path: str = "config/config.yaml") -> dict:
    with open(Path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)

import os
import sys
import argparse
import hashlib
import hmac
from typing import Optional
from tqdm import tqdm
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import Settings, load_pipeline_config
from database import Repository, CaseReport
from ingestion.loader import ReportFile, ReportFileLoader


class ReportImporter:
    """Upsert von ReportFile-Objekten in case_report.

    Idempotent: Wiederholtes Importieren aktualisiert bestehende Zeilen
    (Schlüssel: case_id_alias) statt Duplikate zu erzeugen."""

    def __init__(self, repo: Repository, pseudonym_salt: Optional[str] = None):
        self.repo = repo
        self.salt = pseudonym_salt

    def _alias(self, rf: ReportFile) -> str:
        """Pseudonymisierte Fall-ID (deterministisch via HMAC)."""
        if self.salt:
            digest = hmac.new(
                self.salt.encode(),
                rf.original_case_id.encode(),
                hashlib.sha256,
            ).hexdigest()
            return f"CASE-{digest[:12].upper()}"
        return rf.original_case_id

    def import_reports(self, reports: list[ReportFile]) -> tuple[int, int]:
        """Gibt (importiert, übersprungen) zurück."""
        imported = 0
        skipped = 0

        with self.repo.session_scope() as session:
            for rf in tqdm(reports, desc="Processing reports", unit="reports"):
                if not rf.raw_text:
                    skipped += 1
                    continue

                stmt = insert(CaseReport).values(
                    case_id_alias=self._alias(rf),
                    anonym_case_id=rf.original_case_id,
                    parent_folder=rf.parent_folder,
                    filename=rf.filename,
                    report_year=rf.report_year,
                    raw_text=rf.raw_text,
                    created_at=datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["case_id_alias"],
                    set_={
                        "raw_text": rf.raw_text,
                        "filename": rf.filename,
                        "parent_folder": rf.parent_folder,
                        "report_year": rf.report_year,
                    },
                )
                session.execute(stmt)
                imported += 1

        return imported, skipped


def main():

    settings = Settings()
    cfg = load_pipeline_config().get("import", {})

    loader = ReportFileLoader(
        directory=cfg.get("directory", "data/reports"),
        pattern=cfg.get("pattern", "*.txt"),
        recursive=cfg.get("recursive", False),
        limit=cfg.get("limit"),
        encodings=cfg.get("encodings", ["utf-8", "cp1252", "latin-1"]),
        filename_pattern=cfg.get("filename_pattern"),
    )

    print(f"Suche Berichte in: {loader.directory} "
          f"(Pattern: {loader.pattern}, Limit: {loader.limit})")
    files = loader.load()
    print(f"{len(files)} Dateien erfolgreich gelesen")

    if not files:
        return

    repo = Repository(settings.database_url)
    importer = ReportImporter(repo, pseudonym_salt=settings.pseudonym_salt)
    imported, skipped = importer.import_reports(files)
    print(f"Fertig: {imported} importiert/aktualisiert, {skipped} übersprungen")


if __name__ == "__main__":
    main()
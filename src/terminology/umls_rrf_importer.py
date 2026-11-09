"""Bulk-Importer für UMLS ``Full precomputed subset`` (RRF-Dateien).

Liest die Pipe-getrennten Dateien

* ``MRCONSO.RRF``
* ``MRSTY.RRF``
* ``MRREL.RRF``
* ``MRDEF.RRF``

aus einem META-Ordner und befüllt damit die Tabellen ``mrconso``,
``mrsty``, ``mrrel`` und ``mrdef`` in der konfigurierten Postgres-Datenbank.

Konventionen (analog zu ``importer.py``, ``database.py``):
- Logger via ``logging.getLogger(__name__)``
- Type Hints (PEP 604)
- Deutsche Docstrings
- ``session_scope``-Pattern aus ``database.Repository``
- SQLAlchemy ``Base`` aus ``database``

Effizienz:
- Streaming, Generator-basiert
- ``psycopg2.copy_expert`` via ``raw_connection`` für maximalen
  Durchsatz, Fallback auf ``executemany`` in Batches.

CLI:
    python -m src.terminology.umls_rrf_importer --meta-dir /path/to/META \
        [--truncate] [--batch-size 50000] [--schema myschema] [--no-index]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from sqlalchemy import Column, String, Text, Index, text, func
from sqlalchemy.dialects.postgresql import TSVECTOR
# SQLAlchemy 2.x: declarative_base lives in sqlalchemy.orm
# SQLAlchemy 1.4+: same path
# SQLAlchemy 1.3: requires sqlalchemy.ext.declarative
try:
    from sqlalchemy.orm import declarative_base
except ImportError:  # pragma: no cover - legacy fallback
    from sqlalchemy.ext.declarative import declarative_base

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

logger = logging.getLogger(__name__)


# Spaltendefinitionen laut UMLS-RRF-Doku.
# Die letzte Spalte jeder Datei ist ein leeres Feld wegen der
# abschließenden Pipe und wird verworfen.
MRCONSO_COLS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI",
    "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF",
]
MRSTY_COLS = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
MRREL_COLS = [
    "CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA",
    "RUI", "SRUI", "SAB", "SL", "RG", "DIR", "SUPPRESS", "CVF",
]
MRDEF_COLS = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF"]


# ===========================================================================
# SQLAlchemy Models für die UMLS-Tabellen
# ===========================================================================
#
# Wir nutzen einen frischen Declarative-Base, damit wir die UMLS-Tabellen
# unabhängig von der App-DB (``database.Base``) verwalten können und auch
# dann anlegen können, wenn database.py nicht importiert werden kann.

UmlsBase = declarative_base()


class MrConso(UmlsBase):
    """MRCONSO.RRF – Concepts, Atoms, Strings (UMLS Metathesaurus)."""

    __tablename__ = "mrconso"

    # Wir setzen AUI als Primary Key (Atom Unique Identifier).
    # RRF-Reihenfolge ist zugleich Bulk-Insert-Reihenfolge.
    CUI = Column(Text, index=True, nullable=False)
    LAT = Column(Text, index=True, nullable=False)
    TS = Column(Text)
    LUI = Column(Text)
    STT = Column(Text)
    SUI = Column(Text)
    ISPREF = Column(Text)
    AUI = Column(Text, primary_key=True)
    SAUI = Column(Text)
    SCUI = Column(Text)
    SDUI = Column(Text)
    SAB = Column(Text, index=True, nullable=False)
    TTY = Column(Text)
    CODE = Column(Text, index=True)
    STR = Column(Text, nullable=False)
    SRL = Column(Text)
    SUPPRESS = Column(Text)
    CVF = Column(Text)


class MrSty(UmlsBase):
    """MRSTY.RRF – Semantic Types."""

    __tablename__ = "mrsty"

    # RRF-Reihenfolge ist zugleich Bulk-Insert-Reihenfolge.
    CUI = Column(Text, primary_key=True, nullable=False)
    TUI = Column(Text, primary_key=True, nullable=False)
    STN = Column(Text, primary_key=True, nullable=False)
    STY = Column(Text)
    ATUI = Column(Text)
    CVF = Column(Text)


class MrRel(UmlsBase):
    """MRREL.RRF – Relations."""

    __tablename__ = "mrrel"

    CUI1 = Column(Text, index=True)
    AUI1 = Column(Text)
    STYPE1 = Column(Text)
    REL = Column(Text, index=True)
    CUI2 = Column(Text, index=True)
    AUI2 = Column(Text)
    STYPE2 = Column(Text)
    RELA = Column(Text)
    SRUI = Column(Text)
    SAB = Column(Text, index=True)
    SL = Column(Text)
    RG = Column(Text)
    DIR = Column(Text)
    SUPPRESS = Column(Text)
    CVF = Column(Text)
    RUI = Column(Text, primary_key=True)


class MrDef(UmlsBase):
    """MRDEF.RRF – Definitions."""

    __tablename__ = "mrdef"

    CUI = Column(Text, index=True, nullable=False)
    AUI = Column(Text)
    ATUI = Column(Text, primary_key=True)
    SATUI = Column(Text)
    SAB = Column(Text, index=True)
    DEF = Column(Text, nullable=False)
    SUPPRESS = Column(Text)
    CVF = Column(Text)


# ===========================================================================
# Parser
# ===========================================================================

def _iter_rrf(path: Path, n_cols: int) -> Iterator[tuple]:
    """Streamt eine RRF-Datei zeilenweise und liefert Tupel ohne trailing
    leeres Feld.
    """
    if not path.exists():
        raise FileNotFoundError(f"RRF-Datei fehlt: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            # Letzte Spalte ist immer leer -> abschneiden
            if len(parts) and parts[-1] == "":
                parts = parts[:-1]
            # Auf n_cols kürzen/auffüllen
            if len(parts) < n_cols:
                parts = parts + [""] * (n_cols - len(parts))
            yield tuple(parts[:n_cols])


def _coerce(value):
    """Coerce None -> ''."""
    return "" if value is None else value


# ===========================================================================
# Importer
# ===========================================================================

class RRFImporter:
    """Bulk-Importer für UMLS-RRF-Dateien."""

    def __init__(self, database_url: str, *, schema: str | None = None,
                 batch_size: int = 50_000, truncate: bool = True,
                 create_indexes: bool = True):
        from sqlalchemy import create_engine

        self.database_url = database_url
        self.schema = schema
        self.batch_size = batch_size
        self.truncate = truncate
        self.create_indexes = create_indexes

        self.engine = create_engine(database_url, future=True)
        self._stats: dict[str, dict] = {}

    # ------------------------------------------------------------------ public API
    def import_directory(self, meta_dir: str | Path,
                         only: Iterable[str] | None = None) -> dict[str, int]:
        """Importiert ausgewählte RRF-Dateien aus ``meta_dir``.

        Args:
            meta_dir: Pfad zum META-Ordner (enthält MRCONSO.RRF,
                MRSTY.RRF, MRREL.RRF, MRDEF.RRF).
            only: Optionale Teilmenge der Tabellen, z.B.
                ``{"MRREL", "MRDEF"}``. Damit kann man fehlende
                Relationen nachladen, ohne die vorhandenen ``mrconso``/
                ``mrsty`` zu berühren. Default ``None`` = alle vier.

        Returns:
            Dict mit ``{dateiname: importierte_zeilen}``.
        """
        meta_dir = Path(meta_dir)
        if not meta_dir.is_dir():
            raise FileNotFoundError(f"META-Verzeichnis fehlt: {meta_dir}")

        valid = {"MRCONSO", "MRSTY", "MRREL", "MRDEF"}
        wanted = {t.upper() for t in only} if only else valid
        unknown = wanted - valid
        if unknown:
            raise ValueError(f"Unbekannte Tabellen: {sorted(unknown)}; "
                             f"zulässig: {sorted(valid)}")

        logger.info("RRFImporter | Geplante Tabellen: %s",
                    ", ".join(sorted(wanted)))

        self._ensure_schema()
        self._ensure_tables()

        if self.truncate:
            self._truncate_all(wanted)

        started = time.monotonic()

        results = {}
        if "MRCONSO" in wanted:
            results["MRCONSO.RRF"] = self._import_conso(meta_dir / "MRCONSO.RRF")
        if "MRSTY" in wanted:
            results["MRSTY.RRF"] = self._import_sty(meta_dir / "MRSTY.RRF")
        if "MRREL" in wanted:
            results["MRREL.RRF"] = self._import_rel(meta_dir / "MRREL.RRF")
        if "MRDEF" in wanted:
            results["MRDEF.RRF"] = self._import_def(meta_dir / "MRDEF.RRF")

        if self.create_indexes:
            self._create_indexes()

        elapsed = time.monotonic() - started
        total = sum(results.values())
        logger.info(
            "RRFImporter | Import abgeschlossen: %d Zeilen in %.1fs",
            total, elapsed,
        )
        return results

    # ------------------------------------------------------------------ ddl
    def _ensure_schema(self) -> None:
        if not self.schema:
            return
        with self.engine.begin() as conn:
            conn.execute(text(
                f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'
            ))
            conn.execute(text(
                f'SET search_path TO "{self.schema}", public'
            ))
        logger.info("RRFImporter | Schema '%s' sichergestellt", self.schema)

    def _ensure_tables(self) -> None:
        """Legt Tabellen via ``UmlsBase.metadata`` an (idempotent)."""
        # search_path setzen
        if self.schema:
            self.engine.dispose()
            with self.engine.begin() as conn:
                conn.execute(text(
                    f'SET search_path TO "{self.schema}", public'
                ))

        # Wir erzeugen die Tabellen via "checkfirst", damit bestehende
        # nicht zerstört werden.
        for tbl in (MrConso.__table__, MrSty.__table__,
                    MrRel.__table__, MrDef.__table__):
            tbl.create(self.engine, checkfirst=True)
            # UMLS field lengths vary by source; upgrade older importer tables
            # that used guessed VARCHAR limits without losing their data.
            with self.engine.begin() as conn:
                for column in tbl.columns:
                    conn.execute(text(
                        f'ALTER TABLE "{tbl.name}" '
                        f'ALTER COLUMN "{column.name}" TYPE TEXT'
                    ))
        logger.info("RRFImporter | Tabellen sichergestellt")

    def _truncate_all(self, only: Iterable[str] | None = None) -> None:
        """Leert die UMLS-Tabellen (TRUNCATE … RESTART IDENTITY).

        ``only``: optionale Teilmenge der Tabellen-Präfixe (z.B. ``MRREL``).
        Default ``None`` = alle vier.
        """
        all_tables = {"MRCONSO": "mrconso", "MRSTY": "mrsty",
                      "MRREL": "mrrel", "MRDEF": "mrdef"}
        if only is None:
            wanted = list(all_tables.values())
        else:
            wanted = [all_tables[k] for k in (t.upper() for t in only)
                      if k in all_tables]
        with self.engine.begin() as conn:
            for tbl in wanted:
                conn.execute(text(f'TRUNCATE TABLE "{tbl}"'))
        logger.info("RRFImporter | Tabellen geleert (TRUNCATE): %s",
                    ", ".join(wanted) or "<keine>")

    def _create_indexes(self) -> None:
        """Legt zusätzliche Indizes für typische Such-Pfade an."""
        # pg_trgm GIN auf mrconso.STR für Fuzzy-Suche
        statements = [
            # Trigram-Index nur anlegen, wenn pg_trgm verfügbar
            "CREATE EXTENSION IF NOT EXISTS pg_trgm",
            ("CREATE INDEX IF NOT EXISTS ix_mrconso_str_trgm "
             "ON mrconso USING GIN (STR gin_trgm_ops)"),
            ("CREATE INDEX IF NOT EXISTS ix_mrconso_lat_sab "
             "ON mrconso (LAT, SAB)"),
            ("CREATE INDEX IF NOT EXISTS ix_mrconso_code_lower "
             "ON mrconso (LOWER(CODE))"),
            ("CREATE INDEX IF NOT EXISTS ix_mrrel_cui1_rel "
             "ON mrrel (CUI1, REL)"),
            ("CREATE INDEX IF NOT EXISTS ix_mrrel_cui2_rel "
             "ON mrrel (CUI2, REL)"),
            ("CREATE INDEX IF NOT EXISTS ix_mrsty_tui "
             "ON mrsty (TUI)"),
            ("CREATE INDEX IF NOT EXISTS ix_mrdef_sab "
             "ON mrdef (SAB)"),
        ]
        # Jeder Index in SEINER EIGENEN Transaktion: eine Abfrage (z.B.
        # fehlende pg_trgm-Extension oder bereits bestehende Index) darf
        # den Rest nicht mitreissen – ein einzelner Fehler in einer
        # gemeinsamen Transaktion würde alle nachfolgenden Abfragen
        # abbrechen ("current transaction is aborted").
        for stmt in statements:
            try:
                with self.engine.begin() as conn:
                    conn.execute(text(stmt))
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "RRFImporter | Index/Extension '%s' fehlgeschlagen: %s",
                    stmt[:60], e,
                )
        logger.info("RRFImporter | Indizes angelegt")

    # ------------------------------------------------------------------ bulk
    def _bulk_insert(self, model, rows: Iterable[tuple], cols: list[str]) -> int:
        """Führt einen Bulk-Insert aus.

        Strategie:
        1. ``psycopg2.copy_expert`` via ``engine.raw_connection`` (schnellste).
        2. Fallback ``executemany`` in Batches via SQLAlchemy ``connection``.

        Wir nutzen ``TAB`` als Trenner (statt Pipe), weil Tabulatoren in
        UMLS-Strings unüblich sind und Postgres ``COPY`` ohne Escape-
        Logik für Pipe bräuchte. Wir parsen die Pipe-Strings vorab zu
        TSV.

        ``cols`` ist die explizite Spaltenliste in RRF-Reihenfolge (z.B.
        ``MRREL_COLS``). Das ist wichtig, weil die SQLAlchemy-Modell-
        Spaltenreihenfolge (primärkey zuerst) NICHT mit der RRF-Reihenfolge
        übereinstimmte – ``list(model.__table__.columns)`` würde z.B. bei
        ``mrrel`` ``RUI`` ans Ende schieben und jede Zeile ab Position 9
        um ein Feld verschieben (Ursache der leeren ``mrrel``-Tabelle).
        """
        rows = list(rows)
        total = len(rows)
        if total == 0:
            return 0

        tbl_name = model.__table__.name

        # Versuch 1: psycopg2 copy_expert
        copied = self._try_copy(tbl_name, cols, rows)
        if copied:
            return copied

        # Fallback: executemany
        return self._executemany_batches(model, rows, cols)

    def _try_copy(self, tbl_name: str, cols: list[str],
                  rows: list[tuple]) -> int:
        """Versucht einen schnellen ``COPY FROM STDIN``-Import."""
        try:
            import psycopg2  # type: ignore
        except ImportError:
            logger.debug("RRFImporter | psycopg2 nicht verfügbar – Fallback")
            return 0

        try:
            raw = self.engine.raw_connection()
        except Exception as e:
            logger.debug("RRFImporter | raw_connection fehlgeschlagen: %s", e)
            return 0

        copied = 0
        try:
            cur = raw.cursor()
            if self.schema:
                cur.execute(f'SET search_path TO "{self.schema}", public')

            buf = self._make_tsv_buffer(rows, len(cols))
            # SQLAlchemy created the uppercase MRCONSO identifiers quoted;
            # quote every identifier here so COPY addresses the same columns.
            col_csv = ",".join(f'"{col}"' for col in cols)
            quoted_table = f'"{tbl_name}"'
            cur.copy_expert(
                f"COPY {quoted_table} ({col_csv}) "
                f"FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', "
                f"NULL '', QUOTE E'\\b', ESCAPE E'\\b')",
                buf,
            )
            copied = len(rows)
            raw.commit()
            return copied
        except Exception as e:
            logger.warning(
                "RRFImporter | COPY fehlgeschlagen (%s) – Fallback auf executemany",
                e,
            )
            try:
                raw.rollback()
            except Exception:
                pass
            return 0
        finally:
            try:
                raw.close()
            except Exception:
                pass

    def _make_tsv_buffer(self, rows: list[tuple], n_cols: int):
        """Erzeugt einen Datei-ähnlichen Buffer mit TAB-getrennten Zeilen.

        Quotes und Newlines werden neutralisiert, damit ``COPY`` mit
        ``QUOTE '\\b'`` problemlos durchläuft.
        """
        import io

        class _TSVBuf(io.IOBase):
            def __init__(self, data):
                self._data = data
                self._pos = 0

            def read(self, size=-1):
                if size < 0:
                    size = len(self._data) - self._pos
                chunk = self._data[self._pos:self._pos + size]
                self._pos += size
                return chunk

        pieces: list[bytes] = []
        for row in rows:
            row = tuple(_coerce(v) for v in row)
            # Newlines und Tabs aus Inhalten entfernen, damit COPY nicht
            # durcheinanderkommt (UMLS-Strings sind 1-zeilig).
            cleaned = [str(v).replace("\t", " ").replace("\n", " ") for v in row]
            pieces.append(("\t".join(cleaned) + "\n").encode("utf-8"))
        return _TSVBuf(b"".join(pieces))

    def _executemany_batches(self, model, rows: list[tuple],
                             cols: list[str]) -> int:
        """Fallback: executemany in Batches via SQLAlchemy ``connection``."""
        from sqlalchemy import insert

        rows_dicts = [
            dict(zip(cols, (_coerce(v) for v in r))) for r in rows
        ]
        total = 0
        with self.engine.begin() as conn:
            for i in range(0, len(rows_dicts), self.batch_size):
                batch = rows_dicts[i:i + self.batch_size]
                conn.execute(insert(model), batch)
                total += len(batch)
                if total % (self.batch_size * 5) == 0:
                    logger.info(
                        "RRFImporter | Fortschritt: %d Zeilen geschrieben",
                        total,
                    )
        return total

    # ------------------------------------------------------------------ pro-datei
    def _import_conso(self, path: Path) -> int:
        logger.info("RRFImporter | Importiere %s …", path)
        count = 0
        batch: list[tuple] = []
        for row in _iter_rrf(path, len(MRCONSO_COLS)):
            batch.append(row)
            if len(batch) >= self.batch_size:
                count += self._bulk_insert(MrConso, batch, MRCONSO_COLS)
                batch.clear()
                if count % (self.batch_size * 5) == 0:
                    logger.info("… %d MRCONSO-Zeilen", count)
        if batch:
            count += self._bulk_insert(MrConso, batch, MRCONSO_COLS)
        logger.info("RRFImporter | MRCONSO: %d Zeilen importiert", count)
        self._stats["mrconso"] = {"rows": count, "duration_s": None}
        return count

    def _import_sty(self, path: Path) -> int:
        logger.info("RRFImporter | Importiere %s …", path)
        batch: list[tuple] = []
        count = 0
        for row in _iter_rrf(path, len(MRSTY_COLS)):
            batch.append(row)
            if len(batch) >= self.batch_size:
                count += self._bulk_insert(MrSty, batch, MRSTY_COLS)
                batch.clear()
        if batch:
            count += self._bulk_insert(MrSty, batch, MRSTY_COLS)
        logger.info("RRFImporter | MRSTY: %d Zeilen importiert", count)
        return count

    def _import_rel(self, path: Path) -> int:
        logger.info("RRFImporter | Importiere %s …", path)
        batch: list[tuple] = []
        count = 0
        for row in _iter_rrf(path, len(MRREL_COLS)):
            batch.append(row)
            if len(batch) >= self.batch_size:
                count += self._bulk_insert(MrRel, batch, MRREL_COLS)
                batch.clear()
                if count % (self.batch_size * 5) == 0:
                    logger.info("… %d MRREL-Zeilen", count)
        if batch:
            count += self._bulk_insert(MrRel, batch, MRREL_COLS)
        logger.info("RRFImporter | MRREL: %d Zeilen importiert", count)
        return count

    def _import_def(self, path: Path) -> int:
        logger.info("RRFImporter | Importiere %s …", path)
        batch: list[tuple] = []
        count = 0
        for row in _iter_rrf(path, len(MRDEF_COLS)):
            batch.append(row)
            if len(batch) >= self.batch_size:
                count += self._bulk_insert(MrDef, batch, MRDEF_COLS)
                batch.clear()
        if batch:
            count += self._bulk_insert(MrDef, batch, MRDEF_COLS)
        logger.info("RRFImporter | MRDEF: %d Zeilen importiert", count)
        return count


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-Importer für UMLS RRF-Dateien in Postgres.",
    )
    parser.add_argument(
        "--meta-dir", required=True,
        help="Pfad zum META-Ordner (enthält MRCONSO.RRF, MRSTY.RRF, "
             "MRREL.RRF, MRDEF.RRF).",
    )
    parser.add_argument(
        "--tables", default="",
        help="Teilmenge zu importieren (Default: alle). "
             "Beispiel: --tables MRREL,MRDEF  (nur Relationen & Definitionen, "
             "mrconso/mrsty bleiben unangetastet)",
    )
    parser.add_argument(
        "--database-url", default=os.getenv("UMLS_LOCAL_DB_URL", ""),
        help="Postgres-URL (default: $UMLS_LOCAL_DB_URL oder Settings.database_url).",
    )
    parser.add_argument(
        "--schema", default="",
        help="Optionaler Postgres-Schemaname für die UMLS-Tabellen.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50_000,
        help="Anzahl Zeilen pro Batch (Default: 50000).",
    )
    parser.add_argument(
        "--truncate", action=argparse.BooleanOptionalAction, default=True,
        help="Ziel-Tabelle(n) vor dem Import TRUNCATE (Default: ja; "
             "--no-truncate: anhängender Import)",
    )
    parser.add_argument(
        "--no-index", action="store_true",
        help="Zusatzindizes nach dem Import nicht anlegen.",
    )
    args = parser.parse_args(argv)

    tables_only: list[str] | None = None
    if args.tables.strip():
        tables_only = [t.strip().upper() for t in args.tables.split(",")
                       if t.strip()]

    # Logging aus logging_config wenn verfügbar
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    if not args.database_url:
        try:
            from src.config import Settings
            args.database_url = Settings().database_url
        except Exception as e:
            parser.error(
                f"--database-url oder $UMLS_LOCAL_DB_URL nötig: {e}"
            )

    importer = RRFImporter(
        database_url=args.database_url,
        schema=args.schema or None,
        batch_size=args.batch_size,
        truncate=args.truncate,
        create_indexes=not args.no_index,
    )
    results = importer.import_directory(args.meta_dir, only=tables_only)
    print("Import-Ergebnis:")
    for name, n in results.items():
        print(f"  {name}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import (create_engine, Column, Integer, BigInteger, SMALLINT,
                        Text, Boolean, String, Numeric, REAL, UniqueConstraint,
                        ForeignKey, DateTime, inspect, text)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, insert as pg_insert
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import unicodedata


Base = declarative_base()

logger = logging.getLogger(__name__)


# ---------- Existing report data ----------

class CaseReport(Base):
    __tablename__ = "case_report"

    report_id = Column(BigInteger, primary_key=True)
    case_id_alias = Column(Text, unique=True, nullable=False)
    original_case_id = Column(Text)
    anonym_case_id = Column(Text)
    report_type = Column(Text)
    parent_folder = Column(Text)
    filename = Column(Text)
    report_year = Column(SMALLINT)
    diagnose = Column(Text)
    grading = Column(Text)
    organ_primary = Column(Text)
    organ_normalized = Column(ARRAY(Text))
    gewebsart = Column(ARRAY(Text))
    probenart = Column(Text)
    analysemethode = Column(ARRAY(Text))
    faerbungen = Column(ARRAY(Text))
    sonstige_keywords = Column(ARRAY(Text))
    raw_text = Column(Text)
    search_vector = Column(TSVECTOR)
    # embedding = Column(Vector(...))  # pgvector, optional via DDL
    created_at = Column(DateTime(timezone=True))
    processed_at = Column(DateTime(timezone=True), nullable=True,
                          doc="Zeitpunkt der letzten erfolgreichen Pipeline-Verarbeitung")

    report_concepts = relationship("ReportConcept", back_populates="report")
    case_slides = relationship("SlideDescription", back_populates="slide")


# ---------- Slides ----------

class SlideDescription(Base):
    __tablename__ = "slide_description"

    description_id = Column(BigInteger, primary_key=True)
    report_id = Column(BigInteger, ForeignKey("case_report.report_id"), nullable=False)
    slide_id = Column(BigInteger, nullable=False, unique=True)
    wsi_uuid = Column(ARRAY(Text))       # Liste von UUIDs der digitalen Scans
    digital_wsi = Column(Boolean, default=False)

    description = Column(Text)
    staining = Column(Text)
    organ = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    slide = relationship("CaseReport", back_populates="case_slides")


# ---------- Medical terminology ----------

class MedicalConcept(Base):
    __tablename__ = "medical_concept"
    __table_args__ = (UniqueConstraint("terminology", "concept_code"),)

    concept_id = Column(BigInteger, primary_key=True)
    terminology = Column(Text, nullable=False)
    sab = Column(Text)                   # Source Abbreviation, z.B. SNOMEDCT-DE, LNC
    concept_code = Column(Text, nullable=False)
    preferred_term = Column(Text, nullable=False)
    semantic_type = Column(Text)
    finding = Column(Text)               # extrahiertes Label, z.B. TUMOR_ENTITY
    terminology_version = Column(Text)
    active = Column(Boolean, default=True)

    terms = relationship("ConceptTerm", back_populates="concept")


class ConceptTerm(Base):
    __tablename__ = "concept_term"
    __table_args__ = (UniqueConstraint("concept_id", "normalized_term"),)

    term_id = Column(BigInteger, primary_key=True)
    concept_id = Column(BigInteger, ForeignKey("medical_concept.concept_id"),
                        nullable=False)
    term = Column(Text, nullable=False)
    normalized_term = Column(Text, nullable=False)
    language = Column(String(10))
    term_type = Column(Text)      # "FSN" | "SYNONYM" | "DEFINITION"
    source = Column(Text)         # z.B. "snowstorm", "gazetteer"
    active = Column(Boolean, default=True)

    concept = relationship("MedicalConcept", back_populates="terms")


class ConceptRelation(Base):
    __tablename__ = "concept_relation"
    __table_args__ = (UniqueConstraint("source_concept_id",
                                       "target_concept_id", "relation_type"),)

    relation_id = Column(BigInteger, primary_key=True)
    source_concept_id = Column(BigInteger,
                               ForeignKey("medical_concept.concept_id"),
                               nullable=False)
    target_concept_id = Column(BigInteger,
                               ForeignKey("medical_concept.concept_id"),
                               nullable=False)
    relation_type = Column(Text, nullable=False)   # z.B. "is_a"
    distance = Column(Integer)
    terminology_version = Column(Text)


# ---------- Searchable medical annotations ----------

class ReportConcept(Base):
    __tablename__ = "report_concept"

    report_concept_id = Column(BigInteger, primary_key=True)
    report_id = Column(BigInteger, ForeignKey("case_report.report_id"),
                       nullable=False)
    concept_id = Column(BigInteger, ForeignKey("medical_concept.concept_id"),
                        nullable=True)   # NULL = nicht gemappt
    source_field = Column(Text)          # z.B. "raw_text", "diagnose"
    surface_text = Column(Text)
    raw_text = Column(Text)              # deutscher Oberflächentext
    search_term = Column(Text)           # englischer Such-/Normalisierungsbegriff
    ontology = Column(Text)              # verwendeter Provider: snowstorm | umls
    start_offset = Column(Integer)
    end_offset = Column(Integer)
    assertion = Column(Text)
    certainty = Column(Text)
    temporality = Column(Text)
    laterality = Column(Text)
    confidence = Column(REAL)

    report = relationship("CaseReport", back_populates="report_concepts")


def normalize_term(term: str) -> str:
    t = unicodedata.normalize("NFKC", term).strip().lower()
    return " ".join(t.split())


class Repository:
    """Kapselt alle DB-Zugriffe (Upserts mit In-Memory-Cache pro Lauf)."""

    def __init__(self, database_url: str, pool_size: int = 5,
                 max_overflow: int = 10, pool_timeout: int = 30,
                 auto_create_schema: bool = True):
        self.engine = create_engine(database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._concept_cache: dict[tuple[str, str], int] = {}
        print(f"Verbindung zur Datenbank hergestellt: {database_url}")

        if auto_create_schema:
            self.init_schema()
            print("Leere Datenbank mit Schemas strukturiert")


    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def init_schema(self) -> None:
        """Legt alle definierten Tabellen an, sofern sie noch nicht existieren.

        Idempotent: Bestehende Tabellen werden nicht verändert oder
        überschrieben (checkfirst=True). Achtung: create_all führt KEINE
        Migrationen durch – nachträglich geänderte Spalten an existierenden
        Tabellen werden nicht angepasst.
        """
        before = set(inspect(self.engine).get_table_names())
        Base.metadata.create_all(self.engine, checkfirst=True)
        after = set(inspect(self.engine).get_table_names())

        created = sorted(after - before)
        if created:
            logger.info("Neue Tabellen angelegt: %s", ", ".join(created))
        else:
            logger.debug("Datenbankschema bereits vollständig vorhanden.")

    def create_missing_tables(self):
        """Legt nur die neuen Tabellen an (case_report existiert bereits)."""
        for tbl in (SlideDescription, MedicalConcept,
                    ConceptTerm, ConceptRelation, ReportConcept):
            tbl.__table__.create(self.engine, checkfirst=True)

    def ensure_processed_at_column(self) -> None:
        """Fügt processed_at zu case_report hinzu, falls die Spalte fehlt.

        SQLAlchemy create_all aktualisiert bestehende Tabellen nicht.
        Diese Methode führt ein idempotentes ALTER TABLE aus.
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'case_report' AND column_name = 'processed_at'"
            ))
            if not result.fetchone():
                conn.execute(text(
                    "ALTER TABLE case_report ADD COLUMN processed_at TIMESTAMP WITH TIME ZONE"
                ))
                conn.commit()
                logger.info("Spalte processed_at zu case_report hinzugefügt")
            else:
                logger.debug("Spalte processed_at bereits vorhanden")

    # ----- Reports -----
    def get_reports(self, limit: Optional[int] = None,
                    processed: Optional[bool] = False) -> list[CaseReport]:
        """Gibt Reports zurück.

        Args:
            limit: Maximale Anzahl Reports.
            processed: Wenn True, nur bereits verarbeitete; wenn False, nur
                unverarbeitete (processed_at IS NULL); wenn None, alle.
        """
        with self.Session() as s:
            q = s.query(CaseReport).filter(CaseReport.raw_text.isnot(None))
            if processed is True:
                q = q.filter(CaseReport.processed_at.isnot(None))
                logger.info("Lade nur verarbeitete Berichte aus Datenbank")
            elif processed is False:
                q = q.filter(CaseReport.processed_at.is_(None))
                logger.info("Lade nur unverarbeitete Berichte aus Datenbank")
            else:
                logger.info("Lade alle Berichte aus Datenbank")     
            if limit:
                q = q.limit(limit)
            return q.all()

    def mark_report_processed(self, report_id: int) -> None:
        """Setzt processed_at auf den aktuellen Zeitpunkt."""
        with self.Session() as s:
            report = s.query(CaseReport).filter_by(report_id=report_id).first()
            if report:
                report.processed_at = datetime.now(timezone.utc)
                s.commit()
                logger.debug(f"Bericht {report_id} als verarbeitet markiert")

    def upsert_concept(self, terminology: str, concept_code: str,
                       preferred_term: str, semantic_type: Optional[str],
                       terminology_version: Optional[str],
                       session,
                       sab: Optional[str] = None,
                       finding: Optional[str] = None) -> int:
        key = (terminology, concept_code)
        if key in self._concept_cache:
            return self._concept_cache[key]

        existing = session.query(MedicalConcept).filter_by(
            terminology=terminology, concept_code=concept_code).first()
        if existing:
            self._concept_cache[key] = existing.concept_id
            return existing.concept_id

        concept = MedicalConcept(
            terminology=terminology, sab=sab, concept_code=concept_code,
            preferred_term=preferred_term, semantic_type=semantic_type,
            finding=finding, terminology_version=terminology_version,
            active=True)
        session.add(concept)
        session.flush()   # concept_id vergeben
        self._concept_cache[key] = concept.concept_id
        return concept.concept_id

    def upsert_term(self, concept_id: int, term: str, language: str,
                    term_type: str, source: str, session) -> None:
        norm = normalize_term(term)
        statement = pg_insert(ConceptTerm).values(
            concept_id=concept_id, term=term, normalized_term=norm,
            language=language, term_type=term_type, source=source,
            active=True,
        ).on_conflict_do_nothing(
            index_elements=["concept_id", "normalized_term"]
        )
        session.execute(statement)

    def upsert_relation(self, source_id: int, target_id: int,
                        relation_type: str, distance: int,
                        version: Optional[str], session) -> None:
        statement = pg_insert(ConceptRelation).values(
            source_concept_id=source_id, target_concept_id=target_id,
            relation_type=relation_type, distance=distance,
            terminology_version=version,
        ).on_conflict_do_nothing(
            index_elements=["source_concept_id", "target_concept_id", "relation_type"]
        )
        session.execute(statement)

    # ----- Annotationen -----
    def save_report_concepts(self, report_id: int, rows: list[dict]) -> None:
        with self.Session() as s:
            # Idempotenz: alte Annotationen des Reports löschen
            s.query(ReportConcept).filter_by(report_id=report_id).delete()
            for r in rows:
                s.add(ReportConcept(report_id=report_id, **r))
            s.commit()
"""SQLite-backed stand-in for ``src.database.Repository``.

Used by the ``experiments/`` scripts when running against the synthetic test
set, where no PostgreSQL clinical archive is available. It exposes the subset
of the ``Repository`` API the extraction experiments rely on:

- ``session_scope()`` (context manager yielding a SQLAlchemy session)
- ``save_report_concepts(report_id, rows)`` (persist extracted spans)
- ``upsert_concept / upsert_term / upsert_relation`` (ontology concepts)
- ``create_missing_tables()`` (no-op) / ``ensure_processed_at_column()`` (no-op)

It is intentionally minimal: enough to let ``NERPipeline`` and the ontology
``OntologyMapper`` complete a full run and write their results somewhere
inspectable, **without** pulling in Postgres-specific column types
(``ARRAY(Text)``, ``TSVECTOR``) that exist only in the production schema.

The underlying tables are created in a SQLite file (see
``experiments/make_sqlite_sample_db.py``) that already contains the synthetic
``case_report`` rows and a ``report_concept`` table.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, Text, REAL, BigInteger, \
    ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()


class CaseReport(Base):
    __tablename__ = "case_report"
    report_id = Column(BigInteger, primary_key=True)
    anonym_case_id = Column(Text)
    dateiname = Column(Text)
    berichtstyp = Column(Text)
    jahr = Column(Integer)
    raw_text = Column(Text)
    is_synthetic = Column(Integer)
    created_at = Column(Text)


class MedicalConcept(Base):
    __tablename__ = "medical_concept"
    __table_args__ = (UniqueConstraint("terminology", "concept_code"),)
    concept_id = Column(BigInteger, primary_key=True)
    terminology = Column(Text, nullable=False)
    sab = Column(Text)
    concept_code = Column(Text, nullable=False)
    preferred_term = Column(Text, nullable=False)
    semantic_type = Column(Text)
    finding = Column(Text)
    terminology_version = Column(Text)
    active = Column(Integer, default=1)


class ConceptTerm(Base):
    __tablename__ = "concept_term"
    __table_args__ = (UniqueConstraint("concept_id", "normalized_term"),)
    term_id = Column(BigInteger, primary_key=True)
    concept_id = Column(BigInteger, ForeignKey("medical_concept.concept_id"),
                        nullable=False)
    term = Column(Text, nullable=False)
    normalized_term = Column(Text, nullable=False)
    language = Column(Text)
    term_type = Column(Text)
    source = Column(Text)
    active = Column(Integer, default=1)


class ConceptRelation(Base):
    __tablename__ = "concept_relation"
    __table_args__ = (UniqueConstraint("source_concept_id",
                                       "target_concept_id", "relation_type"),)
    relation_id = Column(BigInteger, primary_key=True)
    source_concept_id = Column(BigInteger, ForeignKey("medical_concept.concept_id"),
                               nullable=False)
    target_concept_id = Column(BigInteger, ForeignKey("medical_concept.concept_id"),
                               nullable=False)
    relation_type = Column(Text, nullable=False)
    distance = Column(Integer)
    terminology_version = Column(Text)


class ReportConcept(Base):
    __tablename__ = "report_concept"
    annotation_id = Column(BigInteger, primary_key=True)
    report_id = Column(BigInteger, ForeignKey("case_report.report_id"),
                       nullable=False)
    concept_id = Column(BigInteger, ForeignKey("medical_concept.concept_id"),
                        nullable=True)
    source_field = Column(Text)
    surface_text = Column(Text)
    raw_text = Column(Text)
    search_term = Column(Text)
    ontology = Column(Text)
    start_offset = Column(Integer)
    end_offset = Column(Integer)
    assertion = Column(Text)
    certainty = Column(Text)
    temporality = Column(Text)
    laterality = Column(Text)
    confidence = Column(REAL)


class SampleStore:
    """A minimal, SQLite-backed replacement for ``src.database.Repository``."""

    def __init__(self, database_url: str):
        if not database_url.startswith("sqlite"):
            raise ValueError("SampleStore only supports sqlite:// URLs")
        # SQLite doesn't support pool_size / max_overflow the same way — strip
        # them off any SQLAlchemy engine kwargs the existing callers pass.
        from sqlalchemy.pool import StaticPool
        self.engine = create_engine(database_url, future=True,
                                    poolclass=StaticPool)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._concept_cache: dict[tuple[str, str], int] = {}
        Base.metadata.create_all(self.engine, checkfirst=True)
        print(f"SampleStore (SQLite) bound to: {database_url}")

    # ------------------------------------------------------------------ API
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

    def create_missing_tables(self) -> None:
        """No-op for the sample store (tables are created in __init__)."""

    def ensure_processed_at_column(self) -> None:
        """No-op for the sample store (no processed_at column)."""

    # ---- Ontology / concept persistence (minimal, no Postgres upserts) ----
    def upsert_concept(self, terminology: str, concept_code: str,
                       preferred_term: str, semantic_type: Optional[str],
                       terminology_version: Optional[str], session,
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
            active=1,
        )
        session.add(concept)
        session.flush()
        self._concept_cache[key] = concept.concept_id
        return concept.concept_id

    def upsert_term(self, concept_id: int, term: str, language: str,
                    term_type: str, source: str, session) -> None:
        from src.database import normalize_term
        norm = normalize_term(term)
        existing = session.query(ConceptTerm).filter_by(
            concept_id=concept_id, normalized_term=norm).first()
        if existing:
            return
        session.add(ConceptTerm(concept_id=concept_id, term=term,
                                normalized_term=norm, language=language,
                                term_type=term_type, source=source, active=1))

    def upsert_relation(self, source_id: int, target_id: int,
                        relation_type: str, distance: int,
                        version: Optional[str], session) -> None:
        existing = session.query(ConceptRelation).filter_by(
            source_concept_id=source_id, target_concept_id=target_id,
            relation_type=relation_type).first()
        if existing:
            return
        session.add(ConceptRelation(source_concept_id=source_id,
                                    target_concept_id=target_id,
                                    relation_type=relation_type,
                                    distance=distance,
                                    terminology_version=version))

    def save_report_concepts(self, report_id: int, rows: list[dict]) -> None:
        with self.Session() as s:
            s.query(ReportConcept).filter_by(report_id=report_id).delete()
            for r in rows:
                s.add(ReportConcept(report_id=report_id, **r))
            s.commit()

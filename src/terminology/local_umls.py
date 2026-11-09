"""Lokaler Drop-in-Ersatz für ``UMLSClient`` (umls.py).

Verwendet SQLAlchemy + Postgres und liest die per
``umls_rrf_importer.RRFImporter`` befüllten Tabellen
``mrconso``, ``mrsty``, ``mrrel`` und ``mrdef``.

Die öffentliche Schnittstelle (Methodensignaturen, Rückgabewerte)
ist 1:1 kompatibel zur bestehenden ``UMLSClient``-Klasse, so dass
``mapper.py`` und ``pipeline.py`` unverändert weiterbenutzt werden
können.

Konventionen (analog zu ``umls.py``, ``database.py``):
- Logger via ``logging.getLogger(__name__)``
- Type Hints (PEP 604 ``X | None``)
- Deutsche Docstrings
- ``session_scope``-Pattern aus ``database.Repository``
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

logger = logging.getLogger(__name__)


# Mögliche "Relation Labels" (REL) im UMLS Metathesaurus.
# RN  = narrower than  -> CUI2 ist Oberbegriff (Vorfahre) von CUI1
# RB  = broader than   -> CUI2 ist Unterbegriff
# PAR = has parent     (Quell-Vokabular-spezifisch)
# CHD = has child
# RO  = other relationship
DEFAULT_RELATION_LABELS_ANCESTOR = ("RN", "PAR")


class LocalUMLSClient:
    """Thread-safe lokaler UMLS-Lookup gegen eine Postgres-Datenbank.

    Drop-in-Ersatz für :class:`src.terminology.umls.UMLSClient``.
    Die Methodensignaturen ``__init__``, ``close``, ``__enter__``,
    ``__exit__``, ``search``, ``get_concept`` und ``get_ancestors``
    sind 1:1 kompatibel.
    """

    def __init__(self, api_key: str, version: str = "current",
                 base_url: str = "https://uts-ws.nlm.nih.gov/rest",
                 page_size: int = 25, timeout: int = 30,
                 pool_maxsize: int = 32,
                 *,
                 database_url: str | None = None,
                 repository: Any | None = None,
                 schema: str | None = None,
                 language: str = "ENG"):
        """Initialisiert den lokalen Client.

        Args:
            api_key: Wird aus Kompatibilitätsgründen akzeptiert (bei
                ``database_url``/``repository`` ungenutzt), so dass der
                bestehende ``main.py``-Pfad ohne Änderung weiterläuft.
            version: Terminologie-Version (z. B. ``"current"``,
                ``"2024AA"``). Wird als ``terminology_version``
                zurückgegeben, aber nicht zwingend in den Queries
                gefiltert (die Tabellen enthalten mehrere Versionen).
            base_url: Wird aus Kompatibilitätsgründen ignoriert.
            page_size: Default-Limit für :meth:`search`.
            timeout: Aus Kompatibilität, ungenutzt.
            pool_maxsize: Aus Kompatibilität, ungenutzt.
            database_url: Optionale Postgres-URL. Wenn weder
                ``database_url`` noch ``repository`` gegeben sind,
                wird ``Settings().database_url`` verwendet.
            repository: Optionale :class:`database.Repository`-
                Instanz (bevorzugt, vermeidet Doppel-Engine).
            schema: Optionaler Postgres-Schemaname für die UMLS-Tabellen
                (Default ``None`` = public). Wird via
                ``search_path`` aktiviert, so dass kein Schema-Prefix
                nötig ist.
            language: Sprachfilter für ``mrconso.LAT`` (Default
                ``"ENG"``).
        """
        # Kompat-Felder (aus UMLSClient übernommen)
        self.api_key = api_key
        self.version = version
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.timeout = timeout
        self.pool_maxsize = pool_maxsize

        # Lokale Konfiguration
        self.schema = schema
        self.language = language

        # TGT-Platzhalter nur für API-Kompatibilität (ungenutzt)
        self._tgt: str | None = None
        self._tgt_lock = threading.Lock()
        self._tgt_generation = 0

        # Repository / Engine
        if repository is not None:
            self._repo = repository
            self._owns_repo = False
        else:
            try:
                from src.database import Repository
            except ImportError:
                from database import Repository  # pragma: no cover
            url = database_url or self._default_database_url()
            self._repo = Repository(url, auto_create_schema=False)
            self._owns_repo = True

        # Cache: search-Term -> Liste[CUI]
        self._search_cache: dict[tuple, list[dict]] = {}
        self._concept_cache: dict[str, dict] = {}
        self._ancestor_cache: dict[tuple[str, tuple[str, ...], str, int], list[dict]] = {}
        self._ancestor_graph_cache: dict[tuple[str, tuple[str, ...], int], list[dict]] = {}
        self._search_cache_lock = threading.Lock()

        # Ohne MRREL liefert get_ancestors() fuer jeden CUI eine leere
        # Liste -- ohne Fehler, ohne Logeintrag. Ein 60-Stunden-Lauf
        # befuellt dann konsequent 'concept_relation' mit 0 Zeilen und
        # niemand bemerkt es. Deshalb einmalig beim Start pruefen.
        self._relation_tables_checked = False

        logger.info(
            "LocalUMLSClient | initialisiert (schema=%s, language=%s)",
            self.schema or "<public>", self.language,
        )

    def _check_relation_tables(self) -> None:
        """Warnt einmalig, wenn MRREL/MRDEF die Hierarchieabfrage blockieren.

        MRREL (Relationen) und MRDEF (Definitionen) sind die einzigen
        UMLS-Tabellen, die fuer ``get_ancestors`` noetig sind. Sind sie
        leer oder gar nicht vorhanden, ist der lokale Client faktisch ein
        reiner String-zu-CUI-Lookup ohne jede Ontologie-Hierarchie.
        """
        if self._relation_tables_checked:
            return
        self._relation_tables_checked = True

        try:
            # Kein count(*): MRREL hat im vollstaendigen Metathesaurus
            # zweistellige Millionen Zeilen. Ein Existenztest reicht.
            with self._connect() as conn:
                counts = {
                    table: bool(conn.execute(
                        text(f'SELECT 1 FROM "{table}" LIMIT 1')
                    ).fetchone())
                    for table in ("mrrel", "mrdef")
                }
        except Exception as e:  # pragma: no cover - defensiv
            logger.warning(
                "LocalUMLSClient | Relationstabellen nicht pruefbar (%s) - "
                "Vorfahren-/Hierarchieabfragen werden fehlschlagen.", e,
            )
            return

        if not counts.get("mrrel"):
            logger.error(
                "LocalUMLSClient | MRREL ist LEER. get_ancestors() liefert fuer "
                "jeden CUI [] und 'concept_relation' bleibt vollstaendig "
                "unbefuellt. Vorfahren zuerst importieren: "
                "python3 scripts/umls_rrf_importer.py --meta-dir <UMLS>/META "
                "(MRREL.RRF, MRDEF.RRF)"
            )
        if not counts.get("mrdef"):
            logger.warning(
                "LocalUMLSClient | MRDEF ist LEER - Konzeptdefinitionen "
                "(semantic types, Definitionstexte) fehlen. Mapping und "
                "Namenauflösung funktionieren, Metadaten sind aber lueckenhaft."
            )

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Gibt Ressourcen frei (Cache, optional die Engine)."""
        with self._search_cache_lock:
            self._search_cache.clear()
            self._concept_cache.clear()
            self._ancestor_cache.clear()
            self._ancestor_graph_cache.clear()
        if self._owns_repo:
            try:
                self._repo.engine.dispose()
            except Exception:  # pragma: no cover - defensive
                logger.debug("LocalUMLSClient | Engine-Dispose fehlgeschlagen",
                             exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _default_database_url() -> str:
        """Liest die Default-DB-URL aus ``Settings`` (Fallback)."""
        try:
            try:
                from src.config import Settings
            except ImportError:
                from config import Settings  # pragma: no cover
            return Settings().database_url
        except Exception as e:  # pragma: no cover - Konfig-Fallback
            logger.warning(
                "LocalUMLSClient | Konnte Settings.database_url nicht laden: %s",
                e,
            )
            return os.getenv(
                "UMLS_LOCAL_DB_URL",
                "postgresql+psycopg2://postgres@localhost:5432/umls",
            )

    def _connect(self):
        """Stellt eine Connection her und setzt ggf. ``search_path``."""
        conn = self._repo.engine.connect()
        if self.schema:
            try:
                conn.execute(text(f'SET search_path TO "{self.schema}", public'))
            except (OperationalError, ProgrammingError):
                logger.warning(
                    "LocalUMLSClient | Schema '%s' nicht setzbar – fahre fort",
                    self.schema,
                )
        return conn

    def ensure_indexes(self) -> None:
        """Ergänzt Indizes für lokale Lookups, falls sie fehlen."""
        index_statements = (
            'CREATE EXTENSION IF NOT EXISTS pg_trgm',
            'CREATE INDEX IF NOT EXISTS ix_mrconso_str_trgm '
            'ON mrconso USING GIN ("STR" gin_trgm_ops)',
            'CREATE INDEX IF NOT EXISTS ix_mrconso_cui_lat '
            'ON mrconso ("CUI", "LAT")',
            'CREATE INDEX IF NOT EXISTS ix_mrsty_cui '
            'ON mrsty ("CUI")',
            'CREATE INDEX IF NOT EXISTS ix_mrdef_cui '
            'ON mrdef ("CUI")',
            'CREATE INDEX IF NOT EXISTS ix_mrrel_cui1_rel_sab '
            'ON mrrel ("CUI1", "REL", "SAB")',
        )
        with self._connect() as conn:
            for statement in index_statements:
                conn.execute(text(statement))
            conn.commit()

    @staticmethod
    def _normalize_term_for_search(term: str) -> str:
        """SQL-ILIKE-tauglicher Wildcard-Term."""
        # Escape '%' und '_' für LIKE/ILIKE
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    # ------------------------------------------------------------------ search
    def search(self, term: str, sabs: list[str] | None = None,
               search_type: str = "words", return_id_type: str = "concept",
               limit: int = 5) -> list[dict]:
        """Sucht nach Begriffen in UMLS (lokal).

        Args:
            term: Suchbegriff.
            sabs: Liste der bevorzugten Vokabulare (z.B. ``["SNOMEDCT_US"]``).
            search_type: ``"words"`` (default), ``"exact"``,
                ``"startswith"``, ``"normalizedWords"`` etc.
                - ``"words"`` / ``"normalizedWords"``: ILIKE-Teilstring
                - ``"exact"``: ILIKE-Gleichheit
                - ``"startswith"``: ILIKE-Präfix
            return_id_type: ``"concept"`` (default), ``"sourceConcept"``,
                ``"sourceDescriptor"``. Wird respektiert, indem
                ``ui`` = CUI (concept) bzw. AUI/CODE/SCUI zurückgegeben
                wird.
            limit: Maximale Anzahl Treffer.

        Returns:
            Liste von Dicts analog zur ``UMLSClient.search``-Antwort:
            ``{"ui": "...", "name": "...", "rootSource": "...",
              "uri": "...", "sab": "...", "code": "...",
              "preferredName": "...", "match": "words"}``.
        """
        if not term or not term.strip():
            return []

        cache_key = (term.lower(), tuple(sabs or ()),
                     search_type, return_id_type, limit)
        with self._search_cache_lock:
            if cache_key in self._search_cache:
                return self._search_cache[cache_key]

        # SQL-Filter vorbereiten
        sab_clause = ""
        params: dict[str, Any] = {"term": term, "language": self.language}

        # ILIKE-Muster nach search_type
        if search_type == "exact":
            like_pattern = term.replace("\\", "\\\\")
            like_clause = 'mrc."STR" = :term'
            params["term"] = like_pattern
        elif search_type == "startswith":
            like_pattern = (
                term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            ) + "%"
            like_clause = 'mrc."STR" ILIKE :like_pat'
            params["like_pat"] = like_pattern
        else:
            # "words", "normalizedWords", oder unbekannt -> Teilstring
            params["like_pat"] = self._normalize_term_for_search(term)
            like_clause = 'mrc."STR" ILIKE :like_pat ESCAPE \'\\\''

        # Konfigurierte SABs *bevorzugen*, nicht hart filtern.
        # UMLS hält dasselbe Konzept in mehreren Vokabeln; Pathologie-
        # Begriffe ("Papanicolaou Stains", "Giemsa", ...) sitzen häufig nur
        # in Vokabeln außerhalb der konfigurierten SABs (CHV, NCI, UMD, ...).
        # Ein harter ``SAB = ANY(...)``-Filter würde diese Konzepte
        # unsichtbar machen und concept_id leer lassen. Deshalb: präferierte
        # SABs werden in der Rangfolge zuerst genommen, sonst fällt man auf
        # beliebige Vokabeln zurück – ein SNOMEDCT_US-/MSH-Treffer schlägt
        # weiterhin einen NCI-Treffer (weil er vor ISPREF gelistet ist).
        if sabs:
            sab_rank = '(mrc."SAB" = ANY(:sabs)) DESC, \n'
            params["sabs"] = list(sabs)
        else:
            sab_rank = ""

        # Limit auf eine sinnvolle Obermenge (RANK bevorzugt ISPREF und
        # dann kürzere Strings).
        sql = text(f"""
                 SELECT mrc."CUI"        AS cui,
                     mrc."STR"        AS name,
                     mrc."SAB"        AS sab,
                     mrc."CODE"       AS code,
                     mrc."AUI"        AS aui,
                     mrc."SCUI"       AS scui,
                     mrc."TTY"        AS tty,
                     mrc."ISPREF"     AS ispref
             FROM mrconso mrc
             WHERE mrc."LAT" = :language
               AND {like_clause}
             ORDER BY {sab_rank}
                      (mrc."ISPREF" = 'Y') DESC,
                      LENGTH(mrc."STR") ASC,
                      mrc."CUI" ASC
             LIMIT :limit
        """)
        params["limit"] = max(1, int(limit))

        results: list[dict] = []
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()

                # Fallback: Token-AND für Mehrwort-Begriffe.
                # "hematoxylin eosin" trifft keinen STR-Teilstring, wohlaber
                # jede Wort-Bestandteile einzeln ("Hematoxylin and Eosin").
                # Nur bei "words"-Suche und > 1 Token, sonst bleibt die
                # Semantik unverändert.
                if not rows and search_type == "words":
                    toks = [t for t in term.split() if t.strip()]
                    if 1 < len(toks) <= 12:
                        like_conds = " AND ".join(
                            [f'mrc."STR" ILIKE :tok_{i} ESCAPE \'\\\''
                             for i in range(len(toks))])
                        tok_params = {f"tok_{i}": self._normalize_term_for_search(t)
                                     for i, t in enumerate(toks)}
                        if sabs:
                            tok_params["sabs"] = list(sabs)
                            tok_rank = '(mrc."SAB" = ANY(:sabs)) DESC, \n'
                        else:
                            tok_rank = ""
                        tok_sql = text(f"""
                            SELECT mrc."CUI" AS cui, mrc."STR" AS name,
                                   mrc."SAB" AS sab, mrc."CODE" AS code,
                                   mrc."AUI" AS aui, mrc."SCUI" AS scui,
                                   mrc."TTY" AS tty, mrc."ISPREF" AS ispref
                            FROM mrconso mrc
                            WHERE mrc."LAT" = :tok_lang
                              AND {like_conds}
                            ORDER BY {tok_rank}
                                 (mrc."ISPREF" = 'Y') DESC,
                                 LENGTH(mrc."STR") ASC, mrc."CUI" ASC
                            LIMIT :tok_limit
                        """)
                        tok_params["tok_lang"] = self.language
                        tok_params["tok_limit"] = max(1, int(limit))
                        rows = conn.execute(tok_sql, tok_params).fetchall()
                        if rows:
                            logger.debug(
                                "LocalUMLSClient.search | '%s' per "
                                "Token-AND gefunden", term)
        except (OperationalError, ProgrammingError) as e:
            logger.error(
                "LocalUMLSClient.search | DB-Fehler bei '%s': %s", term, e,
            )
            return []

        for r in rows:
            cui = r.cui
            sab = r.sab
            code = r.code
            name = r.name
            if return_id_type == "sourceConcept":
                ui = code
                uri = f"https://uts-ws.nlm.nih.gov/rest/source-asserted-concept/{sab}/{code}"
            elif return_id_type == "sourceDescriptor":
                ui = code
                uri = f"https://uts-ws.nlm.nih.gov/rest/source/{sab}"
            else:  # "concept"
                ui = cui
                uri = f"https://uts-ws.nlm.nih.gov/rest/content/{self.version}/CUI/{cui}"

            results.append({
                "ui": ui,
                "name": name,
                "rootSource": sab,
                "sab": sab,
                "code": code,
                "uri": uri,
                "preferredName": name if r.ispref == "Y" else name,
                "match": search_type,
            })

        with self._search_cache_lock:
            if len(self._search_cache) >= 50_000:
                self._search_cache.pop(next(iter(self._search_cache)))
            self._search_cache[cache_key] = results[:limit]

        logger.debug(
            "LocalUMLSClient.search | term=%r sabs=%s -> %d Treffer",
            term, sabs, len(results),
        )
        return results[:limit]

    # ------------------------------------------------------------------ concept
    def get_concept(self, cui: str) -> dict:
        """Holt Concept-Details (CUI) inkl. Namen, Definitionen, Atomen.

        Rückgabeformat ist 1:1 analog zur UMLS-API-Antwort:
        ``{"cui", "preferred_name", "semantic_types",
           "definitions": [{"value", "source"}],
           "atoms": [{"term", "source", "term_type", "language", "code"}]}``.
        """
        if not cui:
            return {
                "cui": cui, "preferred_name": "", "semantic_types": [],
                "definitions": [], "atoms": [],
            }

        with self._search_cache_lock:
            cached = self._concept_cache.get(cui)
            if cached is not None:
                return cached

        sql = text("""
            SELECT "CUI", "STR", "SAB", "ISPREF", "LAT", "TTY", "CODE", "AUI", "SCUI", "SUPPRESS"
            FROM mrconso
            WHERE "CUI" = :cui
            ORDER BY ("ISPREF" = 'Y') DESC, "SAB" ASC, "AUI" ASC
        """)

        sem_sql = text("""
            SELECT "TUI", "STY"
            FROM mrsty
            WHERE "CUI" = :cui
            ORDER BY "TUI"
        """)

        def_sql = text("""
            SELECT "DEF", "SAB", "SUPPRESS"
            FROM mrdef
            WHERE "CUI" = :cui
            ORDER BY "SAB"
        """)

        try:
            with self._connect() as conn:
                atoms_rows = conn.execute(sql, {"cui": cui}).fetchall()
                sem_rows = conn.execute(sem_sql, {"cui": cui}).fetchall()
                def_rows = conn.execute(def_sql, {"cui": cui}).fetchall()
        except (OperationalError, ProgrammingError) as e:
            logger.error(
                "LocalUMLSClient.get_concept | DB-Fehler für CUI=%s: %s",
                cui, e,
            )
            return {
                "cui": cui, "preferred_name": "", "semantic_types": [],
                "definitions": [], "atoms": [],
            }

        # Preferred Name: ISPREF='Y' und ggf. ENG
        preferred_name = ""
        atoms: list[dict] = []
        for r in atoms_rows:
            term = r.STR
            atoms.append({
                "term": term,
                "source": r.SAB,
                "term_type": r.TTY,
                "language": r.LAT,
                "code": r.CODE,
                "aui": r.AUI,
                "scui": r.SCUI,
                "ispref": r.ISPREF == "Y",
            })
            if not preferred_name and r.ISPREF == "Y":
                preferred_name = term
            elif not preferred_name and r.LAT == self.language:
                preferred_name = term
        if not preferred_name and atoms:
            preferred_name = atoms[0]["term"] or ""

        semantic_types = [
            {"tui": s.TUI, "name": s.STY}
            for s in sem_rows if s.STY
        ]

        definitions: list[dict] = []
        for d in def_rows:
            value = d.DEF
            if value:
                definitions.append({"value": value, "source": d.SAB})

        result = {
            "cui": cui,
            "preferred_name": preferred_name,
            "semantic_types": [s["name"] for s in semantic_types],
            "semantic_types_full": semantic_types,
            "definitions": definitions,
            "atoms": atoms,
        }
        with self._search_cache_lock:
            if len(self._concept_cache) >= 20_000:
                self._concept_cache.pop(next(iter(self._concept_cache)))
            self._concept_cache[cui] = result
        return result

    # ------------------------------------------------------------------ ancestors
    def get_ancestors(self, cui: str, relation_label: str = "RN",
                      max_results: int = 100,
                      sabs: list[str] | None = None) -> list[dict]:
        """Holt Vorfahren-Relationen eines CUI.

        Implementierts Verhalten analog zu ``UMLSClient.get_ancestors``:
        - Filter auf ``REL`` (``relation_label`` als Default ``"RN"``).
        - Mehrere REL-Labels können mit Komma übergeben werden (z.B.
          ``"RN,PAR"``); dann werden beide Richtungen vereint.
        - ``relatedId``-artige Filterung entfällt (alle Treffer sind
          CUIs), aber Duplikate werden über ``seen``-Set entfernt.
        - Bei der ``RN``-Richtung zeigt ``CUI1 -> CUI2`` (narrower->broader),
          d.h. ``CUI2`` ist der Vorfahre.

        Args:
            cui: Quell-CUI.
            relation_label: Komma-getrennte Liste von REL-Labels
                (Default ``"RN"``). Für Hierarchien empfiehlt sich
                ``"RN,PAR"``.
            max_results: Maximale Anzahl Rückgabe-Einträge.

        Returns:
            Liste ``[{"cui": "...", "name": "..."}]``.
        """
        if not cui:
            return []

        self._check_relation_tables()

        cache_key = (cui, tuple(sabs or ()), relation_label or "RN", max_results)
        with self._search_cache_lock:
            cached = self._ancestor_cache.get(cache_key)
            if cached is not None:
                return cached

        labels = [s.strip() for s in (relation_label or "").split(",") if s.strip()]
        if not labels:
            labels = ["RN"]

        # Bei RN / PAR zeigt CUI2 den Vorfahren. Für RB/CHD zeigt
        # CUI1 den Vorfahren. Wir vereinheitlichen auf "Ziel = Vorfahre".
        # Wenn jemand RB/CHD anfordert, ist das semantisch falsch, aber
        # wir liefern trotzdem das "andere Ende" als Vorfahre-Liste
        # (kompatibel zur ursprünglichen Interpretation, dass das Ziel
        # in der Beziehung immer ein verwandter Knoten ist).
        sab_clause = 'AND mrr."SAB" = ANY(:sabs)' if sabs else ""
        sql = text(f"""
                        SELECT mrr."REL", mrr."CUI1", mrr."CUI2", mrr."RELA",
                                     mrc."STR" AS name2, mrc2."STR" AS name1,
                                     mrr."SAB"
            FROM mrrel mrr
            LEFT JOIN mrconso mrc
                                ON mrc."CUI" = mrr."CUI2" AND mrc."LAT" = :language
                                AND mrc."ISPREF" = 'Y'
            LEFT JOIN mrconso mrc2
                                ON mrc2."CUI" = mrr."CUI1" AND mrc2."LAT" = :language
                                AND mrc2."ISPREF" = 'Y'
                        WHERE mrr."CUI1" = :cui
                            AND mrr."REL" = ANY(:labels)
                            AND COALESCE(mrr."SUPPRESS", 'N') = 'N'
                            {sab_clause}
                        ORDER BY mrr."SAB", mrr."REL"
            LIMIT :max_results
        """)

        ancestors: list[dict] = []
        seen: set[str] = set()
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, {
                    "cui": cui,
                    "labels": labels,
                    "language": self.language,
                    "max_results": max_results * 4,  # Puffer für Duplikate
                    **({"sabs": list(sabs)} if sabs else {}),
                }).fetchall()
        except (OperationalError, ProgrammingError) as e:
            logger.error(
                "LocalUMLSClient.get_ancestors | DB-Fehler für CUI=%s: %s",
                cui, e,
            )
            return []

        for r in rows:
            # Ziel-CUI ist hier CUI2; bei RN ist das der Oberbegriff
            related_cui = r.CUI2
            related_name = r.name2 or r.name1 or related_cui
            if not related_cui or related_cui == cui:
                continue
            if related_cui in seen:
                continue
            seen.add(related_cui)
            ancestors.append({
                "cui": related_cui,
                "name": related_name,
                "rel": r.REL,
                "rela": r.RELA,
                "sab": r.SAB,
            })
            if len(ancestors) >= max_results:
                break

        result = ancestors[:max_results]
        with self._search_cache_lock:
            if len(self._ancestor_cache) >= 20_000:
                self._ancestor_cache.pop(next(iter(self._ancestor_cache)))
            self._ancestor_cache[cache_key] = result
        logger.debug(
            "LocalUMLSClient.get_ancestors | cui=%s rel=%s -> %d Vorfahren",
            cui, relation_label, len(result),
        )
        return result

    def get_ancestor_graph(self, cui: str, cname: str,
                           sabs: list[str] | None = None,
                           max_depth: int = 5,
                           relation_label: str = "RN,PAR") -> list[dict]:
        """Liefert den lokalen Vorfahrengraphen bis zur gewünschten Tiefe.

        ``relation_label`` steuert, welche MRREL-``REL``-Labels als
        Hierarchiekante gelesen werden. ``RN`` allein verliert Kanten: eine
        ``RN``-Kante ohne ``RELA``-Qualifizierung ist eine nackte
        Cross-SAB-Verwandtschaft, während ``PAR`` die explizite
        Eltern-Kante desselben SAB ist. Deshalb Default ``"RN,PAR"`` --
        ``PAR`` ist in ``RN`` enthalten, die Vereinigung liefert also nur
        mehr Recall und keine zyklischen Duplikate (``get_ancestors``
        entdoppelt über ``seen``).
        """
        from collections import deque

        graph_key = (cui, tuple(sabs or ()), max_depth)
        with self._search_cache_lock:
            cached = self._ancestor_graph_cache.get(graph_key)
            if cached is not None:
                return cached

        queue = deque([(cui, None, 0)])
        visited = {cui}
        ancestors: list[dict] = []

        while queue:
            current_cui, parent_cui, distance = queue.popleft()
            parents = self.get_ancestors(current_cui, sabs=sabs, relation_label=relation_label)
            for parent in parents:
                parent_cui = parent["cui"]
                if parent_cui in visited:
                    continue
                visited.add(parent_cui)
                ancestor_distance = distance + 1
                ancestor_name = parent.get("name") or parent_cui
                ancestors.append({
                    "cui": parent_cui,
                    "parent_cui": current_cui,
                    "distance": ancestor_distance,
                    "relation": parent.get("rel"),
                    "name": ancestor_name,
                    "source": parent.get("sab"),
                })
                if ancestor_distance < max_depth:
                    queue.append((parent_cui, current_cui, ancestor_distance))

        with self._search_cache_lock:
            self._ancestor_graph_cache[graph_key] = ancestors
        return ancestors


# Convenience: Alias für Drop-in-Kompatibilität
UMLSClient = LocalUMLSClient

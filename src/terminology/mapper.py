import requests
import logging
import threading
from typing import NamedTuple

from src.models import Entity
from src.database import Repository
from src.terminology.snowstorm import SnowstormClient
from src.terminology.umls import UMLSClient

logger = logging.getLogger(__name__)


class _CacheEntry:
    """Platzhalter im Cache: signalisiert 'wird gerade gemappt'."""
    __slots__ = ("event", "concept_id", "failed")

    def __init__(self):
        self.event = threading.Event()
        self.concept_id: int | None = None
        self.failed = False


class MappingResult(NamedTuple):
    """Ergebnis eines Ontologie-Mappings."""
    concept_id: int | None
    ontology: str | None   # verwendeter Provider, z.B. snowstorm | umls

# Labels ohne Ontologie-Suche (strukturierte Werte, keine Konzepte)
NON_SEARCHABLE = {"TNM", "ICDO_CODE", "TUMORGROESSE", "NEGATION", "GRADING"}

# Root-Konzepte, die nicht als Vorfahre gespeichert werden sollen
SKIP_ANCESTORS = {"138875005"}  # SNOMED CT Concept (Root)


class OntologyMapper:
    """Wählt pro Label oder global zwischen Snowstorm (SNOMED) und UMLS.

    Konfiguration via config.yaml:

        pipeline:
          ontology_provider: snowstorm          # globaler Default: snowstorm | umls
          ontology_provider_per_label:          # optional pro Label überschreiben
            TUMORENTITAET: umls
            MORPHOLOGIE: snowstorm

    UMLS-spezifische Einstellungen kommen aus Settings/.env.
    """

    def __init__(
        self,
        repo: Repository,
        snowstorm_client: SnowstormClient | None = None,
        umls_client: UMLSClient | None = None,
        *,
        default_provider: str = "snowstorm",
        provider_per_label: dict[str, str] | None = None,
        snowstorm_terminology: str = "SNOMEDCT-DE",
        label_ecl: dict[str, str] | None = None,
        ancestor_depth: int = 3,
        cache_wait_timeout: float = 120.0,
        umls_sabs: list[str] | None = None,
        umls_terminology_version: str = "current",
    ):
        self.repo = repo
        self.snowstorm = snowstorm_client
        self.umls = umls_client
        self.default_provider = default_provider
        self.provider_per_label = provider_per_label or {}
        self.snowstorm_terminology = snowstorm_terminology
        self.label_ecl = label_ecl or {}
        self.ancestor_depth = ancestor_depth
        self.umls_sabs = umls_sabs or ["SNOMEDCT_US"]
        self.umls_version = umls_terminology_version
        self.cache_wait_timeout = cache_wait_timeout
        self._cache: dict[str, dict[str, _CacheEntry]] = {
            "snowstorm": {},
            "umls": {},
        }
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def map_entity(self, entity: Entity, report_id: int | str = "?") -> MappingResult:
        """Routed auf den konfigurierten Provider.

        Bereits gemappte ``search_term``-Werte werden aus einem internen
        Cache wiederverwendet, damit identische Begriffe nicht mehrfach gegen
        die Ontologie abgefragt werden.

        Returns:
            MappingResult mit concept_id und verwendetem Provider (ontology).
        """
        provider = self.provider_per_label.get(entity.label.value, self.default_provider)
        if provider not in ("umls", "snowstorm"):
            logger.debug(
                "REPORT %s | MAP | '%s' [%s] → Provider '%s' nicht verfügbar",
                report_id, entity.text, entity.label.value, provider,
            )
            return MappingResult(None, None)

        if entity.label.value in NON_SEARCHABLE or not entity.search_term:
            logger.debug("REPORT %s | MAP | '%s' [%s] übersprungen (nicht suchbar)",
                         report_id, entity.text, entity.label.value)
            return MappingResult(None, None)

        cache = self._cache[provider]
        key = entity.search_term.lower()

        with self._cache_lock:
            entry = cache.get(key)
            if entry is None:
                # Wir sind der erste Thread für diesen Begriff
                entry = _CacheEntry()
                cache[key] = entry
                owner = True
            else:
                owner = False

        if not owner:
            if not entry.event.is_set():
                logger.debug("REPORT %s | MAP | '%s' [%s] → wartet auf laufendes Mapping",
                             report_id, entity.search_term, entity.label.value)
                entry.event.wait(timeout=self.cache_wait_timeout)
                if not entry.event.is_set():
                    logger.warning(
                        "REPORT %s | MAP | '%s' → Timeout beim Warten auf Cache, "
                        "mappe selbst", report_id, entity.search_term)
                    return MappingResult(
                        self._dispatch(entity, provider, report_id), provider)
            logger.debug("REPORT %s | MAP | '%s' [%s] → Cache-Hit %s",
                         report_id, entity.search_term, entity.label.value,
                         entry.concept_id)
            return MappingResult(entry.concept_id, provider)

        try:
            concept_id = self._dispatch(entity, provider, report_id)
            entry.concept_id = concept_id
        except Exception:
            entry.failed = True
            with self._cache_lock:
                # Fehlversuch nicht dauerhaft cachen
                if cache.get(key) is entry:
                    del cache[key]
            raise
        finally:
            entry.event.set()

        return MappingResult(concept_id, provider)

    def _dispatch(self, entity: Entity, provider: str,
                  report_id: int | str) -> int | None:
        """Führt das eigentliche Mapping aus (ohne Cache-Logik)."""
        if provider == "umls" and self.umls:
            return self._map_umls(entity, report_id)
        if provider == "snowstorm" and self.snowstorm:
            return self._map_snowstorm(entity, report_id)
        logger.debug(
            "REPORT %s | MAP | '%s' [%s] → Provider '%s' nicht verfügbar",
            report_id, entity.text, entity.label.value, provider,
        )
        return None

    # ------------------------------------------------------------------ Snowstorm
    def _map_snowstorm(self, entity: Entity, report_id: int | str = "?") -> int | None:
        tag = f"REPORT {report_id} | MAP | snowstorm"

        if entity.label.value in NON_SEARCHABLE or not entity.search_term:
            logger.debug("%s | '%s' [%s] übersprungen (nicht suchbar)", tag, entity.text, entity.label.value)
            return None

        ecl = self.label_ecl.get(entity.label.value)
        logger.debug("%s | Suche '%s' (ECL: %s)", tag, entity.search_term, ecl)

        try:
            hits = self.snowstorm.search(entity.search_term, ecl=ecl, limit=1)
            if not hits and ecl:
                logger.debug("%s | Kein Treffer mit ECL, Fallback ohne Einschränkung", tag)
                hits = self.snowstorm.search(entity.search_term, ecl=None, limit=1)
            if not hits:
                logger.info("%s | '%s' [%s] → KEIN SNOMED-Treffer", tag, entity.search_term, entity.label.value)
                return None
            code = hits[0]["conceptId"]
        except requests.HTTPError as e:
            logger.error(f"{tag} | Error in Snowstorm Request for search: {entity.search_term} - {e}")
            return None

        try:
            details = self.snowstorm.get_concept(code)
        except requests.HTTPError as e:
            logger.warning("%s | Concept %s nicht abrufbar (%s) – übersprungen", tag, code, e)
            return None

        logger.info(
            "%s | '%s' [%s] → SNOMED %s | %s (%s) | %d Descriptions",
            tag, entity.search_term, entity.label.value, code,
            details["pt_term"], details["fsn_semantic_tag"], len(details["descriptions"]),
        )

        with self.repo.Session() as session:
            concept_id = self.repo.upsert_concept(
                self.snowstorm_terminology, code,
                details["pt_term"] or details["fsn_term"],
                semantic_type=details["fsn_semantic_tag"],
                terminology_version=self.snowstorm.branch,
                session=session,
                sab=self.snowstorm_terminology,
                finding=entity.label.value,
            )

            for desc in details["descriptions"]:
                logger.debug("%s |   Term: '%s' [%s/%s]", tag, desc["term"], desc.get("languageCode"), desc.get("type"))
                self.repo.upsert_term(
                    concept_id=concept_id,
                    term=desc["term"],
                    language=desc.get("languageCode") or "en",
                    term_type=desc.get("type") or "SYNONYM",
                    source="snowstorm",
                    session=session,
                )

            if self.ancestor_depth > 0:
                self._store_snowstorm_ancestors(code, concept_id, tag, session)
            session.commit()
        return concept_id

    def _store_snowstorm_ancestors(self, code: str, concept_id: int, tag: str, session):
        try:
            ancestors = self.snowstorm.get_ancestors(code)
        except requests.HTTPError as e:
            logger.warning("%s | Ancestors für %s nicht abrufbar (%s)", tag, code, e)
            ancestors = []

        n_rel = 0
        for i, anc in enumerate(ancestors, start=1):
            anc_code = anc["conceptId"]
            if anc_code in SKIP_ANCESTORS:
                logger.debug("%s |   Ancestor %s übersprungen (SKIP-Liste)", tag, anc_code)
                continue
            anc_pt = (anc.get("pt") or {}).get("term", anc_code)
            logger.debug("%s |   is_a[%d] → %s | %s", tag, i, anc_code, anc_pt)
            anc_id = self.repo.upsert_concept(
                self.snowstorm_terminology, anc_code, anc_pt, None,
                self.snowstorm.branch, session,
                sab=self.snowstorm_terminology, finding=None)
            self.repo.upsert_relation(
                concept_id, anc_id, "is_a", i,
                self.snowstorm.branch, session)
            n_rel += 1
        logger.info("%s | %s → %d Ancestor-Relationen gespeichert", tag, code, n_rel)

    # ------------------------------------------------------------------ UMLS
    def _map_umls(self, entity: Entity, report_id: int | str = "?") -> int | None:
        tag = f"REPORT {report_id} | MAP | umls"

        if entity.label.value in NON_SEARCHABLE or not entity.search_term:
            logger.debug("%s | '%s' [%s] übersprungen (nicht suchbar)", tag, entity.text, entity.label.value)
            return None

        logger.debug("%s | Suche '%s' in SABs: %s", tag, entity.search_term, self.umls_sabs)

        try:
            hits = self.umls.search(
                entity.search_term,
                sabs=self.umls_sabs,
                search_type="words",
                return_id_type="concept",
                limit=1,
            )
            if not hits:
                logger.debug("%s | '%s' [%s] → KEIN UMLS-Treffer", tag, entity.search_term, entity.label.value)
                return None
            cui = hits[0]["ui"]
            cname = hits[0]["name"]
            sab = hits[0].get("rootSource")
        except requests.HTTPError as e:
            logger.error(f"{tag} | Error in UMLS Request for search: {entity.search_term} - {e}")
            return None

        try:
            details = self.umls.get_concept(cui)
        except requests.HTTPError as e:
            logger.warning("%s | Concept %s nicht abrufbar (%s) – übersprungen", tag, cui, e)
            return None

        logger.debug(
            "%s | '%s' [%s] → UMLS %s | %s | SAB=%s | %d Atoms",
            tag, entity.search_term, entity.label.value, cui,
            details["preferred_name"], sab, len(details["atoms"]),
        )

        terminology = f"UMLS_{sab}" if sab else "UMLS"
        semantic_type = ", ".join(details["semantic_types"]) if details["semantic_types"] else None

        with self.repo.Session() as session:
            concept_id = self.repo.upsert_concept(
                terminology=terminology,
                concept_code=cui,
                preferred_term=details["preferred_name"],
                semantic_type=semantic_type,
                terminology_version=self.umls_version,
                session=session,
                sab=sab,
                finding=entity.label.value,
            )

            for atom in details["atoms"]:
                if not atom.get("term"):
                    continue
                self.repo.upsert_term(
                    concept_id=concept_id,
                    term=atom["term"],
                    language=atom.get("language") or "en",
                    term_type=atom.get("term_type") or "SYNONYM",
                    source=f"umls_{atom.get('source') or 'unknown'}",
                    session=session,
                )

            if self.ancestor_depth > 0:
                self._store_umls_ancestors(cui, cname, concept_id, tag, session)
            session.commit()
        return concept_id


    def _store_umls_ancestors(self, cui: str, cname: str, concept_id: int, tag: str, session):
        ancestors = self.umls.get_ancestor_graph(cui, cname=cname, sabs=self.umls_sabs, max_depth=self.ancestor_depth)

        node_ids: dict[str, int] = {cui: concept_id}
        n_rel = 0

        for anc in ancestors:
            anc_id = self.repo.upsert_concept(
                terminology="UMLS",
                concept_code=anc["cui"],
                preferred_term=anc.get("name") or anc["cui"],
                semantic_type=None,
                terminology_version=self.umls_version,
                session=session,
                sab=anc.get("source"),
                finding=None,
            )
            node_ids[anc["cui"]] = anc_id

            # Direkte Kante zum tatsächlichen Elternknoten (distance=1)
            src_id = node_ids.get(anc["parent_cui"])
            if src_id is not None:
                self.repo.upsert_relation(
                    source_id=src_id, target_id=anc_id,
                    relation_type="is_a", distance=1,
                    version=self.umls_version, session=session,
                )
                n_rel += 1

            # Zusätzlich transitive Kante zum Ursprungskonzept
            if anc["distance"] > 1:
                self.repo.upsert_relation(
                    source_id=concept_id, target_id=anc_id,
                    relation_type="ancestor_of",
                    distance=anc["distance"],
                    version=self.umls_version, session=session,
                )
                n_rel += 1

        logger.info("%s | %s → %d Relationen über %d Knoten in %s übergeordneten Leveln", tag, cui, n_rel, len(ancestors), self.ancestor_depth)


class SnowstormMapper:
    """Abwärtskompatibler Wrapper um OntologyMapper (nur Snowstorm)."""

    def __init__(self, client: SnowstormClient, repo: Repository,
                 terminology: str, label_ecl: dict[str, str],
                 max_ancestor_distance: int = 3):
        self._mapper = OntologyMapper(
            repo=repo,
            snowstorm_client=client,
            default_provider="snowstorm",
            snowstorm_terminology=terminology,
            label_ecl=label_ecl,
            ancestor_depth=max_ancestor_distance,
        )

    def map_entity(self, entity: Entity, report_id: int | str = "?") -> MappingResult:
        return self._mapper.map_entity(entity, report_id)


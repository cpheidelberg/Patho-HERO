import logging
from typing import Union
from concurrent.futures import ThreadPoolExecutor

from src.models import Entity, ExtractionResult
from src.extractors.gazetteer import GazetteerExtractor
from src.extractors.regex_extractor import RegexExtractor
from src.extractors.llm_extractor import LLMExtractor
from src.extractors.hf_extractor import HuggingFaceExtractor
from src.context.negation import NegationDetector
from src.terminology.mapper import OntologyMapper, MappingResult


logger = logging.getLogger(__name__)

class NERPipeline:
    def __init__(self, config: dict, llm: Union[LLMExtractor, HuggingFaceExtractor, None],
                 mapper: OntologyMapper | None):
        self.extractors = []
        if config["pipeline"]["use_gazetteer"]:
            self.extractors.append(GazetteerExtractor())
        if config["pipeline"]["use_regex"]:
            self.extractors.append(RegexExtractor(config["regex"]))
        self.llm = llm if config["pipeline"]["use_llm_fallback"] else None
        self.mapper = mapper if config["pipeline"].get("use_ontology", config["pipeline"].get("use_snowstorm")) else None
        self.negation = (NegationDetector(
            config["negation"]["triggers"], config["negation"]["uncertainty_triggers"],
            config["negation"]["window_tokens"])
            if config["pipeline"]["use_negation"] else None
        )
        self.config = config

    def add_extractor(self, extractor) -> None:
        self.extractors.append(extractor)

    def run(self, report_id: int, text: str, source_field: str = "raw_text") -> tuple[list[dict], list[Entity]]:
        """Gibt Zeilen für report_concept zurück (mit/ohne concept_id)."""
        logger.info("=" * 70)
        logger.info("REPORT %s | Start Extraktion (%d Zeichen)", report_id, len(text))
        preview = text[:120].replace("\n", " ")
        logger.debug("REPORT %s | Textvorschau: %s...", report_id, preview)

        entities: list[Entity] = []
        for ex in self.extractors:
            found = ex.extract(text)
            for f in found:
                logger.info("REPORT %s | ENTITÄT | '%s' [%s] search_term='%s' (Quelle: %s, Offset %s-%s)",
                    report_id, f.text, f.label.value, f.search_term, ex.__class__.__name__,
                    getattr(f, "start", "?"), getattr(f, "end", "?"))
            logger.info("REPORT %s | %s: %d Entitäten", report_id, ex.__class__.__name__, len(found))
            entities.extend(found)
        entities = self._deduplicate(entities)

        if self.llm:
            covered = [e.text for e in entities]
            for e in self.llm.extract(text, exclude=covered):
                if any(e.start < b.end and b.start < e.end for b in entities):
                    continue  # überlappt mit bereits extrahierter Entität
                entities.append(e)

        if self.negation:
            entities = self.negation.apply(text, entities)

        for entity in entities:
            logger.debug("Extracted entity: %s [%s] with confidence %s in original text - %s", entity.text, entity.label.value, entity.confidence, text[entity.start:entity.end].lower())
        entities = [e for e in entities if text[e.start:e.end].lower() == e.text.lower()]
        rows = []
        n_mapped, n_skipped = 0, 0
        if self.mapper and entities:
            with ThreadPoolExecutor(max_workers=int(self.config["pipeline"].get("mapping_workers", 1))) as exc:
                mappings = list(exc.map(lambda ent: self.mapper.map_entity(ent, report_id), entities))
        else:
            mappings = [MappingResult(None, None) for _ in entities]

        for e, mapping in zip(entities, mappings):
            if mapping.concept_id:
                n_mapped += 1
            else:
                n_skipped += 1
            rows.append({
                "concept_id": mapping.concept_id,
                "source_field": source_field,
                "surface_text": e.text,
                "raw_text": e.text,
                "search_term": e.search_term,
                "ontology": mapping.ontology,
                "start_offset": e.start,
                "end_offset": e.end,
                "assertion": e.assertion.value,
                "certainty": "confirmed" if not e.negated else "confirmed",
                "temporality": "not_stated",
                "laterality": "not_stated",
                "confidence": 1.0 if e.source.value != "llm" else 0.5,
            })
        logger.info("REPORT %s | FERTIG | %d Entitäten, %d ontology-gemappt, %d ohne Konzept", report_id, len(entities), n_mapped, n_skipped)
        return rows, entities

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        seen, out = set(), []
        for e in entities:
            key = (e.start, e.end, e.label)
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out
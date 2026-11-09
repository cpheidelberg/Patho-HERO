import json
import os
import sys
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(sys.path[0], ".."))

from src.config import Settings, load_pipeline_config
from src.database import Repository
from src.models import Assertion, Entity, Label, Source
from src.terminology.local_umls import LocalUMLSClient
from src.terminology.mapper import OntologyMapper
from src.terminology.snowstorm import SnowstormClient
from src.terminology.umls import UMLSClient


def _enum_or_value(enum_cls, value: Any, default):
    """Convert a JSON value to an enum, falling back if it is not recognized."""
    if value is None:
        return default
    try:
        return enum_cls(value)
    except (TypeError, ValueError):
        return default


def span_to_entity(span: dict[str, Any]) -> Entity:
    """Convert one JSONL span into the application's Entity model."""
    meta = span.get("meta") or {}

    label = _enum_or_value(Label, span.get("label"), Label.MISCELLANEOUS)
    source = _enum_or_value(Source, meta.get("source"), Source.LLM)
    assertion = _enum_or_value(
        Assertion, meta.get("assertion"), Assertion.PRESENT
    )

    return Entity(
        label=label,
        text=span["text"],
        start=int(span["start"]),
        end=int(span["end"]),
        assertion=assertion,
        negated=assertion == Assertion.ABSENT,
        confidence=str(meta.get("confidence", "high")),
        source=source,
        search_term=span.get("search_term") or span["text"],
    )


def load_jsonl(path: str | Path):
    """Yield (report_id, source record, Entity) for every span in the JSONL."""
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            report_id = int(record["meta"]["report_id"])

            for span in record.get("spans", []):
                yield report_id, record, span_to_entity(span)


def create_mappers(settings, config, repo):
    """Build one mapper per provider using the application's existing clients."""
    accept_language = os.getenv("SNOWSTORM_ACCEPT_LANGUAGE", "de,en;q=0.5")
    # One client per SNOMED edition, each on its own branch:
    #   German     -> MAIN          (primary language de)
    #   International -> MAIN/INTL  (primary language en)
    snowstorm_de = SnowstormClient(
        settings.snowstorm_base_url,
        "MAIN",
        "de",
        accept_language,
    )
    snowstorm_intl = SnowstormClient(
        settings.snowstorm_base_url,
        "MAIN/INTL",
        "en",
        accept_language,
    )

    remote_umls = UMLSClient(
        settings.umls_api_key,
    )
    local_umls = LocalUMLSClient(
        settings.umls_api_key,
        repository=repo,
        language=settings.umls_local_language,
    )

    def _base(snowstorm_client, snowstorm_terminology, label_ecl=None):
        return dict(
            repo=repo,
            snowstorm_client=snowstorm_client,
            snowstorm_terminology=snowstorm_terminology,
            label_ecl=label_ecl if label_ecl is not None else config.get("label_ecl", {}),
            ancestor_depth=config["pipeline"].get("ancestor_depth", 1),
            umls_sabs=settings.umls_sabs_list,
            umls_terminology_version=settings.umls_version,
        )

    # A mapper uses one provider at a time. Disable provider-per-label
    # overrides so the experiment always tests the requested provider.
    return {
        "umls_remote": OntologyMapper(
            **_base(None, "UMLS"),
            umls_client=remote_umls,
            default_provider="umls",
            provider_per_label={},
        ),
        "umls_local": OntologyMapper(
            **_base(None, "UMLS"),
            umls_client=local_umls,
            default_provider="umls",
            provider_per_label={},
        ),
        "snomed_de": OntologyMapper(
            **_base(snowstorm_de, "SNOMEDCT-DE"),
            umls_client=None,
            default_provider="snowstorm",
            provider_per_label={},
        ),
        "snomed_intl": OntologyMapper(
            # The label_ecl hierarchies are SNOMED-DE specific; they do not
            # resolve on the International edition, so search it unscoped.
            **_base(snowstorm_intl, "SNOMEDCT", label_ecl={}),
            umls_client=None,
            default_provider="snowstorm",
            provider_per_label={},
        ),
    }


def _map_all(provider_mappers, entity, report_id):
    """Build the per-provider ``mappings`` object (same as eval_extractor)."""
    mappings = {}
    for provider, mapper in provider_mappers.items():
        try:
            mapping = mapper.map_entity(entity, report_id)
            mappings[provider] = {
                "concept_id": mapping.concept_id,
                "ontology": mapping.ontology,
            }
        except Exception as exc:  # keep going across providers/spans
            mappings[provider] = {
                "concept_id": None,
                "ontology": provider,
                "error": str(exc),
            }
    return mappings


def main():
    input_path = Path("experiments/hybrid_ner.jsonl")
    output_path = Path("experiments/hybrid_ner_mappings.jsonl")
    max_workers = 12

    settings = Settings()
    config = load_pipeline_config()
    repo = Repository(settings.database_url)

    mappers = create_mappers(settings, config, repo)


    rows = list(load_jsonl(input_path))
    print(f"{len(rows)} spans to map across {len(mappers)} providers, "
          f"{max_workers} workers", flush=True)

    results = [None] * len(rows)
    done = 0

    def work(i):
        report_id, record, entity = rows[i]
        return {
            "report_id": report_id,
            "text_id": record.get("text_id"),
            "span": {
                "text": entity.text,
                "label": entity.label.value,
                "start": entity.start,
                "end": entity.end,
                "search_term": entity.search_term,
            },
            "mappings": _map_all(mappers, entity, report_id),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(work, i): i for i in range(len(rows))}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(rows):
                print(f"  progress {done}/{len(rows)}", flush=True)

    with output_path.open("w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} records -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
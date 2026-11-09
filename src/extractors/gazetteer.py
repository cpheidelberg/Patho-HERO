import logging
import re
from pathlib import Path

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.models import Entity, Label, Source
from src.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class GazetteerExtractor(BaseExtractor):
    """Dictionary-Matching mit exakten Offsets via Regex."""

    FILE_TO_LABEL = {
        "probe.txt": Label.PROBE,
        "lokalisationen.txt": Label.LOCALISATION,
        "probenarten.txt": Label.PROBE_TYPE,
        "faerbungen.txt": Label.STAINING,
        "ihc_marker.txt": Label.IHC_MARKER,
    }

    def __init__(self, gazetteer_dir: str = "config"):
        self.entries: list[tuple[re.Pattern, Label, str]] = []
        base = Path(gazetteer_dir)
        for filename, label in self.FILE_TO_LABEL.items():
            path = base / filename
            if not path.exists():
                logger.warning("Gazetteer: %s fehlt (wird übersprungen)", path)
                continue
            loaded = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                term = line.strip()
                if not term:
                    continue
                pattern = re.compile(r"\b" + re.escape(term) + r"\b",
                                     re.IGNORECASE)
                self.entries.append((pattern, label, term))
                loaded += 1
            logger.info("Gazetteer: %s -> %d Einträge (%s)", filename, loaded, path)
        if not self.entries:
            logger.warning("Gazetteer: KEINE Einträge geladen aus %s "
                           "-- dieser Extractor ist faktisch INAKTIV", base)

    def extract(self, text: str) -> list[Entity]:
        entities = []
        for pattern, label, canonical in self.entries:
            for m in pattern.finditer(text):
                entities.append(Entity(
                    label=label, text=m.group(), start=m.start(), end=m.end(),
                    source=Source.GAZETTEER, search_term=canonical))
        return entities
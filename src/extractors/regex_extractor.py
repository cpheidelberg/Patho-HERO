import re

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from src.models import Entity, Label, Source
from src.extractors.base import BaseExtractor


class RegexExtractor(BaseExtractor):
    """Strukturierte Angaben: ICD-O, TNM, Tumorgrößen."""

    def __init__(self, patterns: dict[str, str]):
        self.patterns = {
            Label.ONTOLOGY: re.compile(patterns["icdo"]),
            Label.CLINICAL: re.compile(patterns["tnm"]),
            Label.PROBE_SIZE: re.compile(patterns["groesse"], re.IGNORECASE),
        }

    def extract(self, text: str) -> list[Entity]:
        entities = []
        for label, pattern in self.patterns.items():
            for m in pattern.finditer(text):
                entities.append(Entity(
                    label=label, text=m.group(), start=m.start(), end=m.end(),
                    source=Source.REGEX, search_term=None))
        return entities
from abc import ABC, abstractmethod

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.models import Entity


class BaseExtractor(ABC):
    """Einheitliche Schnittstelle für alle Extraktoren.
    Neue Extraktoren (z.B. trainiertes NER-Modell) einfach ableiten."""

    @abstractmethod
    def extract(self, text: str) -> list[Entity]:
        ...
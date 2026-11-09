import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from src.models import Entity, Assertion


class NegationDetector:
    """NegEx-artig: Trigger im Fenster vor der Entität → negated/uncertain."""

    def __init__(self, neg_triggers: list[str], unc_triggers: list[str],
                 window: int = 6):
        self.neg = sorted(neg_triggers, key=len, reverse=True)
        self.unc = sorted(unc_triggers, key=len, reverse=True)
        self.window = window

    def apply(self, text: str, entities: list[Entity]) -> list[Entity]:
        for e in entities:
            prefix = text[max(0, e.start - 120):e.start].lower()
            words = prefix.split()[-self.window:]
            window_text = " ".join(words)
            if any(t.lower() in window_text for t in self.unc):
                e.assertion = Assertion.UNCERTAIN
            elif any(t.lower() in window_text for t in self.neg):
                e.negated = True
                e.assertion = Assertion.ABSENT
        return entities
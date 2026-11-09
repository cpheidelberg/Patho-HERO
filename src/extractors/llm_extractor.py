import json
import os
import re
import requests
import logging
from typing import Optional
from pydantic import BaseModel

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.models import Entity, Label, Source
from src.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'llm_ner_prompt.md')
with open(_PROMPT_PATH, encoding='utf-8') as _f:
    PROMPT = _f.read()

# Der Prompt erlaubt deutsche Labels; intern arbeiten wir mit englischen Label-Namen.
_LABEL_MAP = {
    "Probe": Label.PROBE,
    "Probenart": Label.PROBE_TYPE,
    "Probengröße": Label.PROBE_SIZE,
    "Lokalisation": Label.LOCALISATION,
    "Klinisch": Label.CLINICAL,
    "Verfahren": Label.PROCEDURE,
    "Makroelement": Label.MACRO_ELEMENT,
    "Makromorphologie": Label.MACRO_MORPHOLOGY,
    "Mikroelement": Label.MICRO_ELEMENT,
    "Mikromorphologie": Label.MICRO_MORPHOLOGY,
    "Befund": Label.FINDING,
    "Diagnose": Label.DIAGNOSE,
    "Färbung": Label.STAINING,
    "IHC-Marker": Label.IHC_MARKER,
    "Ontologie": Label.ONTOLOGY,
    "Negation": Label.NEGATION,
    "Sonstiges": Label.MISCELLANEOUS,
}

class LLMEntitySchema(BaseModel):
    """Schema für Ollama structured output – bewusst OHNE Offsets."""
    label: str
    text: str
    search_term: Optional[str] = None
    negated: bool = False


class LLMResponseSchema(BaseModel):
    entities: list[LLMEntitySchema]


class LLMExtractor(BaseExtractor):
    """Fallback für nicht abgedeckte Begriffe. Liefert text + label,
    Offsets werden anschließend im Original verortet (kein LLM-Counting!)."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.0):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.num_ctx = int(os.environ.get("OLLAMA_CONTEXT_SIZE", "8192"))
        # Kontext-Vergrößerung für den einen Wiederholungsversuch, wenn das
        # erste Ergebnis abgeschnitten/ungültig ist. "EOF while parsing" in
        # der Log bedeutet: Antwort wurde an der num_ctx-Grenze abgebrochen.
        try:
            self.num_ctx_retry = int(os.environ.get("OLLAMA_CONTEXT_SIZE_RETRY", str(self.num_ctx * 4)))
        except ValueError:
            self.num_ctx_retry = self.num_ctx * 4
        raw_keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "-1")
        try:
            self.keep_alive = int(raw_keep_alive)   # -1 = keep loaded forever
        except ValueError:
            self.keep_alive = raw_keep_alive          # duration string e.g. "30m"

    def _request(self, prompt: str, num_ctx: int, timeout: int) -> str | None:
        """Führt genau einen Ollama-Request aus und gibt die rohe Antwort zurück.

        Bei Netzwerk-/HTTP-Fehler wird ``None`` geliefert (Aufrufer entscheidet).
        """
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                "format": LLMResponseSchema.model_json_schema(),  # constrained decoding
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": num_ctx,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response")

    @staticmethod
    def _salvage_entities(raw: str) -> list[dict]:
        """Rettet vollständige Entitäten aus einer abgeschnittenen/defekten
        JSON-Ausgabe.

        Liest ausschließlich gut-formed Blöcke ``{"label": ..., "text": ...}``
        aus dem ``entities``-Array. Der letzte, abgeschnittene Eintrag (oder
        jedes Feld mit ungeschlossener Klammer) fällt automatisch heraus,
        weil ``"text": "`` nur über ein abschließendes Zitat erkannt wird.
        Rückgabe sind rohe Dictionaries (label/text/search_term/negated);
        die Validierung gegen das Schema übernimmt der Aufrufer.
        """
        if not raw:
            return []
        pattern = r"\{\s*\"label\"\s*:\s*\"([^\"]*)\"\s*," \
                  r"\s*\"text\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"(.*?)" \
                  r"\}\s*[,]\s*"
        found: list[dict] = []
        for m in re.finditer(pattern, raw):
            label = m.group(1)
            text_val = m.group(2)
            tail = m.group(3) or ""
            negated = False
            search_term = None
            sm = re.search(r"\"search_term\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"", tail)
            if sm and sm.group(1) not in ("", "null"):
                search_term = sm.group(1)
            nm = re.search(r"\"negated\"\s*:\s*(true|false)", tail)
            if nm:
                negated = nm.group(1) == "true"
            found.append({
                "label": label,
                "text": text_val,
                "search_term": search_term,
                "negated": negated,
            })
        return found

    def extract(self, text: str, exclude: Optional[list[str]] = None) -> list[Entity]:
        prompt = PROMPT.format(exclude=", ".join(exclude or ["-"]), text=text)
        logger.debug("LLM | Prompt (%d Zeichen) an %s", len(prompt), self.model)

        timeout = 300
        try:
            timeout = int(os.environ.get("OLLAMA_TIMEOUT", "300"))
        except (TypeError, ValueError):
            logger.warning("OLLAMA_TIMEOUT hat ungültigen Wert, verwende Default 300s")

        def _parse(raw: str | None):
            if raw is None:
                return None, None
            try:
                return LLMResponseSchema.model_validate_json(raw), raw
            except Exception as e:
                return None, (raw, e)

        def _salvage(raw: str | None) -> list | None:
            if raw is None:
                return None
            items = self._salvage_entities(raw)
            if not items:
                return None
            parsed = LLMResponseSchema(entities=items)
            logger.info(
                "LLM | %d Entitäten aus abgeschnittener Antwort gerettet",
                len(parsed.entities))
            return parsed

        # --- Versuch 1 (Standard-Kontext) ---
        raw: str | None = None
        try:
            raw = self._request(prompt, self.num_ctx, timeout)
        except requests.exceptions.RequestException as e:
            logger.warning("LLM | Anfrage an Ollama fehlgeschlagen: %s", e)
            return []

        parsed, parse_err = _parse(raw)
        if parsed is not None:
            items = parsed.entities
        else:
            # --- Fehler: ein einziger Wiederholungsversuch mit anderem Setting
            # (größeres num_ctx, seltener Abbruch am Kontext-Ende). ---
            logger.warning(
                "LLM | Parse-Fehler bei Versuch 1: %s | versuche Retry mit "
                "num_ctx=%s", parse_err[1] if parse_err else "?", self.num_ctx_retry)
            try:
                raw2 = self._request(prompt, self.num_ctx_retry, timeout)
            except requests.exceptions.RequestException as e:
                logger.warning("LLM | Retry-Request fehlgeschlagen: %s", e)
                raw2 = None
            parsed2, _ = _parse(raw2)
            if parsed2 is not None:
                items = parsed2.entities
            else:
                # --- Letzter Ausweg: vollständige Entitäten aus dem jeweiligen
                # (hoffentlich) längeren Roh-Text retten. ---
                salvaged = None
                if raw2 is not None:
                    salvaged = _salvage(raw2)
                if salvaged is None:
                    salvaged = _salvage(raw if raw is not None else raw2)
                if salvaged is not None:
                    items = salvaged.entities
                else:
                    if parse_err:
                        logger.warning(
                            "LLM | Retry ebenfalls unlesbar – keine Entitäten. "
                            "Raw (Retry, gekürzt): %.300s", (raw2 or "")[:300])
                    else:
                        logger.warning("LLM | keine Entitäten (leere oder ungültige Antwort)")
                    return []

        entities: list[Entity] = []
        discarded = 0
        for item in items:
            if item.label in Label.__members__:
                label = Label[item.label]
            elif item.label in _LABEL_MAP:
                label = _LABEL_MAP[item.label]
            else:
                label = Label.MISCELLANEOUS
            # Robust offset lookup:
            #  1. verbatim
            #  2. mit abgeschnittenen Satzzeichen (.,;:!?-) am Rand
            #  3. case-insensitiv (falls die LLM-Zeichensetzung leicht abweicht)
            candidates = [item.text]
            trimmed = item.text.strip(" .,;:!?–—-")
            if trimmed and trimmed != item.text:
                candidates.append(trimmed)
            lo_text = text.lower()
            found = False
            for cand in candidates:
                pos = text.find(cand)
                if pos == -1:
                    pos = lo_text.find(cand.lower())
                if pos == -1:
                    continue
                while pos != -1:
                    entities.append(Entity(
                        label=label, text=cand, start=pos,
                        end=pos + len(cand), source=Source.LLM,
                        confidence="low", negated=item.negated,
                        search_term=item.search_term))
                    pos = text.find(cand, pos + 1)
                found = True
                break
            if not found:
                discarded += 1
                logger.warning("LLM | '%s' nicht im Originaltext gefunden – verworfen", item.text)
        logger.info("LLM | %d Entitäten extrahiert (%d verworfen)", len(entities), discarded)
        return entities
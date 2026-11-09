from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Label(str, Enum):
    PROBE = "probe"
    PROBE_SIZE = "probe_size"
    PROBE_TYPE = "probe_type"
    LOCALISATION = "localisation"
    CLINICAL = "clinical"
    MACRO_ELEMENT = "macro_element"
    MACRO_MORPHOLOGY = "macro_morphology"
    MICRO_ELEMENT = "micro_element"
    MICRO_MORPHOLOGY = "micro_morphology"
    PROCEDURE = "procedure"
    FINDING = "finding"
    DIAGNOSE = "diagnose"
    IHC_MARKER = "ihc_marker"
    STAINING = "staining"
    ONTOLOGY = "ONTOLOGY"
    NEGATION = "NEGATION"
    MISCELLANEOUS = "miscellaneous"

class Assertion(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class Source(str, Enum):
    GAZETTEER = "gazetteer"
    REGEX = "regex"
    LLM = "llm"


class Entity(BaseModel):
    label: Label
    text: str
    start: int
    end: int
    negated: bool = False
    assertion: Assertion = Assertion.PRESENT
    confidence: str = "high"
    source: Source
    search_term: Optional[str] = None   # normalisierter Begriff für Snowstorm


class ExtractionResult(BaseModel):
    report_id: int
    entities: list[Entity] = Field(default_factory=list)


class ReportProcessingResult(BaseModel):
    report_ref: str
    report_id: int
    success: bool
    n_entities: int = 0
    n_mapped: int = 0
    mean_confidence: Optional[float] = None
    error: Optional[str] = None
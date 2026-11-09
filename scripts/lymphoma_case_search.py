#!/usr/bin/env python3
"""Automatische Suche nach Lymphom-Befunden mit den vier Wachstumsmustern.

Das Skript bildet die manuelle Filterlogik des Pathologen nach, die er für
die manuelle Selektion der Lymphom-Schnitte verwendet hat. 
Negationen werden berücksichtigt.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import psycopg
from psycopg.rows import dict_row

from src.config import Settings
from src.logging_config import setup_logging

logger = logging.getLogger("lymphoma_case_search")

# ---------------------------------------------------------------------------
# Kriterien des Pathologen als Regexe
# ---------------------------------------------------------------------------

# (1) Stanzbiopsien -- Stanze, Stanzzylinder, Stanzbiopsie, ...
RE_STANZ = re.compile(r"\bstanz\w*", re.IGNORECASE)

# (2) "Befunde ohne Krebsdiagnose" -- der Pathologe erkennt sie an der
#     Negativaussage am Ende des Befunds.
RE_NO_MALIGNITY = re.compile(
    r"(?:kein\s+anhalt\s+für\s+malignität"
    r"|kein\s+nachweis\s+von\s+malignität"
    r"|keine\s+malignität\s+nachweisbar"
    r"|kein\s+malignitätsnachweis"
    r"|keine\s+anteile\s+einer\s+malignität"
    r"|malignität\s+(?:konnte\s+)?nicht\s+nachgewiesen"
    r"|kein\s+hinweis\s+(?:auf\s+)?eine?\s+maligne?\s*"
    r"(?:neoplasie|erkrankung))",
    re.IGNORECASE,
)

# (6) Materialart -- Wortgrenze, damit "LK" nicht in Fremdwörtern matcht.
RE_MATERIAL_LYMPH_NODE = re.compile(
    r"(?:lymphknoten|\blks?\b)", re.IGNORECASE
)

# Wachstumsmuster. Die Synonyme entsprechen der Liste des Pathologen,
# ergänzt um die in den Befunden tatsächlich vorkommenden Schreibweisen.
#
# "nodulär" und "diffus" haben je einen expliziten und einen unscharfen
# Teil. Der unscharfe Teil ("knotig", "diffus" vor Fibrose-/Einblutungs-
# Substantiven) wird nur gewertet, wenn im selben Satz ein Substantiv
# steht, das ein neoplastisches Wachstum beschreibt -- sonst schluckt
# der Filter "knotige Fibrosezonen" oder "diffuse Sklerose" ein, die mit
# dem Wachstumsmuster des Lymphoms nichts zu tun haben.
PATTERNS: dict[str, re.Pattern[str]] = {
    # (3) nodulär = nödos / knotig
    "nodular": re.compile(
        r"(?:nodulär|nodos|nödös|mikronodulär|knoten[- ]in[- ]knoten"
        r"|knotige[nrs]?\s+(?:infiltrate|proliferat\w*|formationen)"
        r"|knotig\w*)",
        re.IGNORECASE,
    ),
    # (3) diffus = zur Konfluenz neigend / konfluierend.
    # Bewusst wird auch "diffus großzelliges B-Zell-Lymphom" (DLBCL)
    # getroffen: das ist zwar eine Entitätsbezeichnung, bezeichnet aber
    # faktisch ein diffuses Wachstumsmuster.
    "diffuse": re.compile(
        r"(?:diffus\w*|konfluierend\w*|zur\s+konfluenz\s+neigend"
        r"|konfluent\w*)",
        re.IGNORECASE,
    ),
    # (4) minimal-change -- erhaltene Architektur / Grundstruktur
    "minimal-change": re.compile(
        r"(?:erhalten\w*\s+(?:lymphknoten[- ]?architektur|grundstruktur"
        r"|grundarchitektur|architektur|struktur)"
        r"|erhalt\w*\s+grundstruktur"
        r"|geringfügige\s+strukturstörung"
        r"|geringgradige\s+strukturstörung"
        r"|beginnende\s+strukturstörung"
        r"|minimal\s+infiltrat\w*)",
        re.IGNORECASE,
    ),
    # (5) nekrotisch -- Nekrose / Nekroseherd / nekrotisches Areal / ...
    "necrosis": re.compile(
        r"(?:nekrose\w*|nekrotisch\w*|koagulationsnekrose"
        r"|tumorzellnekrose\w*)",
        re.IGNORECASE,
    ),
}

# Teil der "nodulär"/"diffus"-Synonyme, der ohne Wachstumskontext
# zaehlt: "knotige Fibrosezonen", "diffuse Sklerose", "diffuse Einblutung".
RE_IMPLICIT_GROWTH = re.compile(r"knotig\w*|diffus\w*|konfluent\w*",
                                re.IGNORECASE)

# Substantive, die ein neoplastisches Wachstumsmuster beschreiben.
RE_GROWTH_NOUN = re.compile(
    r"\b(?:infiltrat\w*|proliferat\w*|formationen?|nest\w*|aggregat\w*"
    r"|zellen\w*|zellverband\w*|lympho\w*|blast\w*|tumor\w*|tumorzellen"
    r"|geschwulst\w*|neoplas\w*)\b",
    re.IGNORECASE,
)

# Substantive, bei denen "knotig"/"diffus" die Makroskopie oder eine
# Begleitveraenderung beschreiben und nicht das Wachstumsmuster.
RE_GROWTH_NOISE = re.compile(
    r"\b(?:fibr\w*|skleros\w*|sklerot\w*|indur\w*|bindegeweb\w*"
    r"|muskul\w*|kapsel|perinodal\w*|einblutung\w*|blutung\w*"
    r"|granulations\w*|ödem\w*|ödemat\w*|atroph\w*|hyalin\w*"
    r"|lipomat\w*|haemorrhag\w*|hämorrhag\w*)\b",
    re.IGNORECASE,
)

# Negation direkt vor dem Treffer ("Keine Nekrosen", "keine nodulären
# Infiltrate", "ohne nekrotisches Areal", "nicht konfluierend").
RE_NEGATION_WINDOW = re.compile(
    r"(?:kein\w*|ohne|nicht|keiner|keine|fehlt|es\s+finden\s+sich\s+keine)\s+"
    r"(?:[a-zäöüß-]+\s+){0,3}$",
    re.IGNORECASE,
)

PATTERN_LABELS_DE = {
    "nodular": "nodulär",
    "diffuse": "diffus",
    "minimal-change": "minimal-change",
    "necrosis": "nekrotisch",
}

# Abschnitte, in denen je Wachstumsmuster gesucht wird -- siehe
# LymphomaSearch.classify(). Reihenfolge egal, die Abschnitte werden
# verkettet.
#   "mikroskopie" = Abschnitt "Mikroskopie" (mit Fallback auf den ganzen
#                   Befund, wenn es ihn nicht gibt -- er fehlt bei rund
#                   der Haelfte der Befunde)
#   "report"      = gesamter Befund
# Der Pathologe nennt fuer die Muster selbst keine Abschnittsgrenze
# ("diese Woerter meistens direkt im Befundtext"). Begrenzt wird auf
# Mikroskopie und Diagnose, weil Diagnose und Nachbericht die
# abschliessende Beurteilung enthalten ("mit knotigen, die
# Lymphknotenarchitektur komplett aufhebenden Tumorinfiltraten") und
# Material/Klinische Angaben/Makroskopie regelmaessig Stoerwoerter wie
# "knotiges Exzisat" einbringen wuerden.
PATTERN_SCOPE = {
    "nodular": ("mikroskopie", "diagnose", "nachbericht"),
    "diffuse": ("mikroskopie", "diagnose", "nachbericht"),
    "necrosis": ("mikroskopie", "diagnose", "nachbericht"),
    "minimal-change": ("report",),
}

# Lymphom-Diagnosen. Der Pathologe hat in einem Ordner "LymphomaPathology"
# gesucht, also muss ein Lymphom (bzw. eine leukämische Lymphomform) genannt
# sein -- Metastasen eines Karzinoms in Lymphknoten sind nicht gemeint.
RE_LYMPHOMA = re.compile(
    r"(?:hodgkin[- ]?lymphom\w*|klassisch\w*\s+klassischen?\s+hodgkin"
    r"|non[- ]?hodgkin[- ]?lymphom\w*|\bnhl\b|lymphom\w*|lymphomatos\w*"
    r"|lymphatische\s+leuk[äa]mie\w*|\bcll\b|\bsll\b"
    r"|lymphoplasmozellul[äa]r\w*|mantelzell\w*|marginalzonen\w*"
    r"|follicul\w*\s+lymphom|burkitt\w*|mycosis\s+fungoides"
    r"|angioimmunblastisch\w*|peripher\s+t-zell\w*|nodal\w*\s+marginalzonen)",
    re.IGNORECASE,
)

# Vorl[äa]ufiger Befund nannte entz[üu]ndlich/unspezifisch, der Abschluss
# ein Lymphom -> Kriterium 4, zweiter Weg.
RE_PROVISIONAL_BENIGN = re.compile(
    r"(?:reaktiv\w*|entz[üu]ndlich\w*|unspezifisch\w*|hyperplasie\w*"
    r"|favorisiert\w*\s+differentialdiagnostisch\w*"
    r"|kein\s+konventionell[- ]mikroskopisch\w*\s+nachweis"
    r"|nicht\s+konventionell[- ]mikroskopisch\w*\s+erkennbar)",
    re.IGNORECASE,
)

# Grobe Unterteilung der Diagnose in die drei Spalten der Excel-Liste des
# Pathologen (hl / aggressiv / indolent). Bewusst konservativ: der
# Pathologe hat selbst nach Vorbericht und Zusatzbericht differenziert,
# eine echte WHO-Kodierung steht in der Datenbank nicht zur Verfuegung.
RE_TYPE_HL = re.compile(r"hodgkin", re.IGNORECASE)

RE_TYPE_AGGRESSIVE = re.compile(
    r"(?:burkitt|aggressiv\w*|hochmalign\w*|high[- ]grade|blast\w*"
    r"|primär\w*\s+mediastinal\w*|peripher\w*\s+t-zell\w*"
    r"|angioimmunblastisch\w*|t-zell\w*|mantelzell\w*|nos\w*)",
    re.IGNORECASE)

RE_TYPE_INDOLENT = re.compile(
    r"(?:follicul\w*|marginalzonen?\w*|lymphozytisch\w*|kleinzellig\w*"
    r"|grad\s*1(?:\s*/\s*2)?\b|low[- ]grade"
    r"|nodal\w*\s+sklerosierend\w*|lymphozyten[- ]prädominant\w*)",
    re.IGNORECASE)

# ---------------------------------------------------------------------------
# Befund-Struktur
# ---------------------------------------------------------------------------

# Abschnitts-Überschriften am Zeilenanfang. Reihenfolge unerheblich, das
# erste Vorkommen gewinnt (Mehrfachauszeichnung wie "Diagnose nach
# Paraffineinbettung" wird an "Diagnose" gebunden).
SECTION_HEAD = re.compile(
    r"(?mi)^[ \t]*(?:\d+[.)][ \t]*)?(?P<name>"
    r"Material(?:art)?"
    r"|Klinische\s+Angaben"
    r"|Makroskopie"
    r"|Mikroskopie"
    r"|Diagnose(?:\s+nach\s+Paraffineinbettung)?"
    r"|Kommentar(?:\s*-{1,3}\s*Interpretation)?"
    r"|Tumorklassifikation"
    r"|Zusatzinformationen"
    r"|Zusatzbericht"
    r"|Befund"
    r"|Nachbericht"
    r"|Immunhistochemie"
    r"|Zusammenfassung"
    r"|Schlussfolgerung"
    r"|Nachtrag"
    r")[ \t]*:[ \t]*\n?"
)

CASE_SUFFIX = re.compile(
    r"_(?:Hauptbefund|Zusatzbericht\s*\d*|Nachbericht\s*\d*|Molekularbericht"
    r"|Histo\s+Privat|Histo\s+privat|Immunhistochemie|Zusatzbefund"
    r"|Schnellschnitt(?:durchsage)?|Zytologie|Blockbericht)\b.*$",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """NFC + einheitliche Leerzeichen, damit Regexe stabil greifen."""
    text = unicodedata.normalize("NFC", text or "")
    return text.replace("\u00a0", " ").replace("\u2019", "'")


def split_sections(text: str) -> dict[str, str]:
    """Zerlegt einen Befund in Abschnitte; Keys sind lowercase.

    Mehrfach vorkommende Abschnitte (etwa ``Diagnose`` und ``Nachbericht``)
    werden verkettet, nicht ueberschrieben -- die abschliessende Diagnose
    steht bei diesen Befunden typischerweise im Nachbericht.
    """
    sections: dict[str, str] = {}
    matches = list(SECTION_HEAD.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = re.sub(r"\s+", " ", m.group("name")).strip().lower()
        # "Diagnose nach Paraffineinbettung" ist eine Unterform von "Diagnose"
        if name.startswith("diagnose"):
            name = "diagnose"
        elif name.startswith("kommentar"):
            name = "kommentar"
        sections[name] = sections.get(name, "") + text[m.end():end]
    return sections


# Abschnitte, in denen die abschliessende Tumordiagnose steht.
DIAGNOSIS_SECTIONS = ("diagnose", "nachbericht", "befund", "zusatzbericht")


def diagnosis_text(sections: dict[str, str], text: str) -> str:
    """Die Abschnitte mit der Tumordiagnose, sonst der ganze Befund."""
    parts = [sections[k] for k in DIAGNOSIS_SECTIONS if sections.get(k, "").strip()]
    return "\n".join(parts) if parts else text


def is_negated(text: str, start: int) -> bool:
    """True, wenn kurz vor `start` eine Negation steht ("Keine Nekrosen")."""
    window = text[max(0, start - 60):start]
    return bool(RE_NEGATION_WINDOW.search(window))


# Wie nah ein "Rauschwort" (Fibrose, Sklerose, Einblutung ...) am Treffer
# liegen muss, um ihn zu verwerfen. 25 Zeichen nach dem Treffer deckt
# "knotige Fibrosezonen" und "diffuse Sklerose" ab, trifft aber nicht
# mehr "diffuse Infiltrate mit fibrösem Stroma" weiter hinten im Satz.
NOISE_AHEAD = 25
NOISE_BEHIND = 30


def describes_growth(text: str, start: int) -> bool:
    """True, wenn der Treffer ein neoplastisches Wachstum beschreibt.

    Betrifft die unscharfen Synonyme "knotig"/"diffus": "knotige
    Infiltrate" ja, "knotige Fibrosezonen" nein.
    """
    ahead = text[start:start + NOISE_AHEAD]
    behind = text[max(0, start - NOISE_BEHIND):start]
    if RE_GROWTH_NOISE.search(ahead) or RE_GROWTH_NOISE.search(behind):
        return False
    return bool(RE_GROWTH_NOUN.search(text[
        max(0, start - NOISE_BEHIND):start + NOISE_AHEAD]))


def find_pattern_hits(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    """Alle nicht-verneinten Treffer als (start, Treffertext)."""
    hits: list[tuple[int, str]] = []
    for m in pattern.finditer(text):
        if is_negated(text, m.start()):
            continue
        # Nur unscharfe Synonyme brauchen den Wachstumskontext.
        if RE_IMPLICIT_GROWTH.fullmatch(m.group(0)) and not describes_growth(
                text, m.start()):
            continue
        hits.append((m.start(), m.group(0)))
    return hits


def snippet(text: str, start: int, width: int = 110) -> str:
    """Textausschnitt um einen Treffer, für die Beleg-Spalte."""
    left = max(0, start - width // 3)
    right = min(len(text), start + width)
    out = re.sub(r"\s+", " ", text[left:right]).strip()
    if left > 0:
        out = "…" + out
    if right < len(text):
        out += "…"
    return out


def base_case_id(anonym_case_id: str) -> str:
    """``case-12345_Zusatzbericht 2`` -> ``case-12345``."""
    return CASE_SUFFIX.sub("", anonym_case_id or "").strip()


# ---------------------------------------------------------------------------
# Ergebnis-Datenklassen
# ---------------------------------------------------------------------------

@dataclass
class ReportHit:
    report_id: int
    anonym_case_id: str
    filename: str
    case_id: str                      # Original-Fallnummer (Join-Key)
    anonym_case: str = ""             # Pseudonym, z. B. "case-37566"
    labels: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    lymphoma_terms: list[str] = field(default_factory=list)
    material: str = ""
    provisional_benign: bool = False


@dataclass
class CaseHit:
    case_id: str                      # Original-Fallnummer (Join-Key)
    anonym_case: str = ""             # Pseudonym, z. B. "case-37566"
    report_ids: list[int] = field(default_factory=list)
    report_names: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    lymphoma_terms: list[str] = field(default_factory=list)
    lymphoma_type: str = ""
    provisional_benign: bool = False
    material: str = ""


# Reihenfolge der Präzedenz, wenn ein Fall mehrere Muster zeigt. Nekrose
# zuerst, weil sie im Befund am explizitesten ist; minimal-change zuletzt,
# weil "erhaltene Architektur" auch in diffusen Befunden vorkommt.
LABEL_PRIORITY = ["necrosis", "nodular", "diffuse", "minimal-change"]


def classify_lymphoma_type(diagnose: str) -> str:
    """Grobe Unterteilung wie in der Tabelle des Pathologen: hl/indolent/
    aggressiv. Rein textbasiert, keine WHO-Kodierung."""
    d = diagnose or ""
    if RE_TYPE_HL.search(d):
        return "hl"
    if RE_TYPE_AGGRESSIVE.search(d):
        return "aggressiv"
    if RE_TYPE_INDOLENT.search(d):
        return "indolent"
    return "nhl-unspezifiziert"


# ---------------------------------------------------------------------------
# Kernsuche
# ---------------------------------------------------------------------------

class LymphomaSearch:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.settings = Settings()
        self.conn = psycopg.connect(
            dbname=args.db or self.settings.db_name,
            user=self.settings.db_user,
            password=self.settings.db_password,
            host=self.settings.db_host,
            port=self.settings.db_port,
            row_factory=dict_row,
        )

    def close(self) -> None:
        self.conn.close()

    # -- Kriterien ---------------------------------------------------------

    def reject_reason(self, text: str, sections: dict[str, str]) -> str | None:
        """Gibt den Grund zurück, warum der Befund ausscheidet, sonst None."""
        if not self.args.keep_stanz and RE_STANZ.search(text):
            return "stanz"
        if self.args.exclude_non_malignant and RE_NO_MALIGNITY.search(text):
            return "kein-malignitaetsnachweis"
        if not self._material_ok(text, sections):
            return "material"
        return None

    def _material_ok(self, text: str, sections: dict[str, str]) -> bool:
        """Kriterium 6: Materialart Lymphknoten / LK / LKs.

        Der Pathologe hat in einem strukturierten Feld gefiltert. Im
        Freitext des Befunds steht die Materialart aber nicht immer
        wortgleich: aus einem Lymphknoten in der Leiste wird manchmal
        "Weichgewebsexzisat (Leiste rechts)". Deshalb gibt es zwei Stufen:

        * ``--material-strict`` (Default): nur der Abschnitt ``Material``
          entscheidet, wortgrenze-basiert.
        * ``--material-lenient``: zusaetzlich gilt der Befund als
          Lymphknoten-Material, wenn er in Makroskopie, Mikroskopie oder
          Diagnose Lymphknotengewebe benennt.
        """
        material = sections.get("material")
        if material and material.strip() and RE_MATERIAL_LYMPH_NODE.search(material):
            return True
        if self.args.material_strict or not (material and material.strip()):
            scope = material if material and material.strip() else text
            return bool(RE_MATERIAL_LYMPH_NODE.search(scope))
        body = "\n".join(sections.get(k, "") for k in
                         ("makroskopie", "mikroskopie", "diagnose"))
        return bool(RE_MATERIAL_LYMPH_NODE.search(body))

    def is_lymphoma(self, sections: dict[str, str],
                    text: str) -> tuple[bool, list[str]]:
        scope = diagnosis_text(sections, text)
        terms = sorted({m.group(0).lower() for m in RE_LYMPHOMA.finditer(scope)})
        return bool(terms), terms

    def classify(self, text: str, sections: dict[str, str]) -> dict[str, list[str]]:
        """Kriterien 3-5: Wachstumsmuster mit Belegen.

        Der Suchbereich ist je Muster verschieden (siehe
        ``PATTERN_SCOPE``): ``nodulär``/``diffus``/``nekrotisch`` werden in
        Mikroskopie, Diagnose und Nachbericht gesucht -- der Pathologe
        nennt dafür keine Abschnittsgrenze, und die abschließende
        Beurteilung steht im Nachbericht. ``minimal-change`` wird im
        gesamten Befund gesucht, weil es laut Kriterium 4 auch aus dem
        vorläufigen Befund hervorgeht.

        Fehlt der Abschnitt ``Mikroskopie`` -- er ist nur bei etwa der
        Hälfte der Befunde vorhanden --, wird auf den gesamten Befund
        zurückgefallen.
        """
        wanted = set(self.args.pattern)
        evidence: dict[str, list[str]] = {}
        for name, pattern in PATTERNS.items():
            if name not in wanted:
                continue
            scope = self._pattern_scope(name, text, sections)
            if self.args.negation_filter:
                hits = find_pattern_hits(scope, pattern)
            else:
                hits = [(m.start(), m.group(0)) for m in pattern.finditer(scope)]
            if hits:
                evidence[name] = [snippet(scope, s) for s, _ in hits]
        return evidence

    def _pattern_scope(self, name: str, text: str,
                       sections: dict[str, str]) -> str:
        """Der Text, in dem nach `name` gesucht wird."""
        wanted = PATTERN_SCOPE.get(name, ("report",))
        if "report" in wanted or not self.args.scope_to_mikroskopie:
            return text
        parts = [sections[k] for k in wanted if sections.get(k, "").strip()]
        return "\n".join(parts) if parts else text

    # -- Iteration --------------------------------------------------------

    def iter_reports(self) -> Iterator[dict]:
        """Alle Befunde mit Text, optional auf eine Report-ID-Menge limitiert."""
        sql = ["SELECT report_id, anonym_case_id, original_case_id, filename,",
               "raw_text", "FROM case_report"]
        params: list = []
        if self.args.report_ids:
            sql.append("WHERE report_id = ANY(%s)")
            params.append(self.args.report_ids)
        else:
            sql.append("ORDER BY report_id")
            if self.args.limit:
                sql.append("LIMIT %s")
                params.append(self.args.limit)
        logger.info("Lese Befunde aus %s …", self.settings.db_name)
        # Serverseitiger Cursor: der Befundstext aller 75k Berichte passt
        # nicht bequem in den Client-Speicher.
        with self.conn.cursor(name="lymphoma_scan", row_factory=dict_row) as cur:
            cur.itersize = 2000
            cur.execute(" ".join(sql), params)
            for row in cur:
                yield row

    def run(self) -> tuple[list[CaseHit], Counter]:
        rejected: Counter = Counter()
        report_hits: list[ReportHit] = []
        seen = 0
        for row in self.iter_reports():
            seen += 1
            text = normalize(row["raw_text"])
            sections = split_sections(text)
            reason = self.reject_reason(text, sections)
            if reason:
                rejected[reason] += 1
                continue

            has_lymphoma, terms = self.is_lymphoma(sections, text)
            if self.args.require_lymphoma and not has_lymphoma:
                rejected["kein-lymphom"] += 1
                continue
            if not self.args.require_lymphoma:
                terms = terms if has_lymphoma else []

            evidence = self.classify(text, sections)
            # Kriterium 4, zweiter Weg: der vorlaeufige Befund nannte eine
            # entzuendliche/unspezifische Veraenderung. Das kann im
            # Hauptbefund stehen, waehrend die Lymphomdiagnose erst im
            # Nachbericht folgt -- die Verknuepfung passiert daher erst
            # auf Fall-Ebene in aggregate_cases().
            provisional = bool(RE_PROVISIONAL_BENIGN.search(text))
            flip_candidate = (self.args.minimal_change_flip
                              and provisional and has_lymphoma)
            if not evidence and not flip_candidate:
                rejected["kein-wachstumsmuster"] += 1
                continue

            hit = ReportHit(
                report_id=row["report_id"],
                anonym_case_id=row["anonym_case_id"],
                filename=row["filename"],
                # Join-Key zur manuellen Fallliste. Fuellt ihn
                # scripts/backfill_original_case_id.py; NULL kann nur
                # vorkommen, wenn der Importordner kein Lookup hat.
                case_id=row["original_case_id"] or base_case_id(row["anonym_case_id"]),
                anonym_case=base_case_id(row["anonym_case_id"]),
                labels=list(evidence.keys()),
                evidence=evidence,
                lymphoma_terms=terms,
                material=re.sub(r"\s+", " ", sections.get("material", "")).strip()[:200],
                provisional_benign=provisional,
            )
            report_hits.append(hit)
            if seen % 5000 == 0:
                logger.info("  %d Befunde gelesen, %d Treffer", seen, len(report_hits))

        logger.info("%d Befunde gelesen, %d Befund-Treffer, verworfen: %s",
                    seen, len(report_hits), dict(rejected))
        return aggregate_cases(report_hits, self.args.minimal_change_flip), rejected

    # -- Ergebnis ---------------------------------------------------------

    def materialize(self) -> tuple[list[CaseHit], Counter]:
        """Führt die Suche aus und legt das Ergebnis in der DB ab."""
        cases, rejected = self.run()
        self._persist(cases)
        return cases, rejected

    def _persist(self, cases: list[CaseHit]) -> None:
        """Schreibt die Treffer in eine Hilfstabelle, damit sie in der
        Datenbank nachvollziehbar bleiben (``CREATE TABLE IF NOT EXISTS``).

        Die Tabelle wird bei jedem Lauf geleert und neu befuellt; sie
        enthaelt nur Suchergebnisse und keine Quelldaten.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lymphoma_pattern_case (
                    case_id            TEXT PRIMARY KEY,
                    anonym_case        TEXT,
                    labels             TEXT[]   NOT NULL,
                    report_ids         BIGINT[] NOT NULL,
                    lymphoma_type      TEXT,
                    provisional_benign BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
                )""")
            # Aeltere Laeufe hatten die Spalte anonym_case noch nicht.
            cur.execute("ALTER TABLE lymphoma_pattern_case "
                        "ADD COLUMN IF NOT EXISTS anonym_case TEXT")
            cur.execute("TRUNCATE lymphoma_pattern_case")
            for c in cases:
                cur.execute(
                    "INSERT INTO lymphoma_pattern_case "
                    "(case_id, anonym_case, labels, report_ids, lymphoma_type, "
                    " provisional_benign) VALUES (%s, %s, %s, %s, %s, %s)",
                    (c.case_id, c.anonym_case, c.labels, c.report_ids,
                     c.lymphoma_type, c.provisional_benign))
        self.conn.commit()
        logger.info("Ergebnis in Tabelle lymphoma_pattern_case geschrieben (%d Fälle)",
                    len(cases))


def aggregate_cases(reports: list[ReportHit],
                    minimal_change_flip: bool = False) -> list[CaseHit]:
    """Fasst Befund-Treffer zu Fällen zusammen.

    ``minimal_change_flip`` bildet Kriterium 4, zweiter Weg ab: der
    vorläufige Befund eines Falls nannte eine entzündliche oder
    unspezifische Veränderung, in einem anderen Befund desselben Falls
    (Zusatz-/Nachbericht) steht dann die Lymphomdiagnose. Diese
    Verknüpfung ist nur auf Fall-Ebene sichtbar.
    """
    by_case: dict[str, CaseHit] = {}
    for r in reports:
        c = by_case.setdefault(r.case_id, CaseHit(case_id=r.case_id))
        c.anonym_case = c.anonym_case or r.anonym_case
        c.material = c.material or r.material
        if r.report_id not in c.report_ids:
            c.report_ids.append(r.report_id)
            c.report_names.append(r.filename or r.anonym_case_id)
        for label in r.labels:
            if label not in c.labels:
                c.labels.append(label)
            c.evidence.setdefault(label, [])
            for ev in r.evidence.get(label, []):
                if ev not in c.evidence[label]:
                    c.evidence[label].append(ev)
        for t in r.lymphoma_terms:
            if t not in c.lymphoma_terms:
                c.lymphoma_terms.append(t)
        c.provisional_benign = c.provisional_benign or r.provisional_benign

    for c in by_case.values():
        if (minimal_change_flip and c.provisional_benign and c.lymphoma_terms
                and "minimal-change" not in c.labels):
            c.labels.append("minimal-change")
            c.evidence.setdefault(
                "minimal-change",
                ["Vorläufiger Befund entzündlich/unspezifisch, "
                 "Abschlussdiagnose Lymphom"])
        c.labels.sort(key=lambda x: LABEL_PRIORITY.index(x)
                      if x in LABEL_PRIORITY else 99)
        c.report_ids.sort()
        c.report_names = [
            n for _, n in sorted(zip(c.report_ids, c.report_names))]
        c.lymphoma_type = classify_lymphoma_type(" ".join(c.lymphoma_terms))
    return sorted(by_case.values(), key=lambda c: c.case_id)


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["case_id", "anonym_case", "labels", "labels_de", "lymphoma_type",
               "n_reports", "report_ids", "report_files", "material",
               "lymphoma_terms", "provisional_benign",
               "evidence_nodular", "evidence_diffuse",
               "evidence_minimal-change", "evidence_necrosis"]


def case_to_row(c: CaseHit) -> dict:
    return {
        "case_id": c.case_id,
        "anonym_case": c.anonym_case,
        "labels": ";".join(c.labels),
        "labels_de": ";".join(PATTERN_LABELS_DE.get(x, x) for x in c.labels),
        "lymphoma_type": c.lymphoma_type,
        "n_reports": len(c.report_ids),
        "report_ids": ";".join(str(r) for r in c.report_ids),
        "report_files": ";".join(c.report_names),
        "material": c.material,
        "lymphoma_terms": ";".join(c.lymphoma_terms),
        "provisional_benign": c.provisional_benign,
        **{f"evidence_{k}": " || ".join(c.evidence.get(k, [])[:3])
           for k in ("nodular", "diffuse", "minimal-change", "necrosis")},
    }


def write_outputs(cases: list[CaseHit], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lymphoma_pattern_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for c in cases:
            w.writerow(case_to_row(c))
    json_path = out_dir / "lymphoma_pattern_cases.json"
    json_path.write_text(
        json.dumps([case_to_row(c) for c in cases], ensure_ascii=False, indent=2),
        encoding="utf-8")
    logger.info("CSV  -> %s", csv_path)
    logger.info("JSON -> %s", json_path)


# ---------------------------------------------------------------------------
# Arbeitsmappe fuer den Pathologen
# ---------------------------------------------------------------------------

WORKBOOK_HEADERS = {
    "case_id": "Fallnummer (original)",
    "anonym_case": "Fallnummer (pseudonymisiert)",
    "labels_de": "Wachstumsmuster",
    "labels": "Wachstumsmuster (Codes)",
    "lymphoma_type": "Lymphom-Typ (heuristisch)",
    "n_reports": "Anzahl Befunde",
    "report_files": "Befunde",
    "report_ids": "report_id",
    "material": "Material (Abschnitt Material)",
    "lymphoma_terms": "Lymphom-Begriffe im Befund",
    "provisional_benign": "Vorlaeufiger Befund entzuendlich/unspezifisch",
    "evidence_nodular": "Beleg: nodulaer",
    "evidence_diffuse": "Beleg: diffus",
    "evidence_minimal-change": "Beleg: minimal-change",
    "evidence_necrosis": "Beleg: nekrotisch",
}

WORKBOOK_WIDTHS = {
    "case_id": 20, "anonym_case": 22, "labels_de": 26, "labels": 26,
    "lymphoma_type": 20, "n_reports": 14, "report_files": 40,
    "report_ids": 18, "material": 34, "lymphoma_terms": 30,
    "provisional_benign": 26,
    "evidence_nodular": 70, "evidence_diffuse": 70,
    "evidence_minimal-change": 70, "evidence_necrosis": 70,
}

COMPARISON_HEADERS = {
    "case_id_manual": "Fallnummer (manuell)",
    "category": "Typ (manuell)",
    "subcategory": "Muster (manuell)",
    "n_nodes": "Lymphknoten (manuell)",
    "found": "automatisch gefunden",
    "anonym_case": "Fallnummer (pseudonymisiert)",
    "labels_de": "Wachstumsmuster (automatisch)",
    "label_match": "Muster stimmt ueberein",
    "review_pattern": "Muster laut Pathologe (bitte eintragen)",
    "review_keep": "Fall relevant? (bitte eintragen)",
    "review_note": "Bemerkung",
    "quote": "Zitat aus dem Befundtext (manuell)",
}

COMPARISON_WIDTHS = {
    "case_id_manual": 20, "category": 14, "subcategory": 16, "n_nodes": 18,
    "found": 20, "anonym_case": 22, "labels_de": 26, "label_match": 22,
    "review_pattern": 26, "review_keep": 22, "review_note": 34,
    "quote": 90,
}


def _write_sheet(ws, rows: list[dict], headers: dict[str, str],
                 widths: dict[str, int]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    keys = list(headers)
    ws.append([headers[k] for k in keys])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
        cell.alignment = Alignment(vertical="top")
    for r in rows:
        ws.append([r.get(k, "") for k in keys])
    for i, k in enumerate(keys, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(k, 18)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def write_workbook(cases: list[CaseHit], out_path: Path,
                   comparison: list[dict] | None = None,
                   rejected: Counter | None = None) -> None:
    """Arbeitsmappe, die der Pathologe durchsehen und abzeichnen kann.

    Blaetter:
      * ``gefundene_faelle`` -- alle automatisch gefundenen Faelle mit
        den Beleg-Stellen, an denen das Wachstumsmuster erkannt wurde.
      * ``vergleich_manual`` -- seine manuelle Liste mit dem Ergebnis der
        automatischen Suche, mit leeren Spalten zum Eintragen.
      * ``uebersicht`` -- angewandte Kriterien, Verteilung der Muster und
        die Schritte, die zum Ausschluss von Befunden gefuehrt haben.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()

    ws = wb.active
    ws.title = "gefundene_faelle"
    _write_sheet(ws, [case_to_row(c) for c in cases],
                 WORKBOOK_HEADERS, WORKBOOK_WIDTHS)

    if comparison is not None:
        ws = wb.create_sheet("vergleich_manual")
        _write_sheet(ws, comparison, COMPARISON_HEADERS, COMPARISON_WIDTHS)

    ws = wb.create_sheet("uebersicht")
    ws.append(["Automatische Suche nach Lymphom-Wachstumsmustern"])
    ws["A1"].font = Font(bold=True, size=13)
    ws.append([])
    ws.append(["Angewandte Kriterien des Pathologen"])
    ws["A4"].font = Font(bold=True)
    for text in [
        "1  keine 'Stanz*'-Befunde (nur ganze Lymphknoten)",
        "2  keine 'Kein Anhalt/Nachweis von Malignitaet'-Befunde",
        "3  nodulaer = nodulaer/noedos/knotig, diffus = diffus/konfluierend",
        "4  minimal-change = erhaltene Architektur bzw. vorlaeufiger Befund "
        "entzuendlich/unspezifisch und Abschlussdiagnose Lymphom",
        "5  nekrotisch = Nekrose/Nekroseherd/nekrotisches Areal; gesucht in "
        "Mikroskopie, Diagnose und Nachbericht",
        "6  Materialart: Lymphknoten / LK / LKs (Wortgrenze)",
        "   zusaetzlich: Diagnose muss eine Lymphom-Bezeichnung nennen",
        "   Treffer mit 'Keine Nekrosen' o. Ae. werden nicht gezaehlt "
        "(Negation)",
    ]:
        ws.append([text])
    ws.append([])
    ws.append(["Verteilung der Wachstumsmuster (Mehrfachnennung moeglich)"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    ws.append(["Fallzahl gesamt", len(cases)])
    labels: Counter = Counter()
    for c in cases:
        for l in c.labels:
            labels[l] += 1
    for key in LABEL_PRIORITY:
        ws.append([PATTERN_LABELS_DE[key], labels.get(key, 0)])
    ws.append([])
    ws.append(["Verteilung der Lymphom-Typen (heuristisch aus dem Text)"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    types: Counter = Counter(c.lymphoma_type or "nhl-unspezifiziert"
                             for c in cases)
    for key, n in types.most_common():
        ws.append([key, n])
    if rejected:
        ws.append([])
        ws.append(["Verworfene Befunde nach Grund"])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
        for key, n in rejected.most_common():
            ws.append([key, n])
    ws.column_dimensions["A"].width = 95
    ws.column_dimensions["B"].width = 14
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    logger.info("XLSX -> %s", out_path)


def print_summary(cases: list[CaseHit], rejected: Counter) -> None:
    labels = Counter()
    types = Counter()
    for c in cases:
        for l in c.labels:
            labels[l] += 1
        types[c.lymphoma_type or "nhl-unspezifiziert"] += 1

    print("\n=== Automatische Suche: Lymphom-Wachstumsmuster ===")
    print(f"Gefundene Fälle: {len(cases)}")
    print("\nFälle je Wachstumsmuster (Mehrfachnennung möglich):")
    for key in LABEL_PRIORITY:
        print(f"  {PATTERN_LABELS_DE[key]:<16} {labels.get(key, 0):>6}")
    print("\nFälle je Lymphom-Typ (Text heuristisch):")
    for key, n in types.most_common():
        print(f"  {key:<22} {n:>6}")
    print("\nVerworfene Befunde nach Grund:")
    for key, n in rejected.most_common():
        print(f"  {key:<26} {n:>6}")


# ---------------------------------------------------------------------------
# Vergleich mit der manuellen Liste des Pathologen
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> list[dict]:
    """Liest die Excel-Liste des Pathologen (Spalten case ID/…/subcategory)."""
    import openpyxl

    ws = openpyxl.load_workbook(path)["caseids"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[0]:
            continue
        rows.append({
            "case_id_manual": str(r[0]).strip(),
            "category": (r[2] or "").strip().lower(),
            "subcategory": (r[3] or "").strip().lower(),
            "quote": r[4] or "",
            "n_nodes": r[5],
        })
    return rows


SUBCATEGORY_TO_LABEL = {
    "nodular": "nodular",
    "diffuse": "diffuse",
    "minimal-change": "minimal-change",
    "necrosis": "necrosis",
}


def compare_with_ground_truth(cases: list[CaseHit],
                              truth: list[dict]) -> list[dict]:
    """Vergleicht die manuelle Liste mit der automatischen Suche.

    Der Join erfolgt exakt ueber die Original-Fallnummer
    (``case_report.original_case_id``, gefuellt aus
    ``look_up_table.json``). Nur Faelle, die es in der Datenbank ueberhaupt
    gibt, koennen verglichen werden -- ``patientdb_scarif`` enthaelt nur
    den Import ``2024_anonymized``, das Lookup kennt daher ausschliesslich
    Fallnummern des Jahrgangs 2024. Alle uebrigen manuellen Faelle werden
    als ``nicht_in_datenbank`` ausgewiesen, statt als Fehltreffer.
    """
    auto_by_case = {c.case_id: c for c in cases}
    results = []
    for t in truth:
        manual_label = SUBCATEGORY_TO_LABEL.get(t["subcategory"], "")
        hit = auto_by_case.get(t["case_id_manual"])
        if t["case_id_manual"] in auto_by_case:
            status = "gefunden"
        else:
            # Existiert der Fall in der DB, wurde er nur nicht selektiert?
            status = ("nicht_in_datenbank"
                      if not t["case_id_manual"].startswith("E/2024/")
                      else "nicht_gefunden")
        auto_labels = hit.labels if hit else []
        results.append({
            **t,
            "found": status,
            "anonym_case": hit.anonym_case if hit else "",
            "labels_de": ";".join(PATTERN_LABELS_DE.get(x, x)
                                  for x in auto_labels),
            "label_match": "ja" if (hit and manual_label in auto_labels)
                           else ("nein" if hit else ""),
            "quote": t["quote"],
        })
    return results


def print_comparison(results: list[dict], cases: list[CaseHit]) -> None:
    comparable = [r for r in results if r["found"] != "nicht_in_datenbank"]
    in_db = [r for r in results if r["found"] != "nicht_in_datenbank"]
    found = [r for r in results if r["found"] == "gefunden"]
    missed = [r for r in results if r["found"] == "nicht_gefunden"]
    absent = [r for r in results if r["found"] == "nicht_in_datenbank"]

    print("\n=== Vergleich mit der manuellen Liste des Pathologen ===")
    print(f"Manuelle Zeilen:                  {len(results)}")
    print(f"Davon in der Datenbank vorhanden: {len(in_db)}")
    print(f"  davon automatisch gefunden:      {len(found)}")
    print(f"  davon nicht gefunden:            {len(missed)}")
    print(f"Nicht in der Datenbank "
          f"(anderer Jahrgang als 2024):    {len(absent)}")

    if in_db:
        agree = sum(1 for r in in_db if r["label_match"] == "ja")
        print(f"\nMuster stimmt überein: {agree}/{len(in_db)} "
              f"(ohne Treffer: {sum(1 for r in in_db if not r['label_match'])})")
        print("\nJe Kategorie (nur vergleichbare Fälle):")
        for key in ("nodular", "diffuse", "minimal-change", "necrosis"):
            rows = [r for r in in_db
                    if SUBCATEGORY_TO_LABEL.get(r["subcategory"], "") == key]
            ok = sum(1 for r in rows if r["label_match"] == "ja")
            hit = sum(1 for r in rows if r["found"] == "gefunden")
            print(f"  {PATTERN_LABELS_DE[key]:<16} "
                  f"gefunden {hit}/{len(rows)}  Muster korrekt {ok}/{len(rows)}")

    print("\nVergleichbare Fälle im Detail:")
    for r in sorted(comparable, key=lambda x: x["case_id_manual"]):
        print(f"  {r['case_id_manual']:<16} {r['subcategory']:<15} "
              f"{r['found']:<18} {r['anonym_case']:<12} "
              f"auto={r['labels_de'] or '-':<28} "
              f"muster={'ok' if r['label_match'] == 'ja' else r['label_match'] or '-'}")

    if missed:
        print("\nIn der Datenbank, aber nicht gefunden:")
        for r in missed:
            print(f"  {r['case_id_manual']:<16} {r['subcategory']:<15} "
                  f"Pseudonym={r['anonym_case'] or '-'}")

    auto_ids = {c.case_id for c in cases}
    extra = sorted(auto_ids - {r["case_id_manual"] for r in in_db})
    print(f"\nAutomatisch gefunden, manuell nicht gelistet "
          f"(nur Jahrgang 2024): {len(extra)}")
    for cid in extra[:30]:
        print(f"  {cid}")
    if len(extra) > 30:
        print(f"  ... und {len(extra) - 30} weitere")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Lymphom-Befunde nach Wachstumsmuster suchen "
                    "(Kriterien des Pathologen).")
    ap.add_argument("--db", default=None,
                    help="Datenbankname (Default: DB_NAME aus .env)")
    ap.add_argument("--out", default="results/lymphoma_search",
                    help="Ausgabeordner fuer CSV/JSON")
    ap.add_argument("--pattern", nargs="+", default=list(PATTERNS.keys()),
                    choices=list(PATTERNS.keys()),
                    help="Welche Wachstumsmuster beruecksichtigt werden")
    ap.add_argument("--limit", type=int, default=0,
                    help="Nur die ersten N Befunde lesen (0 = alle)")
    ap.add_argument("--report-ids", type=int, nargs="*", default=None,
                    help="Nur diese report_id lesen (fuer Stichproben)")
    ap.add_argument("--keep-stanz", action="store_true",
                    help="Kriterium 1 abschalten: Stanz*-Befunde behalten")
    ap.add_argument("--material-strict", action="store_true", default=True,
                    help="Kriterium 6 strikt: nur der Abschnitt Material "
                         "entscheidet (Default an)")
    ap.add_argument("--material-lenient", dest="material_strict",
                    action="store_false",
                    help="Kriterium 6 auch dann erfuellt, wenn Makroskopie, "
                         "Mikroskopie oder Diagnose Lymphknotengewebe nennen")
    ap.add_argument("--exclude-non-malignant", action="store_true", default=True,
                    help="Kriterium 2: 'Kein Anhalt/Nachweis von Malignitaet' "
                         "ausschliessen (Default an)")
    ap.add_argument("--keep-non-malignant", dest="exclude_non_malignant",
                    action="store_false",
                    help="Kriterium 2 abschalten")
    ap.add_argument("--require-lymphoma", action="store_true", default=True,
                    help="Lymphom-Diagnose verlangen (Default an)")
    ap.add_argument("--no-require-lymphoma", dest="require_lymphoma",
                    action="store_false",
                    help="Lymphom-Diagnose nicht verlangen")
    ap.add_argument("--scope-to-mikroskopie", action="store_true", default=True,
                    help="nodulaer/diffus/nekrotisch nur in Mikroskopie, "
                         "Diagnose und Nachbericht suchen (Default an); "
                         "minimal-change wird im gesamten Befund gesucht")
    ap.add_argument("--whole-text", dest="scope_to_mikroskopie",
                    action="store_false",
                    help="alle Muster im gesamten Befund suchen")
    ap.add_argument("--negation-filter", action="store_true", default=True,
                    help="'Keine Nekrosen' etc. als Negation werten (Default an)")
    ap.add_argument("--no-negation-filter", dest="negation_filter",
                    action="store_false", help="Negation nicht auswerten")
    ap.add_argument("--minimal-change-flip", action="store_true",
                    help="Kriterium 4, zweiter Weg: vorlaeufiger Befund "
                         "entzuendlich/unspezifisch, Abschluss ein Lymphom")
    ap.add_argument("--no-persist", action="store_true",
                    help="Ergebnis nicht in lymphoma_pattern_case schreiben")
    ap.add_argument("--ground-truth", default=None,
                    help="Excel des Pathologen fuer den Vergleich")
    ap.add_argument("--workbook", default=None,
                    help="Pfad der Arbeitsmappe fuer den Pathologen "
                         "(Default: <out>/lymphoma_funde.xlsx)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)

    setup_logging(log_dir=args.log_dir, level=args.log_level)
    args.out = Path(args.out)

    search = LymphomaSearch(args)
    try:
        if args.no_persist:
            cases, rejected = search.run()
        else:
            cases, rejected = search.materialize()
        print_summary(cases, rejected)
        write_outputs(cases, args.out)

        comparison = None
        if args.ground_truth:
            truth = load_ground_truth(Path(args.ground_truth))
            comparison = compare_with_ground_truth(cases, truth)
            (args.out / "comparison.json").write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print_comparison(comparison, cases)
            logger.info("Vergleich -> %s", args.out / "comparison.json")

        workbook = Path(args.workbook) if args.workbook else (
            args.out / "lymphoma_funde.xlsx")
        write_workbook(cases, workbook, comparison, rejected)
    finally:
        search.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
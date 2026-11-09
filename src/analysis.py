#!/usr/bin/env python3
"""Analyse der Extraktionsergebnisse.

Auswertung auf zwei Ebenen:
  A) Strukturierung einzelner Berichte (Extraktion, Mapping-Rate, Labels)
  B) Vernetzung der Berichte untereinander (geteilte Konzepte,
     gemeinsame Ancestors, Ähnlichkeitspaare)

Aufruf:
    python3 src/analysis.py
    python3 src/analysis.py --min-shared 2      # mind. N gemeinsame Konzepte für Vernetzung
    python3 src/analysis.py --unmatched-report  # Liste nicht gemappter Begriffe (für Dictionary-Pflege)
"""

import argparse
import logging
import os
from collections import Counter, defaultdict

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


# ---------------------------------------------------------------------------
# A) Strukturierung pro Bericht
# ---------------------------------------------------------------------------

def analyse_reports(cur) -> dict[int, dict]:
    """Sammelt pro Bericht: Textlänge, Entitäten, Mapping-Status, Labels."""
    cur.execute("""
        SELECT rc.report_id,
               rc.report_concept_id,
               LENGTH(cr.raw_text)        AS text_length,
               rc.surface_text,
               rc.concept_id,
               mc.preferred_term
        FROM report_concept rc
        JOIN case_report cr ON cr.report_id = rc.report_id
        LEFT JOIN medical_concept mc ON mc.concept_id = rc.concept_id
        ORDER BY rc.report_id
    """)
    rows = cur.fetchall()

    reports: dict[int, dict] = {}
    for (report_id, case_id, text_len, entity_text, concept_id, pt) in rows:
        r = reports.setdefault(report_id, {
            "case_id": case_id,
            "text_length": text_len,
            "entities": [],
            "mapped": 0,
            "unmapped": [],
        })
        r["entities"].append({
            "text": entity_text, "concept_id": concept_id,
            "preferred_term": pt,
        })
        if concept_id:
            r["mapped"] += 1
        else:
            r["unmapped"].append(entity_text)
    return reports


def print_report_analysis(reports: dict[int, dict]) -> None:
    print("\n" + "=" * 78)
    print("A) STRUKTURIERUNG PRO BERICHT")
    print("=" * 78)

    total_entities = total_mapped = 0
    global_unmapped: Counter = Counter()

    for report_id, r in sorted(reports.items()):
        n = len(r["entities"])
        mapped = r["mapped"]
        total_entities += n
        total_mapped += mapped
        global_unmapped.update(r["unmapped"])

        print(f"\n--- Report {report_id} ({r['case_id']}) | "
              f"{r['text_length']} Zeichen | {n} Entitäten, {mapped} gemappt "
              f"({100*mapped/n:.0f}%)" if n else f"Report {report_id}: keine Entitäten")
        if r["unmapped"]:
            print(f"    NICHT gemappt: {', '.join(sorted(set(r['unmapped'])))}")

    print("\n" + "-" * 78)
    print("GESAMT:")
    print(f"  Berichte:            {len(reports)}")
    print(f"  Entitäten gesamt:    {total_entities}")
    print(f"  Davon gemappt:       {total_mapped} "
          f"({100*total_mapped/max(total_entities,1):.1f}%)")
    print(f"\n  Häufigste nicht gemappte Begriffe "
          f"(Kandidaten für Dictionary/Synonym-Pflege):")
    for term, count in global_unmapped.most_common(20):
        print(f"    {count:3d}x  {term}")


# ---------------------------------------------------------------------------
# B) Vernetzung zwischen Berichten
# ---------------------------------------------------------------------------

def report_concept_sets(cur) -> dict[int, set[int]]:
    """Menge gemappter SNOMED-Codes pro Bericht."""
    cur.execute("""
        SELECT report_id, concept_id
        FROM report_concept
        WHERE concept_id IS NOT NULL
    """)
    sets: dict[int, set[int]] = defaultdict(set)
    for report_id, concept_id in cur.fetchall():
        sets[report_id].add(concept_id)   # int direkt verwenden
    return sets


def analyse_direct_overlap(concept_sets: dict[int, set[str]],
                           cur, min_shared: int) -> list[tuple]:
    """Berichtspaare mit gemeinsamen Konzepten (Jaccard-Ähnlichkeit)."""
    report_ids = sorted(concept_sets)
    pairs = []
    for i, a in enumerate(report_ids):
        for b in report_ids[i + 1:]:
            shared = concept_sets[a] & concept_sets[b]
            if len(shared) < min_shared:
                continue
            union = concept_sets[a] | concept_sets[b]
            jaccard = len(shared) / len(union) if union else 0
            pairs.append((a, b, shared, jaccard))
    pairs.sort(key=lambda p: -p[3])

    if not pairs:
        return pairs

    # PTs der geteilten Konzepte auflösen
    all_shared = {c for _, _, shared, _ in pairs for c in shared}
    cur.execute("""
        SELECT concept_id, preferred_term FROM medical_concept
        WHERE concept_id = ANY(%s::bigint[])
    """, (list(all_shared),))
    terms = {cid: pt for cid, pt in cur.fetchall()}   # keys sind int
	
    print("\n" + "=" * 78)
    print(f"B1) DIREKT GETEILTE KONZEPTE (min. {min_shared} gemeinsam)")
    print("=" * 78)
    for a, b, shared, jaccard in pairs:
        print(f"\n  Report {a} <-> Report {b}: {len(shared)} gemeinsam, "
              f"Jaccard={jaccard:.2f}")
        for cid in sorted(shared):
            print(f"      {cid:>10}  {terms.get(cid, '?')}")
    return pairs

def analyse_ancestor_network(cur) -> None:
    """Vernetzung über gemeinsame Ancestors (Oberbegriffe).

    Schema: concept_relation(source_concept_id = Kind, target_concept_id = Ancestor,
                             relation_type = 'is_a', distance = Stufen)
    Beispiel: Zwei Berichte teilen kein Konzept, aber beide Konzepte
    hängen unter demselben Ancestor ('Finding of cervix' etc.).
    """
    cur.execute("""
        SELECT rc.report_id,
               rel.target_concept_id        AS ancestor_id,
               anc.preferred_term           AS ancestor_pt,
               COUNT(DISTINCT rc.concept_id) AS n_descendants,
               MIN(rel.distance)            AS min_distance
        FROM report_concept rc
        JOIN concept_relation rel
          ON rel.source_concept_id = rc.concept_id
         AND rel.relation_type = 'is_a'
        JOIN medical_concept anc
          ON anc.concept_id = rel.target_concept_id
        WHERE rc.concept_id IS NOT NULL
        GROUP BY rc.report_id, rel.target_concept_id, anc.preferred_term
    """)
    # ancestor_id -> {report_id: (pt, n_descendants)}
    ancestor_map: dict[int, dict[int, tuple]] = defaultdict(dict)
    for report_id, ancestor_id, ancestor_pt, n, _min_dist in cur.fetchall():
        ancestor_map[ancestor_id][report_id] = (ancestor_pt, n)

    print("\n" + "=" * 78)
    print("B2) VERNETZUNG ÜBER GEMEINSAME ANCESTORS")
    print("=" * 78)
    print("\nAncestors, die in >= 3 Berichten vorkommen (thematische Cluster):\n")
    rows = [(aid, reports) for aid, reports in ancestor_map.items()
            if len(reports) >= 3]
    rows.sort(key=lambda x: -len(x[1]))
    if not rows:
        print("  (keine Ancestors in >= 3 Berichten gefunden)")
    for ancestor_id, reports in rows:
        pt = next(iter(reports.values()))[0]
        detail = ", ".join(
            f"R{rid}({n})" for rid, (pt_, n) in sorted(reports.items()))
        print(f"  {len(reports):2d} Berichte | {ancestor_id:>10} {pt}")
        print(f"      └─ {detail}")


def analyse_unmapped_overlap(reports: dict[int, dict]) -> None:
    """Nicht gemappte Begriffe, die in mehreren Berichten auftauchen –
    besonders wertvoll, da systematische Lücken."""
    term_reports: dict[str, set[int]] = defaultdict(set)
    for report_id, r in reports.items():
        for term in set(r["unmapped"]):
            term_reports[term].add(report_id)

    recurring = {t: ids for t, ids in term_reports.items() if len(ids) > 1}
    if not recurring:
        return
    print("\n" + "=" * 78)
    print("B3) WIEDERKEHRENDE NICHT GEMAPPTE BEGRIFFE (systematische Lücken)")
    print("=" * 78)
    for term, ids in sorted(recurring.items(), key=lambda x: -len(x[1])):
        print(f"  {len(ids):2d} Berichte (R{', R'.join(map(str, sorted(ids)))}): {term}")


def print_summary(reports: dict[int, dict],
                  concept_sets: dict[int, set[str]]) -> None:
    mapped_concepts = set().union(*concept_sets.values()) if concept_sets else set()
    print("\n" + "=" * 78)
    print("ZUSAMMENFASSUNG / PLAUSIBILITÄTSCHECK")
    print("=" * 78)
    n_entities = sum(len(r["entities"]) for r in reports.values())
    n_mapped_refs = sum(r["mapped"] for r in reports.values())
    print(f"  Extrahierte Entitäts-Referenzen:   {n_entities}")
    print(f"  Gemappte Referenzen:               {n_mapped_refs}")
    print(f"  DISTINCT SNOMED-Konzepte:          {len(mapped_concepts)}")
    print(f"\n  Erklärung für die Diskrepanz:")
    print(f"  {n_mapped_refs} gemappte Referenzen verteilen sich auf nur")
    print(f"  {len(mapped_concepts)} verschiedene Konzepte -> Konzepte werden")
    print(f"  berichtsübergreifend wiederverwendet (erwartetes Verhalten bei")
    print(f"  ähnlichen Befunden, z. B. alle Zervix-Biopsien).")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-shared", type=int, default=1,
                        help="Min. gemeinsame Konzepte für Berichtspaare (default: 1)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            reports = analyse_reports(cur)
            print_report_analysis(reports)

            concept_sets = report_concept_sets(cur)
            analyse_direct_overlap(concept_sets, cur, args.min_shared)
            analyse_ancestor_network(cur)
            analyse_unmapped_overlap(reports)

            print_summary(reports, concept_sets)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
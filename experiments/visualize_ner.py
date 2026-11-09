#!/usr/bin/env python3
"""
Vergleicht die Ergebnisse der drei NER-Ansätze (MedCAT, OpenMed, Hybrid-Pipeline)
anhand der JSONL-Exporte und erzeugt Diagramme:

  1. Violinplot: Anzahl Entitäten je Berichtstyp und Methode
  2. Zeichenabdeckung je Berichtstyp (Mittelwert der Summe der Spans)
  3. Überlappung der Annotationen: Anteil der von einem Ansatz abgedeckten
     Zeichenpositionen, die auch von einem anderen Ansatz abgedeckt werden.

Läuft vom Projektroot, z. B.:
    python experiments/visualize_ner.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APPROACHES = [
    ("MedCAT (SNOMED)", "experiments/medcat_ner.jsonl", "medcat"),
    ("OpenMed", "experiments/openmed_ner.jsonl", "openmed"),
    ("Hybrid-Pipeline", "experiments/hybrid_ner.jsonl", "hybrid"),
]


def lade_entitaeten(path: Path, fmt: str) -> dict[int, list[dict]]:
    """Lädt je report_id die Entitäten als einheitliche dicts
    (start, end, text, label)."""

    ergebnisse: dict[int, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            report_id = int(record["meta"]["report_id"])
            spans = []

            if fmt == "medcat":
                for e in record["analyse"].get("entities", {}).values():
                    spans.append({
                        "start": e["start"],
                        "end": e["end"],
                        "text": e["source_value"],
                        "label": ", ".join(e.get("type_ids", [])) or "unknown",
                    })
            elif fmt == "openmed":
                # "extract" liefert strukturierte PII-Entitäten mit Labels
                for e in record.get("extract", []):
                    spans.append({
                        "start": e["start"],
                        "end": e["end"],
                        "text": e["text"],
                        "label": e.get("label", "unknown"),
                    })
            elif fmt == "hybrid":
                for e in record.get("spans", []):
                    spans.append({
                        "start": e["start"],
                        "end": e["end"],
                        "text": e["text"],
                        "label": e.get("label") or "unknown",
                    })

            ergebnisse[report_id] = spans
    return ergebnisse


def zeichenmenge(spans: list[dict]) -> set[int]:
    """Menge aller von den Spans abgedeckten Zeichenpositionen."""
    chars: set[int] = set()
    for s in spans:
        chars.update(range(s["start"], s["end"]))
    return chars


def main():

    parser = argparse.ArgumentParser(
        description="Vergleichsplots für MedCAT, OpenMed und Hybrid-NER."
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("experiments/ner_comparison.png"),
                        help="Ausgabe-PNG (default: experiments/ner_comparison.png)")
    args = parser.parse_args()

    # Testset-Liste für Berichtstypen laden
    df = pd.read_excel("experiments/berichte.xlsx")
    berichtstyp = {int(r.report_id): r.berichtstyp for r in df.itertuples(index=False)}

    alle: dict[str, dict[int, list[dict]]] = {}
    for name, path, fmt in APPROACHES:
        if not Path(path).exists():
            print(f"WARN: {path} nicht gefunden – {name} wird übersprungen")
            continue
        alle[name] = lade_entitaeten(Path(path), fmt)
        print(f"{name}: {sum(len(v) for v in alle[name].values())} Entitäten "
              f"in {len(alle[name])} Berichten")

    if not alle:
        print("Keine Eingabedateien gefunden.")
        return

    # Berichtstypen in konsistenter Reihenfolge
    typen = sorted({berichtstyp.get(rid, "unbekannt")
                    for data in alle.values() for rid in data})
    print(f"Berichtstypen: {typen}")

    # Zeichenmengen je Ansatz und Bericht
    zeichen = {name: {rid: zeichenmenge(spans) for rid, spans in data.items()}
               for name, data in alle.items()}

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    colors = plt.cm.tab10.colors

    # --- Plot 1: Violinplot Entitäten pro Berichtstyp
    ax = axes[0]
    for idx, (name, data) in enumerate(alle.items()):
        counts_pro_typ = []
        for typ in typen:
            counts = [len(spans) for rid, spans in data.items()
                      if berichtstyp.get(rid) == typ]
            counts_pro_typ.append(counts)
        positions = np.arange(len(typen)) + (idx - len(alle) / 2 + 0.5) * 0.25
        vp = ax.violinplot(counts_pro_typ, positions=positions, widths=0.25,
                           showmeans=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(colors[idx % 10])
            body.set_alpha(0.6)
        vp["cmeans"].set_color(colors[idx % 10])
        # Legende manuell
        ax.scatter([], [], color=colors[idx % 10], label=name, s=60)
    ax.set_xticks(np.arange(len(typen)))
    ax.set_xticklabels(typen, rotation=20, ha="right")
    ax.set_ylabel("Anzahl Entitäten pro Bericht")
    ax.set_title("Entitäten pro Berichtstyp und Methode")
    ax.legend()

    # --- Plot 2: Zeichenabdeckung je Berichtstyp
    ax = axes[1]
    width = 0.8 / len(alle)
    for idx, (name, data) in enumerate(alle.items()):
        means = []
        for typ in typen:
            covered = [sum(s["end"] - s["start"] for s in spans)
                       for rid, spans in data.items()
                       if berichtstyp.get(rid) == typ]
            means.append(np.mean(covered) if covered else 0)
        positions = np.arange(len(typen)) - 0.4 + width / 2 + idx * width
        ax.bar(positions, means, width=width, label=name, color=colors[idx % 10])
    ax.set_xticks(np.arange(len(typen)))
    ax.set_xticklabels(typen, rotation=20, ha="right")
    ax.set_ylabel("Mittlere abgedeckte Zeichen pro Bericht")
    ax.set_title("Zeichenabdeckung pro Berichtstyp (Mittelwert)")
    ax.legend()

    # --- Plot 3: Überlappung der Zeichenpositionen (Pairwise)
    ax = axes[2]
    names = list(alle.keys())
    n = len(names)
    overlap = np.zeros((n, n))
    union_sizes = np.zeros((n, n))
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            ratios = []
            union_total = 0
            inter_total = 0
            for rid in set(alle[name_a]) & set(alle[name_b]):
                chars_a = zeichen[name_a][rid]
                if not chars_a:
                    continue
                inter = len(chars_a & zeichen[name_b][rid])
                ratios.append(inter / len(chars_a))
                inter_total += inter
                union_total += len(chars_a)
            overlap[i, j] = np.mean(ratios) * 100 if ratios else 0
            union_sizes[i, j] = inter_total / union_total * 100 if union_total else 0

    im = ax.imshow(overlap, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            text = f"{overlap[i, j]:.1f}%"
            if i != j and union_sizes[i, j] > 0:
                text += f"\n({union_sizes[i, j]:.1f}% union)"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if overlap[i, j] > 50 else "black", fontsize=9)
    ax.set_title("Überlappung der Annotationen (Zeichenpositionen)\n"
                 "Zeile: Anteil der Zeichen von Zeile, die auch in Spalte abgedeckt sind")
    fig.colorbar(im, ax=ax, label="Überlappung (%)")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nDiagramm gespeichert: {args.output.resolve()}")


if __name__ == "__main__":
    main()

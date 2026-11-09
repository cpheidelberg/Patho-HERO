#!/usr/bin/env python3
"""Multi-model benchmark for the hybrid NER pipeline (LLM stage).

Fixed test set: the 100 reports extracted in ``experiments/hybrid_ner.jsonl``
(the medgemma:27b baseline run: report texts + gazetteer + regex spans).
For each Ollama model the LLM-fallback stage of the hybrid pipeline is
re-run over all 100 reports and every resulting LLM span is mapped against
the LOCAL UMLS PostgreSQL data (``LocalUMLSClient`` + ``OntologyMapper``),
identical to ``src/main.py``'s ontology setup.

Compared models (default): medgemma:27b, medgemma1.5:4b, qwen3.8:27b.

Per model, per ``--runs`` (default: 5, LLM sampling makes the extraction
non-deterministic -> mean ± s.d. is reported):
  * wall-clock time (whole run, incl. ontology mapping)
  * #LLM spans, #ontologized LLM spans (local UMLS hit)
  * #LLM spans, #non-ontologized LLM spans (local UMLS miss)
  * shared LLM search terms across models (union across the runs)
  * per-label span breakdown

Per-run outputs (raw spans + mapping outcome):
  experiments/bench/<model>.run<k>.jsonl
Summary (all models, per-run values + shared terms):
  experiments/bench/bench_models_summary.json

Usage (from the project root, with the ``patientdb`` conda env active):
    python experiments/bench_models.py
    python experiments/bench_models.py --limit 8 --workers 8 \
        --models medgemma:27b,medgemma1.5:4b,qwen3.8:27b
    python experiments/bench_models.py --runs 5
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.config import Settings, load_pipeline_config
from src.database import Repository
from src.extractors.llm_extractor import LLMExtractor
from src.logging_config import setup_logging
from src.models import Entity, Label, Source
from src.terminology.local_umls import LocalUMLSClient
from src.terminology.mapper import OntologyMapper

DEFAULT_MODELS = "medgemma:27b,medgemma1.5:4b,qwen3.8:27b"
DEFAULT_INPUT = Path("experiments/hybrid_ner.jsonl")


# ---------------------------------------------------------------------------
# Ollama model unloader — prevents one model occupying VRAM "forever" and
# blocking the next model from being loaded.
# ---------------------------------------------------------------------------

def _unload_model(base_url: str, model: str) -> None:
    """Sends an Ollama request with keep_alive=0 to force the model out of
    GPU / host memory so the next model can be loaded.

    Uses a minimal prompt (empty string, num_predict=0, no streaming) to keep
    the call as cheap as possible while still triggering Ollama's
    unload-on-complete path.
    """
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": "", "stream": False,
                  "keep_alive": 0, "options": {"num_predict": 0}},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"    [unload] {model}: Ollama unloads after keep_alive=0 request "
              f"(HTTP {resp.status_code})")
    except requests.exceptions.RequestException as exc:
        print(f"    [unload] WARN: failed to unload {model} via Ollama: {exc}")


# ---------------------------------------------------------------------------
# Deterministic pipeline pieces (gazetteer/regex/negation/dedup/filter) —
# byte-identical to src/pipeline.py NERPipeline for the deterministic stage,
# so all models are compared on the same deterministic basis.
# ---------------------------------------------------------------------------

class DeterministicStage:
    """Gazetteer + Regex + Negation, 1:1 copy of NERPipeline's deterministic
    stages (src/pipeline.py) so the models are compared under identical
    conditions."""

    def __init__(self, config: dict):
        gaz_dir = Path("config")
        FILE_TO_LABEL = {
            "probe.txt": Label.PROBE,
            "lokalisationen.txt": Label.LOCALISATION,
            "probenarten.txt": Label.PROBE_TYPE,
            "faerbungen.txt": Label.STAINING,
            "ihc_marker.txt": Label.IHC_MARKER,
        }
        self._gz_entries = []
        for filename, label in FILE_TO_LABEL.items():
            path = gaz_dir / filename
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                term = line.strip()
                if not term:
                    continue
                self._gz_entries.append((
                    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE),
                    label, term))

        rx_cfg = config.get("regex", {})
        self._rx = [
            (Label.ONTOLOGY, re.compile(rx_cfg["icdo"])),
            (Label.CLINICAL, re.compile(rx_cfg["tnm"])),
            (Label.PROBE_SIZE, re.compile(rx_cfg["groesse"], re.IGNORECASE)),
        ]

        neg_cfg = config.get("negation", {})
        self.neg = None
        if neg_cfg and config.get("pipeline", {}).get("use_negation", True):
            from src.context.negation import NegationDetector
            self.neg = NegationDetector(
                neg_cfg["triggers"], neg_cfg["uncertainty_triggers"],
                neg_cfg.get("window_tokens", 6))

    def extract(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        for pattern, label, canonical in self._gz_entries:
            for m in pattern.finditer(text):
                entities.append(Entity(
                    label=label, text=m.group(), start=m.start(), end=m.end(),
                    source=Source.GAZETTEER, search_term=canonical))
        for label, pattern in self._rx:
            for m in pattern.finditer(text):
                entities.append(Entity(
                    label=label, text=m.group(), start=m.start(), end=m.end(),
                    source=Source.REGEX, search_term=None))
        # Dedup wie NERPipeline._deduplicate
        seen, out = set(), []
        for e in entities:
            key = (e.start, e.end, e.label)
            if key not in seen:
                seen.add(key)
                out.append(e)
        if self.neg is not None:
            out = self.neg.apply(text, out)
        return [e for e in out if text[e.start:e.end].lower() == e.text.lower()]


# ---------------------------------------------------------------------------
# LLM stage: exakt die Entitäts-/Overlap-Logik von NERPipeline.run()
# (src/pipeline.py), aber pro Run einmalig pro Modell geladen.
# ---------------------------------------------------------------------------

class LLMStage:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0,
                 keep_alive: str = "2m"):
        # Set the environment variable before creating the extractor so it
        # picks up the desired keep_alive duration
        import os
        os.environ["OLLAMA_KEEP_ALIVE"] = keep_alive
        self._ex = LLMExtractor(base_url, model, temperature)

    def extract(self, text: str, deterministic: list[Entity]) -> list[Entity]:
        covered = [e.text for e in deterministic]
        ents: list[Entity] = []
        for e in self._ex.extract(text, exclude=covered):
            if any(e.start < b.end and b.start < e.end for b in deterministic):
                continue  # überlappt mit bereits extrahierter Entität
            ents.append(e)
        return ents


# ---------------------------------------------------------------------------
# Baseline loader (reports from experiments/hybrid_ner.jsonl)
# ---------------------------------------------------------------------------

def _load_baseline_records(input_path: Path, limit: int | None) -> list[dict]:
    """Lädt die 100 Baseline-Berichte (ein Datensatz pro report_id)."""
    recs: dict[int, dict] = {}
    order: list[int] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("meta_type") == "experiment_header":
                continue
            rid = int(r["meta"]["report_id"])
            if rid not in recs:
                recs[rid] = r
                order.append(rid)
    out = [recs[rid] for rid in order]
    if limit and limit > 0:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def stdev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def analyze_run(path: Path) -> dict:
    """Analysiert eine per-Run-JSONL: Spans nach Quelle, Mapping-Kennzahlen."""
    d: dict[int, dict] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("meta_type") == "experiment_header":
            continue
        d[int(r["meta"]["report_id"])] = r

    tot = llm = rx = gaz = 0
    mapped = llm_mapped = llm_unmapped = 0
    labels: Counter = Counter()
    terms: set[str] = set()
    for rec in d.values():
        for s in rec.get("spans", []):
            src = (s.get("meta") or {}).get("source")
            if src == "llm":
                llm += 1
                if s.get("concept_id"):
                    llm_mapped += 1
                else:
                    llm_unmapped += 1
                labels[s.get("label") or "?"] += 1
                st = s.get("search_term")
                if st:
                    terms.add(st.lower().strip())
            elif src == "regex":
                rx += 1
            elif src == "gazetteer":
                gaz += 1
            if s.get("concept_id"):
                mapped += 1
            tot += 1
    n_reports = len(d)
    return dict(n_reports=n_reports, tot=tot, llm=llm, rx=rx, gaz=gaz,
                mapped=mapped, llm_mapped=llm_mapped, llm_unmapped=llm_unmapped,
                n_terms=len(terms), labels=dict(labels), terms=terms)


def summarize_runs(run_results: list[dict]) -> dict:
    """Aggregiert die per-Run-Kennzahlen zu mean/stdev + per-run-Werte."""
    metric_keys = ["n_reports", "tot", "llm", "rx", "gaz", "mapped",
                   "llm_mapped", "llm_unmapped"]
    out: dict = {"n_runs": len(run_results)}
    for k in metric_keys:
        vals = [r[k] for r in run_results]
        mean = sum(vals) / len(vals) if vals else 0
        out[k + "_mean"] = round(mean, 3)
        out[k + "_stdev"] = round(stdev(vals), 3)
        out["run_" + k] = vals
    walls = [r["wall_s"] for r in run_results]
    out["wall_s_mean"] = round(sum(walls) / len(walls), 1) if walls else 0
    out["wall_s_stdev"] = round(stdev(walls), 1)
    out["run_wall_s"] = [round(w, 1) for w in walls]
    per_report = [r["per_report_s"] for r in run_results]
    out["per_report_s_mean"] = round(sum(per_report) / len(per_report), 2) if per_report else 0
    out["per_report_s_stdev"] = round(stdev(per_report), 3)

    # Ontology-resolved share of the LLM-derived spans (0..1),
    # mittlere der per-Run-Shares. Kern-Metrik des UMLS-Mappings pro Model.
    shares = [r["llm_mapped"] / r["llm"] if r["llm"] else 0.0 for r in run_results]
    out["mapped_share_mean"] = round(sum(shares) / len(shares), 4) if shares else 0
    out["mapped_share_stdev"] = round(stdev(shares), 4)
    out["run_mapped_share"] = shares

    # Label-Aufteilung: per-Run-Zählung, gemittelt
    label_keys = sorted({lk for r in run_results for lk in r.get("labels", {}).keys()})
    labels_mean: dict[str, float] = {}
    for lk in label_keys:
        vals = [r.get("labels", {}).get(lk, 0) for r in run_results]
        labels_mean[lk] = round(sum(vals) / len(vals), 3)
    out["labels_mean"] = labels_mean
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _span_dict(entity: Entity, mapping) -> dict:
    spans_dict = {
        "start": entity.start,
        "end": entity.end,
        "text": entity.text,
        "label": entity.label.value,
        "terminology": mapping.ontology,
        "concept_id": mapping.concept_id,
        "search_term": entity.search_term,
        "meta": {
            "source": entity.source.value,
            "assertion": entity.assertion.value,
            "confidence": 1.0 if entity.source.value != "llm" else 0.5,
        },
    }
    return spans_dict


def main():
    ap = argparse.ArgumentParser(
        description="Multi-model hybrid NER benchmark (LLM-Stage, lokales UMLS-Postgres)")
    ap.add_argument("--models", type=str, default=DEFAULT_MODELS,
                    help="Kommagetrennte Ollama-Modell-Tags (default: "
                         f"{DEFAULT_MODELS})")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="Baseline-JSONL mit den 100 Berichten (default: "
                         f"{DEFAULT_INPUT})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Nur die ersten N Berichte (default: alle 100)")
    ap.add_argument("--workers", type=int, default=10,
                    help="Parallele Berichte pro Modell-Lauf")
    ap.add_argument("--map-workers", type=int, default=10,
                    help="Parallele UMLS-Mapping-Worker pro Lauf")
    ap.add_argument("--runs", type=int, default=5,
                    help="Anzahl Extraktionsläufe pro Modell (Default: 5)")
    ap.add_argument("--outdir", type=Path, default=Path("experiments/bench"))
    ap.add_argument("--label-file", type=str,
                    default="bench_models_summary.json",
                    help="Zusammenfassungs-JSON unter --outdir (Default: "
                         "bench_models_summary.json)")
    ap.add_argument("--keep-alive", type=str, default="2m",
                    help="How long Ollama keeps each model loaded in memory after "
                         "the last request (default: 2m, or 0 to force unload "
                         "immediately; use -1 for infinite)")
    args = ap.parse_args()

    if args.runs < 1:
        ap.error("--runs muss mindestens 1 sein")

    settings = Settings()
    config = load_pipeline_config()
    setup_logging(log_dir=config["logging"]["log_dir"],
                  level=config["logging"]["log_level"])

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    records = _load_baseline_records(args.input, args.limit)
    n_reports = len(records)
    if not records:
        raise SystemExit(f"Keine Berichte in {args.input} gefunden.")

    # Deterministische Stufen (einmalig geladen, alle Models identisch)
    deterministic = DeterministicStage(config)

    # --- Local UMLS (Postgres) + OntologyMapper (identisch zu src/main.py) ---
    print(f"Local UMLS-Postgres: {settings.umls_local_db_url or settings.database_url}")
    print(f"  language={settings.umls_local_language}  "
          f"sabs={settings.umls_sabs_list}")
    repo = Repository(settings.database_url)
    umls = LocalUMLSClient(
        api_key=settings.umls_api_key,
        version=settings.umls_version,
        repository=repo,
        schema=settings.umls_local_schema or None,
        language=settings.umls_local_language,
    )
    # Preflight: mrconso muss existieren (sonst früh Fehler melden, nicht
    # erst bei der ersten Suche nach 30 Sekunden).
    with umls._repo.engine.connect() as conn:
        conn.exec_driver_sql("SELECT 1 FROM mrconso LIMIT 1")
    print("  mrconso: OK")
    mapper = OntologyMapper(
        repo=repo,
        snowstorm_client=None,
        umls_client=umls,
        default_provider="umls",
        provider_per_label={},
        snowstorm_terminology="UMLS",
        label_ecl=config.get("label_ecl", {}),
        ancestor_depth=config["pipeline"].get("ancestor_depth", 1),
        umls_sabs=settings.umls_sabs_list,
        umls_terminology_version=settings.umls_version,
    )

    file_lock = threading.Lock()
    results: dict[str, dict] = {}
    term_sets: dict[str, set[str]] = {m: set() for m in models}

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {len(models)} Modelle x {args.runs} Runs x {n_reports} Berichte")
    print(f"Inputs: {args.input}  |  Outputs: {args.outdir}/")
    print(f"Workers: reports={args.workers}  mapping={args.map_workers}")
    print(f"{'=' * 60}\n")

    for i, m in enumerate(models):
        print(f"[{i + 1}/{len(models)}] OLLAMA_MODEL={m}  runs={args.runs}")
        per_run: list[dict] = []
        for r in range(1, args.runs + 1):
            safe = m.replace('/', '_').replace(':', '_')
            out_path = args.outdir / f"{safe}.run{r}.jsonl"

            # LLM-Stage für dieses Model (einmal pro Run)
            llm_stage = LLMStage(settings.ollama_base_url, m,
                                 settings.ollama_temperature,
                                 keep_alive=args.keep_alive)

            t0 = time.perf_counter()
            processed: dict[int, dict] = {}
            failed: list[str] = []

            def _work(rec: dict):
                rid = int(rec["meta"]["report_id"])
                text = rec["text"]
                try:
                    det = deterministic.extract(text)
                    llm_stage_extract = llm_stage.extract(text, det)
                    entities = det + llm_stage_extract
                    # Mapping über den OntologyMapper (LocalUMLSClient),
                    # thread-gesichert über args.map-workers parallele
                    # Worker-Threads (LLM-Blocking bleibt pro Report).
                    with ThreadPoolExecutor(max_workers=args.map_workers,
                                            thread_name_prefix=f"map-{safe}") as map_exc:
                        mappings = list(map_exc.map(
                            lambda e: mapper.map_entity(e, rid), entities))
                    spans = [_span_dict(e, m) for e, m in zip(entities, mappings)]
                    return rid, spans, None
                except Exception as exc:
                    return rid, None, str(exc)

            def _collect(rec: dict) -> None:
                rid = int(rec["meta"]["report_id"])
                try:
                    rid_, spans, err = _work(rec)
                    if spans is None:
                        raise RuntimeError(err or "unknown error")
                    new = dict(rec)
                    new["spans"] = spans
                    processed[rid] = new
                except Exception as e:
                    failed.append(f"{rid}: {e}")

            with ThreadPoolExecutor(max_workers=args.workers,
                                    thread_name_prefix=f"bench-{safe}") as exc:
                futures = [exc.submit(_collect, rec) for rec in records]
                for fut in as_completed(futures):
                    fut.result()

            wall_s = time.perf_counter() - t0

            # Per-Run-JSONL schreiben (Struktur wie hybrid_ner.jsonl)
            with file_lock:
                with out_path.open("w", encoding="utf-8") as out:
                    header = {
                        "meta_type": "experiment_header",
                        "experiment": {
                            "models": models,
                            "model": m,
                            "run_id": r,
                            "n_reports": n_reports,
                            "workers": args.workers,
                            "map_workers": args.map_workers,
                            "local_umls_url":
                                settings.umls_local_db_url or settings.database_url,
                            "umls_language": settings.umls_local_language,
                            "umls_sabs": settings.umls_sabs_list,
                            "input_file": str(args.input),
                            "baseline_model": "medgemma:27b",
                            "output_file": str(out_path),
                            "failed_reports": failed,
                        }
                    }
                    out.write(json.dumps(header, ensure_ascii=False) + "\n")
                    for rid in sorted(processed):
                        rec = processed[rid]
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            res = analyze_run(out_path)
            res["wall_s"] = round(wall_s, 1)
            res["per_report_s"] = round(wall_s / max(1, n_reports), 2)
            res["failed_reports"] = failed
            term_sets[m].update(res["terms"])
            del res["terms"]
            per_run.append(res)
            print(f"    run {r}/{args.runs}: wall={wall_s:.1f}s "
                  f"spans={res['tot']}  llm={res['llm']}  "
                  f"llm_mapped={res['llm_mapped']} "
                  f"({res['llm_mapped']/res['llm']*100 if res['llm'] else 0:.1f}%)  "
                  f"terms={res['n_terms']}")

        # Unload this model from Ollama so the next model can load without
        # waiting for Ollama's internal timeout (default: keep forever).
        _unload_model(settings.ollama_base_url, m)

        summary = summarize_runs(per_run)
        results[m] = summary
        print(f"    {m}: LLM spans={summary['llm_mean']:.2f}±{summary['llm_stdev']} "
              f"mapped={summary['llm_mapped_mean']:.2f}±{summary['llm_mapped_stdev']}  "
              f"share={summary['mapped_share_mean']:.3f}±{summary['mapped_share_stdev']}  "
              f"wall(mean)={summary['wall_s_mean']}s")

    shared = {
        f"{m1} & {m2}": sorted(term_sets[m1] & term_sets[m2])
        for i, m1 in enumerate(models) for m2 in models[i + 1:]
    }

    final = {
        "runs": results,
        "shared_terms": shared,
        "models": models,
        "n_runs_per_model": args.runs,
        "n_reports": n_reports,
        "limit": args.limit,
        "workers": args.workers,
        "map_workers": args.map_workers,
        "input": str(args.input),
        "local_umls_url": settings.umls_local_db_url or settings.database_url,
        "umls_language": settings.umls_local_language,
        "umls_sabs": settings.umls_sabs_list,
        "umls_version": settings.umls_version,
        "baseline_model": "medgemma:27b",
    }
    spath = args.outdir / args.label_file
    spath.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote summary: {spath}")

    widths = max(20, *(len(m) for m in models))
    col = 22

    def line_mean_sd(label: str, key: str, stdev_key: str) -> None:
        cells = []
        for m in models:
            v = results[m].get(key)
            if v is None:
                cell = "-"
            else:
                cell = f"{v:.2f}±{results[m].get(stdev_key, 0):.2f}"
            cells.append(cell.rjust(col - 2))
        print(f"{label:<{widths}} " + "  ".join(cells))

    print("\n" + "=" * (widths + 3 * col * len(models)))
    print(f"{'Metric (mean ± s.d.)':<{widths}} " +
          "  ".join(f"{m:>{col - 2}}" for m in models))
    print("-" * (widths + 3 * col * len(models)))
    line_mean_sd("Wall time (s)", "wall_s_mean", "wall_s_stdev")
    line_mean_sd("Per report (s)", "per_report_s_mean", "per_report_s_stdev")
    line_mean_sd("Total spans", "tot_mean", "tot_stdev")
    line_mean_sd("  LLM-derived", "llm_mean", "llm_stdev")
    line_mean_sd("  Regex-derived", "rx_mean", "rx_stdev")
    line_mean_sd("  Gazetteer-derived", "gaz_mean", "gaz_stdev")
    line_mean_sd("Ontologized (total)", "mapped_mean", "mapped_stdev")
    line_mean_sd("  LLM -> local UMLS hit", "llm_mapped_mean", "llm_mapped_stdev")
    line_mean_sd("  LLM -> local UMLS miss", "llm_unmapped_mean", "llm_unmapped_stdev")
    line_mean_sd("  LLM hit share (0..1)", "mapped_share_mean", "mapped_share_stdev")
    line_mean_sd("Distinct LLM terms (mean)", "n_terms_mean", "n_terms_stdev")
    union_cells = "  ".join(f"{len(term_sets[m]):>{col - 2}}" for m in models)
    print(f"{'Distinct LLM terms (union)':<{widths}} " + union_cells)
    print("-" * (widths + 3 * col * len(models)))
    print("Shared LLM search terms (union across runs):")
    for k, v in shared.items():
        preview = v[:8]
        print(f"  {k}: {preview}{'...' if len(v) > 8 else ''}   ({len(v)} total)")
    print()

    print(f"Per-run JSONL: {args.outdir}/<model>.run<k>.jsonl")
    print(f"  (n={args.runs} pro Model, Struktur wie experiments/hybrid_ner.jsonl)")

    umls.close()


if __name__ == "__main__":
    main()

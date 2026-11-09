# Patho-HERO

**Patho-HERO: Archive-scale noisy whole-slide-image cohort generation from German pathology reports using Histology Entity Recognition and Ontology linking.**

This repository contains the *code* used for the Patho-HERO publication.
The clinical report archive, the SNOMED CT / UMLS ontology data, and the internal PAS-Net infrastructure are not part of this public repository.
To keep the pipeline reproducible without access to patient data, the repository ships a fully synthetic 50-report test set (in `berichte.xlsx` and `sample_reports.sqlite`) and every experiment script runs against it by default.

## Repository layout

```
src/
  config.py                  # Settings + load_pipeline_config
  database.py                # PostgreSQL Repository & ORM models (production path)
  pipeline.py                # NERPipeline: gazetteer + regex + LLM-fallback + negation + ontology-mapping
  extractors/
    base.py, gazetteer.py, regex_extractor.py, llm_extractor.py, hf_extractor.py
  context/negation.py        # German NegEx / ConText-style detector
  terminology/
    mapper.py                # label-aware ontology mapper (Snowstorm or UMLS)
    snowstorm.py             # SNOMED CT via a local Snowstorm server
    umls.py                  # UMLS REST API
    local_umls.py            # local UMLS (Postgres) client
    umls_rrf_importer.py     # Reciprocal-Fusion UMLS importer
  ingestion/                 # importer + loader used by the production pipeline
  analysis.py                # ad-hoc analytical queries against the report DB
  main.py                    # production entrypoint (Postgres pipeline)

config/
  config.yaml                # extraction configuration
  config_cluster.yaml        # cluster-specific overrides (if applicable)
  probe.txt, probenarten.txt, lokalisationen.txt,
  faerbungen.txt, ihc_marker.txt   # gazetteer term lists
  llm_ner_prompt.md          # one-shot LLM extractor prompt

experiments/
  hybrid_ner.py              # end-to-end NER on the test set (primary experiment)
  hybrid_extractor.py        # multi-provider ontology mapping on hybrid_ner spans
  medcat_ner.py              # MedCAT (SNOMED CT) baseline
  openmed_ner.py             # OpenMed baseline
  create_report_testset.py   # production: random report list from the DB
  make_synthetic_testset.py  # synthetic test set factory (50 reports, no patient data)
  make_sqlite_sample_db.py   # build sample_reports.sqlite
  sample_store.py            # SQLite-backed Repository stand-in for the experiments
  visualize_ner.py           # comparison of MedCAT vs OpenMed vs Hybrid
  plot_ner_figure.py         # NER comparison publication figure
  plot_ontology_panels.py    # ontology mapping panels
  plot_lymphoma_use_case.py  # lymphoma use-case (yield + concordance)
  plot_archive_usecase.py    # combined 2x2 publication figure
  bench_models.py            # multi-model LLM benchmark (medgemma / qwen3.8)

scripts/
  lymphoma_case_search.py    # rule-based lymphoma-cohort retrieval

LICENSE
requirements.txt
README.md
.gitignore
```

## Sample data

Real clinical reports cannot be published, so the repository ships with:

- `experiments/berichte.xlsx` — **50 synthetic pathology reports**,
  templated from the vocabulary the deterministic extractors already
  know about (the `config/*.txt` gazetteer lists, ICD-10-GM / ICD-O-3
  codes, TNM formulae, sizes, staining and negation phrases). All cells
  are fabricated from a fixed seed (default 42).
- `experiments/sample_reports.sqlite` — a SQLite database with the same
  50 reports in a `case_report` table plus an empty `report_concept`
  table that accumulates spans produced by the experiments.

To regenerate both (or produce a different size / seed):

```bash
python3 experiments/make_synthetic_testset.py            # → experiments/berichte.xlsx
python3 experiments/make_sqlite_sample_db.py             # → experiments/sample_reports.sqlite
```

Neither file contains patient data.

## Quick start (self-contained, no Postgres, no patient data)

1. **Install dependencies**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Point the experiments at the SQLite sample** by setting
   `PATHOHERO_SQLITE` to the file produced in step 0 (or by re-running
   `make_sqlite_sample_db.py` if it is missing):

   ```bash
   export PATHOHERO_SQLITE="$(pwd)/experiments/sample_reports.sqlite"
   ```

3. **Run the hybrid pipeline** (gazetteer + regex + LLM fallback via
   Ollama if a local Ollama server is reachable):

   ```bash
   python3 experiments/hybrid_ner.py -i experiments/berichte.xlsx -l 5 \
     -o experiments/hybrid.jsonl
   ```

   The pipeline probes `$OLLAMA_BASE_URL/api/tags` (default
   `http://localhost:11434/api/tags`) with a 3-second timeout. On a
   successful handshake it picks an available medical model in order:
   `medgemma1.5:4b` → `medgemma:27b` → `qwen3.8:27b` → default from
   `OLLAMA_MODEL`. On any failure the LLM stage is skipped and
   gazetteer + regex + negation still run.

4. **Inspect the results in SQLite** (the experiment also persists the
   extracted spans into the sample DB):

   ```bash
   sqlite3 experiments/sample_reports.sqlite \
     "select report_id, surface_text, ontology, start_offset, end_offset
      from report_concept
      order by report_id, start_offset;"
   ```

   or programmatically:

   ```python
   import sqlite3
   con = sqlite3.connect("experiments/sample_reports.sqlite")
   for row in con.execute("select * from report_concept"): print(row)
   ```

5. **Optional: produce the publication figures.** These scripts read the
   JSONL/CSV exports produced by the experiment above and render a
   matplotlib figure. They all honour `PATHOHERO_SQLITE`:

   ```bash
   # 2x2 publication figure (a–d panels)
   python3 experiments/plot_archive_usecase.py

   # NER comparison (MedCAT vs OpenMed vs Hybrid)
   python3 experiments/plot_ner_figure.py

   # Ontology mapping panels
   python3 experiments/plot_ontology_panels.py

   # Lymphoma use-case
   python3 experiments/plot_lymphoma_use_case.py
   ```

## Default ontology provider

`config.yaml` sets `use_ontology: true` with `ontology_provider: umls`.
At runtime the mapper resolves the UMLS client in this order:

| Condition | UMLS client built |
|---|---|
| `UMLS_LOCAL_ENABLED=true` **and** `UMLS_LOCAL_DB_URL` reachable | `LocalUMLSClient` (a 4-table Postgres install — what the production deployment uses) |
| else, `UMLS_API_KEY` is set | `UMLSClient` hitting `https://uts-ws.nlm.nih.gov/rest` (the remote REST API) |
| else | **no UMLS client**; mapper returns `concept_id=NULL` for every entity |

Without local installation, remote UMLS REST API is easily used when you explicitly set `UMLS_API_KEY`. 
Therefore, provide:

```bash
export UMLS_API_KEY=<your UTS key>        # free from https://uts.nlm.nih.gov/uts/
export UMLS_VERSION=2026AA
export UMLS_SABS=SNOMEDCT_US,MSH,ICD10
# and in config/config.yaml keep (or set):
#   use_ontology: true
#   ontology_provider: umls
```

In sample mode (a SQLite store, remote UMLS), the pipeline runs end-to-end.
Without API key provided, spans are persisted with `concept_id=NULL` which is the expected output unless you configure one of the two UMLS
paths above.

## Production configuration

The default `config/config.yaml` targets the *production* pipeline  (`use_gazetteer`, `use_regex`, `use_llm_fallback`, `use_ollama`, `use_ontology`). 
To run the full production code against your clinical database, provide `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` in the environment (see `src/config.py`) and run `python3 src/main.py` as usual.
The `experiments/` scripts will use PostgreSQL by default. 
Set `PATHOHERO_SQLITE` to opt in to the synthetic-sample path described above.

## Citing

Blattgerste, C., Kireev, A., Legnar, M., Aliyari, S., Hesser, J., &
Weis, C.-A. *Patho-HERO: Archive-scale noisy whole-slide-image cohort
generation from German pathology reports using Histology Entity
Recognition and Ontology linking*. To be published.

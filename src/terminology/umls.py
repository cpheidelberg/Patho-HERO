import logging
import os
import threading
from pathlib import Path

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def _resolve_verify() -> bool | str:
    """Bestimmt das TLS-Verify-Ziel für alle UMLS-Requests."""
    cert = os.getenv("SSL_CERT")
    if cert:
        path = Path(cert)
        if path.exists():
            logger.debug("UMLS | SSL-Zertifikat: %s", path)
            return str(path)
        logger.warning("UMLS | SSL_CERT gesetzt, aber Datei fehlt: %s", cert)
    else:
        logger.debug("UMLS | SSL_CERT nicht gesetzt")

    logger.warning("UMLS | TLS-Verifikation deaktiviert (verify=False)")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return False


class UMLSClient:
    """Thread-safe Client für die UMLS REST API (UTS)."""

    def __init__(self, api_key: str, version: str = "current",
                 base_url: str = "https://uts-ws.nlm.nih.gov/rest",
                 page_size: int = 25, timeout: int = 30,
                 pool_maxsize: int = 32):
        self.api_key = api_key
        self.version = version
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.timeout = timeout


        # --- TGT-Zustand (geteilt, daher unter Lock) ---
        self._tgt: str | None = None
        self._tgt_lock = threading.Lock()
        self._tgt_generation = 0  # zählt, wie oft das TGT ersetzt wurde

        # --- HTTP-Session mit Retry für parallele Last ---
        self._session = requests.Session()
        verify = _resolve_verify()
        if verify:
            self._session.verify = _resolve_verify()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry,
                              pool_connections=pool_maxsize,
                              pool_maxsize=pool_maxsize)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def close(self) -> None:
        """Gibt die HTTP-Verbindungen frei."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------ auth
    def _fetch_tgt(self) -> str:
        """Beschafft ein neues Ticket Granting Ticket. Nur unter Lock aufrufen."""
        resp = self._session.post(
            "https://utslogin.nlm.nih.gov/cas/v1/api-key",
            data={"apikey": self.api_key},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        # Location-Header enthält das TGT
        tgt = resp.headers["location"]
        logger.debug("UMLS TGT erhalten: %s", tgt)
        return tgt

    def _fetch_service_ticket(self, tgt: str) -> str:
        """Holt ein einmalig gültiges Service Ticket zu einem TGT."""
        resp = self._session.post(
            tgt,
            data={"service": "http://umlsks.nlm.nih.gov"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.text

    def _get_service_ticket(self) -> tuple[str, int]:
        """Liefert ein Service Ticket samt TGT-Generation.

        Die Generation erlaubt es dem Aufrufer, genau dieses TGT zu
        invalidieren, ohne ein zwischenzeitlich erneuertes zu treffen.
        """
        with self._tgt_lock:
            if not self._tgt:
                self._tgt = self._fetch_tgt()
                self._tgt_generation += 1
            tgt = self._tgt
            generation = self._tgt_generation

        # Service Ticket außerhalb des Locks: pro Request ein eigenes ST,
        # das darf und soll parallel laufen.
        return self._fetch_service_ticket(tgt), generation

    def _invalidate_tgt(self, seen_generation: int) -> None:
        """Verwirft das TGT nur, wenn es noch das vom Aufrufer benutzte ist."""
        with self._tgt_lock:
            if self._tgt_generation == seen_generation:
                self._tgt = None
                logger.debug("UMLS TGT (gen %d) invalidiert", seen_generation)

    def _auth_headers(self) -> dict:
        return {"Accept": "application/json"}

    def _get(self, url: str, params: dict | None = None) -> dict:
        """Führt einen authentifizierten GET-Request aus."""
        ticket, generation = self._get_service_ticket()
        p = {"ticket": ticket, **(params or {})}
        resp = self._session.get(url, params=p, headers=self._auth_headers(), timeout=self.timeout)
        if resp.status_code == 401:
            # TGT kann abgelaufen sein – neu beschaffen und nochmal versuchen
            self._invalidate_tgt(generation)
            ticket, _ = self._get_service_ticket()
            p["ticket"] = ticket
            resp = self._session.get(url, params=p, headers=self._auth_headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ search
    def search(self, term: str, sabs: list[str] | None = None,
               search_type: str = "words", return_id_type: str = "concept",
               limit: int = 5) -> list[dict]:
        """Sucht nach Begriffen in UMLS."""
        url = f"{self.base_url}/search/{self.version}"
        params: dict = {
            "string": term,
            "searchType": search_type,
            "returnIdType": return_id_type,
            "pageSize": min(limit, self.page_size),
        }
        if sabs:
            params["sabs"] = ",".join(sabs)

        data = self._get(url, params)
        results = data.get("result", {}).get("results", [])
        return results[:limit]


    def source_code_to_cui(self, root_source: str, code: str) -> str | None:
        """Sucht das UMLS Concept mit CUI für einen Source-Code."""
        url = f"{self.base_url}/search/{self.version}"
        params = {
            "string": code,
            "inputType": "sourceUi",
            "searchType": "exact",
            "sabs": root_source,
            "returnIdType": "concept",
        }
        data = self._get(url, params=params)
        results = data.get("result", {}).get("results", [])
        if not results:
            return None

        return results[0].get("ui")
    
    # ------------------------------------------------------------------ concept
    def get_concept(self, cui: str) -> dict:
        """Holt Concept-Details (CUI) inkl. aller bevorzugten Namen."""
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}"
        data = self._get(url)
        concept = data.get("result", {})

        # Name + Definitionen + Atoms (= Terme aus verschiedenen Vokabularen)
        name = concept.get("name", "")
        definitions = []
        for d in concept.get("definitions", []):
            if isinstance(d, dict):
                value = d.get("value")
                if value:
                    definitions.append({"value": value, "source": d.get("rootSource")})
            elif isinstance(d, str) and d.strip():
                definitions.append({"value": d, "source": None})

        atoms = []
        atom_page = concept.get("atoms", "")
        if atom_page:
            atoms = self._collect_atoms(atom_page)

        return {
            "cui": cui,
            "preferred_name": name,
            "semantic_types": [
                st.get("name") for st in concept.get("semanticTypes", [])
                if st.get("name")
            ],
            "definitions": definitions,
            "atoms": atoms,
        }

    def _collect_atoms(self, atom_url: str) -> list[dict]:
        """Sammelt alle Atoms eines Concepts (Terme + Vokabular + Typ)."""
        atoms: list[dict] = []
        url = atom_url
        while url and len(atoms) < 500:  # Sicherheitsbremse
            data = self._get(url)
            page_atoms = data.get("result", [])
            for a in page_atoms:
                atoms.append({
                    "term": a.get("name"),
                    "source": a.get("rootSource"),
                    "term_type": a.get("termType"),
                    "language": a.get("language"),
                    "code": a.get("code"),
                })
            # Pagination via Links
            next_link = None
            for link in data.get("page", {}).get("relations", []):
                if link.get("relation") == "next":
                    next_link = link.get("url")
                    break
            url = next_link
        return atoms

    # ------------------------------------------------------------------ relations
    def get_ancestors(self, cui: str, sabs: list[str] | None = None, max_results: int = 100) -> list[dict]:
        """Holt Vorfahren-Relationen eines CUI."""
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}/relations"
        params = {
            "pageSize": min(max_results, self.page_size),
            "includeRelationLabels": "RN,CHD",
        }

        ancestors = []
        seen = set()
        page_number = 1
        while len(ancestors) < max_results:
            params["pageNumber"] = page_number
            try:
                data = self._get(url, params)
            except requests.HTTPError as e:
                logger.debug("%s | Keine Oberbegriffe gefunden (%s)", cui, e)
                return []

            relations = data.get("result", [])
            if not relations:
                break

            for rel in relations:
                related = rel.get("relatedId", "")
                sab = rel.get("rootSource")
                if sab not in sabs:
                    continue
                code = related.split("/")[-1]

                related_cui = self.source_code_to_cui(sab, code)
                if related_cui in seen or related_cui is None:
                    continue
                seen.add(related_cui)
                ancestors.append({
                    "cui": related_cui,
                    "name": rel.get("relatedIdName"),
                    "relation": rel.get("relationLabel")
                })
                if len(ancestors) >= max_results:
                    break

            # Pagination
            page = data.get("page", {})
            if page_number >= page.get("pageCount", 1):
                break
            page_number += 1
        logger.debug(f"{len(ancestors)} of {len(relations)} ancestors in SABs: {sabs}") 

        return ancestors[:max_results]


    def get_ancestor_graph(self, cui: str, cname: str, sabs: list[str] | None = None, max_depth: int = 5) -> list[dict]:
        """
        Liefert alle Ancestors eines CUI über die gegebene Relation (standardmäßig RB)
        bis zu max_depth. Jeder Ancestor wird mit distance, parent_cui, name, source zurückgegeben.
        """
        from collections import deque

        queue = deque()
        queue.append((cui, None, 0))

        visited: set[str] = set()
        visited.add(cui)
        hierarchy_paths: dict[str, list[str]] = {cui: [cname]}

        ancestors: list[dict] = []

        logger.debug("UMLS ancestor hierarchy start: %s (%s)", cname, cui)

        while queue:
            current_cui, parent_cui, distance = queue.popleft()
            logger.debug(f"Analyse ancestors for {current_cui} ({distance})")

            current_path = hierarchy_paths.get(current_cui, [current_cui])
            parents = self.get_ancestors(current_cui, sabs)

            logger.debug(f"{distance+1}. level Ancestors: {len(parents)}")
            for p in parents:
                parent_cui = p["cui"]
                if parent_cui in visited:
                    continue

                visited.add(parent_cui)

                ancestor_distance = distance + 1
                ancestor_name = p.get("name") or parent_cui
                ancestor_path = current_path + [ancestor_name]
                hierarchy_paths[parent_cui] = ancestor_path

                ancestors.append(
                    {
                        "cui": parent_cui,
                        "parent_cui": current_cui,
                        "distance": ancestor_distance,
                        "relation": p.get("relation"),
                        "name": ancestor_name,
                        "source": p.get("source"),
                    }
                )

                logger.debug(f"UMLS ancestor hierarchy: {cname} ({cui}) -> {ancestor_name} ({p['cui']}) - distance: {ancestor_distance}")

                if ancestor_distance < max_depth:
                    queue.append((parent_cui, current_cui, ancestor_distance))

        return ancestors
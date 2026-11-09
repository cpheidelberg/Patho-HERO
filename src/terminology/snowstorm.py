import requests


class SnowstormClient:
    """Thin Client für lokalen Snowstorm mit deutscher SNOMED-Edition."""

    def __init__(self, base_url: str, branch: str, language: str,
                 accept_language: str):
        self.base_url = base_url.rstrip("/")
        self.branch = branch
        self.language = language
        self.headers = {
            "Accept": "application/json",
            "Accept-Language": accept_language,
        }

    def search(self, term: str, ecl: str | None = None,
               limit: int = 5, retry_without_language: bool = True) -> list[dict]:
        """Textsuche, optional eingeschränkt auf ECL-Hierarchie.

        Auf der (de facto englischdominierten) deutschen Edition filtert
        ``language=de`` ausschließlich auf deutsche Beschreibungen und kann
        Konzepte ausblenden, die nur englische Terme tragen. Wird deshalb
        ohne Treffer einmal ohne Language-Filter wiederholt, um diese Treffer
        nicht zu verlieren (nur bei leerem Ergebnis ein zweiter Request).
        """
        params = {
            "term": term,
            "language": self.language,
            "limit": limit,
            "activeFilter": True,
            "termActive": True,
        }
        if ecl:
            params["ecl"] = ecl

        url = f"{self.base_url}/{self.branch}/concepts"
        resp = requests.get(url, params=params,
                            headers=self.headers, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items and retry_without_language and "language" in params:
            fallback_params = {k: v for k, v in params.items() if k != "language"}
            resp = requests.get(url, params=fallback_params,
                                headers=self.headers, timeout=30)
            resp.raise_for_status()
            items = resp.json().get("items", [])
        return items

    def get_concept(self, concept_id: str) -> dict:
        """Holt Concept inkl. FSN, PT und aller Descriptions in einem Request.

        Liefert normalisiertes Dict:
        {
            "fsn_term": str | None,
            "fsn_semantic_tag": str | None,
            "pt_term": str | None,
            "descriptions": [{"term": str, "languageCode": str, "type": str}]
        }
        """
        resp = requests.get(
            f"{self.base_url}/browser/{self.branch}/concepts/{concept_id}",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        concept = resp.json()

        descriptions = [
            {
                "term": d.get("term"),
                "languageCode": d.get("lang"),
                "type": d.get("type"),  # "FSN" | "SYNONYM"
            }
            for d in concept.get("descriptions", [])
            if d.get("active", False)
        ]

        fsn = concept.get("fsn") or {}
        pt = concept.get("pt") or {}
        return {
            "fsn_term": fsn.get("term"),
            "fsn_semantic_tag": concept.get("fsn", {}).get("semanticTag") or self._semantic_tag_from_fsn(fsn.get("term")),
            "pt_term": pt.get("term"),
            "descriptions": descriptions,
        }

    def _semantic_tag_from_fsn(self, fsn_term: str | None) -> str | None:
        """Fallback: '(disorder)' aus dem FSN extrahieren, falls nicht als Feld vorhanden."""
        if fsn_term and fsn_term.endswith(")") and "(" in fsn_term:
            return fsn_term.rsplit("(", 1)[-1].rstrip(")")
        return None

    def get_ancestors(self, concept_code: str) -> list[dict]:
        """Transitive Vorfahren (is_a), für concept_relation."""
        resp = requests.get(
            f"{self.base_url}/browser/{self.branch}/concepts/{concept_code}/ancestors",
            params={"form": "inferred"}, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
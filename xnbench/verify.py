"""Verification against public chemistry databases (PubChem and/or ChEMBL).

Two checks are made for each valid prediction:

1. **Structure lookup** - does the predicted InChIKey exist in the database?
   A hit means the prediction is a known, real compound (a useful sanity
   check, though new compounds in a paper may legitimately be absent).
2. **Label check** (PDF mode) - resolve the name found under the drawing
   (e.g. "Sorafenib") to an InChIKey and compare it with the prediction.

All responses are cached in a JSON file so repeated runs are fast,
reproducible, and polite to the public APIs. Network errors are not cached.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
from pathlib import Path

import requests

log = logging.getLogger(__name__)

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
NOT_FOUND = "__not_found__"


class Verifier:
    def __init__(self, sources: list[str], cache_file: Path,
                 timeout_s: float = 15, sleep_s: float = 0.25):
        unknown = set(sources) - {"pubchem", "chembl"}
        if unknown:
            raise ValueError(f"Unknown verification sources: {sorted(unknown)}")
        self.sources = list(sources)
        self.timeout = timeout_s
        self.sleep = sleep_s
        self.cache_file = cache_file
        self.cache: dict[str, str] = {}
        if cache_file.exists():
            self.cache = json.loads(cache_file.read_text(encoding="utf-8"))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "xn-benchmark/1.0 (academic benchmarking)"

    # ------------------------------------------------------------------ #
    # HTTP with caching
    # ------------------------------------------------------------------ #
    def _get_json(self, cache_key: str, url: str):
        """Return parsed JSON, None for 'not found', or raise on network error."""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return None if cached == NOT_FOUND else json.loads(cached)
        time.sleep(self.sleep)
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code == 404:
            self.cache[cache_key] = NOT_FOUND
            return None
        resp.raise_for_status()
        self.cache[cache_key] = resp.text
        return resp.json()

    def save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self.cache), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # PubChem
    # ------------------------------------------------------------------ #
    def pubchem_cid(self, inchikey: str) -> str | None:
        data = self._get_json(f"pubchem:ik:{inchikey}",
                              f"{PUBCHEM}/compound/inchikey/{inchikey}/cids/JSON")
        cids = (data or {}).get("IdentifierList", {}).get("CID", [])
        return str(cids[0]) if cids else None

    def pubchem_name_to_inchikey(self, name: str) -> str | None:
        quoted = urllib.parse.quote(name, safe="")
        data = self._get_json(f"pubchem:name:{name.lower()}",
                              f"{PUBCHEM}/compound/name/{quoted}/property/InChIKey/JSON")
        props = (data or {}).get("PropertyTable", {}).get("Properties", [])
        return props[0].get("InChIKey") if props else None

    # ------------------------------------------------------------------ #
    # ChEMBL
    # ------------------------------------------------------------------ #
    def chembl_id(self, inchikey: str) -> str | None:
        url = (f"{CHEMBL}/molecule.json?molecule_structures__standard_inchi_key="
               f"{inchikey}&limit=1")
        data = self._get_json(f"chembl:ik:{inchikey}", url)
        mols = (data or {}).get("molecules", [])
        return mols[0].get("molecule_chembl_id") if mols else None

    def chembl_name_to_inchikey(self, name: str) -> str | None:
        quoted = urllib.parse.quote(name, safe="")
        url = f"{CHEMBL}/molecule.json?pref_name__iexact={quoted}&limit=1"
        data = self._get_json(f"chembl:name:{name.lower()}", url)
        mols = (data or {}).get("molecules", [])
        structures = (mols[0].get("molecule_structures") or {}) if mols else {}
        return structures.get("standard_inchi_key")

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def verify(self, inchikey: str, label: str = "", check_label: bool = True) -> dict:
        """Return verification columns for one prediction."""
        out: dict[str, object] = {}
        if not inchikey:
            return out
        for src in self.sources:
            lookup = self.pubchem_cid if src == "pubchem" else self.chembl_id
            name_lookup = (self.pubchem_name_to_inchikey if src == "pubchem"
                           else self.chembl_name_to_inchikey)
            try:
                ident = lookup(inchikey)
                out[f"{src}_id"] = ident or ""
                out[f"{src}_found"] = ident is not None
            except requests.RequestException as exc:
                log.warning("%s lookup failed for %s: %s", src, inchikey, exc)
                out[f"{src}_found"] = None   # unknown, not "absent"
            if check_label and label:
                try:
                    ref_key = name_lookup(label)
                    out[f"{src}_label_inchikey"] = ref_key or ""
                    out[f"{src}_label_match"] = (None if ref_key is None
                                                 else ref_key == inchikey)
                except requests.RequestException as exc:
                    log.warning("%s name lookup failed for '%s': %s", src, label, exc)
        return out

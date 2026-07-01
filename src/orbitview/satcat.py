"""Searchable index of all active on-orbit satellites (CelesTrak catalog).

Downloads CelesTrak's "active" general-perturbations CSV (~16k objects),
caches it, and keeps a slim in-memory {norad_id, name, intl_id} index for fast
substring search. Any result can then be tracked via the usual TLE fetch.
"""

from __future__ import annotations

import csv
import threading
import time
import urllib.error
import urllib.request

from . import config

CATALOG_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv"
CATALOG_MAX_AGE_SECONDS = 24 * 60 * 60

_LOCK = threading.Lock()
_INDEX: list[dict] | None = None
_LOADED_AT = 0.0


class CatalogError(RuntimeError):
    """Raised when the satellite catalog cannot be obtained."""


def _cache_path():
    return config.CACHE_DIR / "satcat_active.csv"


def _download() -> bytes:
    request = urllib.request.Request(CATALOG_URL, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _ensure_csv():
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CATALOG_MAX_AGE_SECONDS
    if not fresh:
        try:
            data = _download()
            if b"NORAD_CAT_ID" in data:  # basic sanity check before caching
                path.write_bytes(data)
        except (OSError, urllib.error.URLError) as exc:
            if not path.exists():
                raise CatalogError("Could not download the satellite catalog.") from exc
    return path


def _parse(path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                norad_id = int(row["NORAD_CAT_ID"])
            except (KeyError, ValueError):
                continue
            rows.append(
                {
                    "norad_id": norad_id,
                    "name": (row.get("OBJECT_NAME") or "").strip(),
                    "intl_id": (row.get("OBJECT_ID") or "").strip(),
                }
            )
    return rows


def _index() -> list[dict]:
    global _INDEX, _LOADED_AT
    with _LOCK:
        if _INDEX is not None and (time.time() - _LOADED_AT) < CATALOG_MAX_AGE_SECONDS:
            return _INDEX
    parsed = _parse(_ensure_csv())
    with _LOCK:
        _INDEX = parsed
        _LOADED_AT = time.time()
    return parsed


def search(query: str, *, limit: int = 50) -> dict:
    """Substring search over satellite names and NORAD/international ids."""
    index = _index()
    q = query.strip().lower()
    if not q:
        return {"count": len(index), "query": query, "results": []}

    digits = q.isdigit()
    scored: list[tuple[int, dict]] = []
    for entry in index:
        if digits:
            sid = str(entry["norad_id"])
            if sid.startswith(q):
                scored.append((0, entry))
            elif q in sid:
                scored.append((2, entry))
        else:
            name = entry["name"].lower()
            pos = name.find(q)
            if pos == 0:
                scored.append((0, entry))
            elif pos > 0:
                scored.append((1, entry))
            elif q in entry["intl_id"].lower():
                scored.append((2, entry))
    scored.sort(key=lambda item: (item[0], item[1]["name"]))
    return {
        "count": len(index),
        "query": query,
        "results": [entry for _, entry in scored[:limit]],
    }


def catalog_size() -> int:
    return len(_index())

"""Whole-constellation tracking — fetch a CelesTrak group and propagate members.

A constellation (Starlink, GPS, OneWeb, …) is far too large to track as
individual objects, so it is rendered as a lightweight cloud of current
positions. We download and cache the group's TLE set, then propagate (a capped,
evenly-sampled subset of) its members to the requested time.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from skyfield.api import EarthSatellite, wgs84

from . import config, satcat
from . import tle as tle_module
from .propagation import _timescale

GROUP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
GROUP_MAX_AGE_SECONDS = 12 * 60 * 60
NAME_SEARCH_LIMIT = 200  # max satellites for a name-based search (TLE per satellite)
MAX_MEMBERS = 1200  # cap propagated/rendered members so huge groups stay responsive

# Curated CelesTrak groups offered in the picker.
CONSTELLATIONS = (
    {"id": "starlink", "name": "Starlink", "group": "starlink"},
    {"id": "oneweb", "name": "OneWeb", "group": "oneweb"},
    {"id": "gps", "name": "GPS", "group": "gps-ops"},
    {"id": "galileo", "name": "Galileo", "group": "galileo"},
    {"id": "glonass", "name": "GLONASS", "group": "glo-ops"},
    {"id": "beidou", "name": "BeiDou", "group": "beidou"},
    {"id": "iridium", "name": "Iridium NEXT", "group": "iridium-NEXT"},
    {"id": "globalstar", "name": "Globalstar", "group": "globalstar"},
    {"id": "planet", "name": "Planet", "group": "planet"},
    {"id": "geo", "name": "Geostationary", "group": "geo"},
)
CONST_BY_ID = {c["id"]: c for c in CONSTELLATIONS}

_LOCK = threading.Lock()
# group -> (list of (name, line1, line2) triples, fetched_at). We cache the raw
# elements (cheap) and only build EarthSatellite objects for the sampled subset.
_ELEMENTS: dict[str, tuple[list, float]] = {}


class ConstellationError(RuntimeError):
    """Raised when a constellation's elements cannot be obtained."""


def list_constellations() -> list[dict]:
    return [dict(c) for c in CONSTELLATIONS]


def _cache_path(group: str):
    return config.CACHE_DIR / "groups" / f"{group}.tle"


def _download(group: str) -> bytes:
    request = urllib.request.Request(
        GROUP_URL.format(group=group), headers={"User-Agent": config.USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _ensure_tle(group: str):
    path = _cache_path(group)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < GROUP_MAX_AGE_SECONDS
    if not fresh:
        try:
            data = _download(group)
            if b"1 " in data and b"2 " in data:
                path.write_bytes(data)
        except (OSError, urllib.error.URLError) as exc:
            if not path.exists():
                raise ConstellationError(f"Could not download constellation '{group}'.") from exc
    return path


def _parse_tle_lines(lines: list[str]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            triples.append(("", lines[i], lines[i + 1]))
            i += 2
        elif (i + 2 < len(lines) and lines[i + 1].startswith("1 ")
              and lines[i + 2].startswith("2 ")):
            triples.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return triples


def _parse_elements(path) -> list[tuple[str, str, str]]:
    lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip()]
    return _parse_tle_lines(lines)


def _group_elements(group: str) -> list[tuple[str, str, str]]:
    with _LOCK:
        cached = _ELEMENTS.get(group)
        if cached is not None and (time.time() - cached[1]) < GROUP_MAX_AGE_SECONDS:
            return cached[0]
    triples = _parse_elements(_ensure_tle(group))
    with _LOCK:
        _ELEMENTS[group] = (triples, time.time())
    return triples


def positions(
    constellation_id: str, *, epoch: float | None = None, limit: int = MAX_MEMBERS
) -> dict:
    """Current (or epoch-time) subpoints of a constellation's members."""
    meta = CONST_BY_ID.get(constellation_id)
    group = meta["group"] if meta else constellation_id
    triples = _group_elements(group)
    total = len(triples)
    if total > limit:  # evenly sample so the cloud still shows the full pattern
        step = (total + limit - 1) // limit
        triples = triples[::step]

    ts = _timescale()
    when = datetime.fromtimestamp(epoch, UTC) if epoch is not None else datetime.now(UTC)
    t = ts.from_datetime(when)

    members = []
    for name, line1, line2 in triples:
        try:
            norad_id = int(line1[2:7])
        except ValueError:
            norad_id = None
        try:
            sp = wgs84.subpoint(EarthSatellite(line1, line2, name, ts).at(t))
            members.append(
                {
                    "norad_id": norad_id,
                    "name": name or (f"NORAD {norad_id}" if norad_id else "Unknown"),
                    "lat": float(sp.latitude.degrees),
                    "lng": float(sp.longitude.degrees),
                    "alt_km": float(sp.elevation.km),
                }
            )
        except Exception:  # noqa: BLE001 — a member that fails to propagate is skipped
            continue

    return {
        "id": constellation_id,
        "name": meta["name"] if meta else constellation_id,
        "group": group,
        "total": total,
        "shown": len(members),
        "members": members,
    }


def _propagate_triples(
    triples: list[tuple[str, str, str]],
    epoch: float | None,
    limit: int,
) -> tuple[int, list[dict]]:
    """Propagate a (possibly capped) list of TLE triples to subpoints."""
    total = len(triples)
    sample = triples
    if total > limit:
        step = (total + limit - 1) // limit
        sample = triples[::step]

    ts = _timescale()
    when = datetime.fromtimestamp(epoch, UTC) if epoch is not None else datetime.now(UTC)
    t = ts.from_datetime(when)

    members = []
    for name, line1, line2 in sample:
        try:
            norad_id = int(line1[2:7])
        except ValueError:
            norad_id = None
        try:
            sp = wgs84.subpoint(EarthSatellite(line1, line2, name, ts).at(t))
            members.append(
                {
                    "norad_id": norad_id,
                    "name": name or (f"NORAD {norad_id}" if norad_id else "Unknown"),
                    "lat": float(sp.latitude.degrees),
                    "lng": float(sp.longitude.degrees),
                    "alt_km": float(sp.elevation.km),
                }
            )
        except Exception:  # noqa: BLE001
            continue
    return total, members


def positions_by_name(
    name_query: str, *, epoch: float | None = None, limit: int = NAME_SEARCH_LIMIT
) -> dict:
    """Positions of satellites whose name contains ``name_query``.

    Uses the local satcat index for fast substring matching, then fetches TLEs
    for matched satellites concurrently (with per-satellite disk caching).
    Capped at ``NAME_SEARCH_LIMIT`` to keep TLE fetches bounded.
    """
    q = name_query.strip()
    empty = {"id": f"name:{q}", "name": q.title(), "group": f"name:{q.upper()}",
             "total": 0, "shown": 0, "members": []}
    if not q:
        return empty

    cap = min(limit, NAME_SEARCH_LIMIT)
    cat = satcat.search(q, limit=cap)
    matched = cat["results"]
    if not matched:
        return empty

    # Fetch TLEs concurrently — most will be cache hits after first load.
    triples: list[tuple[str, str, str]] = []
    workers = min(20, len(matched))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(tle_module.get_tle, s["norad_id"]): s for s in matched}
        for fut in as_completed(futs):
            try:
                t = fut.result()
                triples.append((t.name, t.line1, t.line2))
            except Exception:  # noqa: BLE001 — skip satellites with no TLE
                continue

    _total, members = _propagate_triples(triples, epoch, limit=len(triples))
    capped = len(matched) >= cap
    label = f"{q.title()} ({len(matched)}{'+'  if capped else ''})"
    return {
        "id": f"name:{q}",
        "name": label,
        "group": f"name:{q.upper()}",
        "total": len(matched),
        "shown": len(members),
        "members": members,
    }

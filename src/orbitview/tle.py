"""Fetching and caching of Two-Line Element (TLE) orbital data.

TLEs are cached per NORAD id under ``<cache>/tle/<id>.tle`` and re-downloaded
from CelesTrak when older than :data:`config.TLE_MAX_AGE_SECONDS`. A stale or
unreachable network falls back to the cached copy so the app keeps working
offline once primed.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config


class TleError(RuntimeError):
    """Raised when TLE data cannot be obtained from network or cache."""


@dataclass(frozen=True)
class Tle:
    """A parsed TLE set."""

    norad_id: int
    name: str
    line1: str
    line2: str
    fetched_at: float  # epoch seconds of the cache file's mtime


def _cache_path(norad_id: int) -> Path:
    return config.CACHE_DIR / "tle" / f"{norad_id}.tle"


def _download(norad_id: int) -> bytes:
    url = config.CELESTRAK_GP_URL.format(norad_id=norad_id)
    request = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_SECONDS) as response:
        return response.read()


def _parse(norad_id: int, text: str, fetched_at: float) -> Tle:
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise TleError(f"TLE for {norad_id} is incomplete.")

    # CelesTrak returns either 2 lines (just the elements) or 3 (name + elements).
    if len(lines) >= 3 and not lines[0].startswith("1 "):
        name, line1, line2 = lines[0], lines[1], lines[2]
    else:
        name, line1, line2 = f"NORAD {norad_id}", lines[0], lines[1]

    if not (line1.startswith("1 ") and line2.startswith("2 ")):
        raise TleError(
            f"Response for {norad_id} was not a TLE — the satellite may not exist "
            "in the catalog."
        )
    return Tle(norad_id, name.strip(), line1, line2, fetched_at)


def get_tle(norad_id: int, *, force_refresh: bool = False) -> Tle:
    """Return a fresh TLE for ``norad_id``, using the cache when possible."""
    path = _cache_path(norad_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    cached_age = (time.time() - path.stat().st_mtime) if path.exists() else None
    is_fresh = cached_age is not None and cached_age < config.TLE_MAX_AGE_SECONDS

    if not force_refresh and is_fresh:
        return _parse(norad_id, path.read_text(encoding="utf-8"), path.stat().st_mtime)

    try:
        raw = _download(norad_id)
        text = raw.decode("utf-8", errors="replace")
        parsed = _parse(norad_id, text, time.time())  # validate before caching
        path.write_bytes(raw)
        return parsed
    except (OSError, urllib.error.URLError) as exc:
        if path.exists():
            # Network failed but we have something cached — use it rather than die.
            return _parse(norad_id, path.read_text(encoding="utf-8"), path.stat().st_mtime)
        raise TleError(
            f"Could not download TLE for {norad_id} and no cached copy is available."
        ) from exc

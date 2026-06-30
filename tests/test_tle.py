"""Tests for TLE parsing and caching (no network required)."""

from __future__ import annotations

import time

import pytest

from iss_tracker import tle

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9000
2 25544  51.6400 208.0000 0006703 130.0000 325.0000 15.50000000    07
"""

TWO_LINE_ONLY = (
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9000\n"
    "2 25544  51.6400 208.0000 0006703 130.0000 325.0000 15.50000000    07\n"
)


def test_parse_three_line_tle():
    parsed = tle._parse(25544, ISS_TLE, time.time())
    assert parsed.name == "ISS (ZARYA)"
    assert parsed.line1.startswith("1 25544U")
    assert parsed.line2.startswith("2 25544")


def test_parse_two_line_tle_synthesizes_name():
    parsed = tle._parse(25544, TWO_LINE_ONLY, time.time())
    assert parsed.name == "NORAD 25544"
    assert parsed.line1.startswith("1 ")
    assert parsed.line2.startswith("2 ")


def test_parse_rejects_non_tle_payload():
    with pytest.raises(tle.TleError):
        tle._parse(999999, "No GP data found", time.time())


def test_parse_rejects_truncated():
    with pytest.raises(tle.TleError):
        tle._parse(25544, "1 25544U only one line", time.time())


def test_get_tle_uses_cache(tmp_path, monkeypatch):
    """A fresh cache file is read without hitting the network."""
    from iss_tracker import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    cache_file = tmp_path / "tle" / "25544.tle"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(ISS_TLE, encoding="utf-8")

    def explode(_norad_id):  # pragma: no cover - must not be called
        raise AssertionError("network should not be used for a fresh cache")

    monkeypatch.setattr(tle, "_download", explode)
    parsed = tle.get_tle(25544)
    assert parsed.name == "ISS (ZARYA)"


def test_get_tle_falls_back_to_stale_cache_on_network_error(tmp_path, monkeypatch):
    from iss_tracker import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    cache_file = tmp_path / "tle" / "25544.tle"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(ISS_TLE, encoding="utf-8")
    # Make the cache look stale so a refresh is attempted.
    old = time.time() - 10 * 3600
    import os

    os.utime(cache_file, (old, old))

    def explode(_norad_id):
        raise OSError("network down")

    monkeypatch.setattr(tle, "_download", explode)
    parsed = tle.get_tle(25544)  # should not raise — falls back to stale cache
    assert parsed.line2.startswith("2 25544")

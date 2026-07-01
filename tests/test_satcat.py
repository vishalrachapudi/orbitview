"""Tests for the satellite catalog search (offline; index monkeypatched)."""

from __future__ import annotations

import pytest

from orbitview import satcat

INDEX = [
    {"norad_id": 25544, "name": "ISS (ZARYA)", "intl_id": "1998-067A"},
    {"norad_id": 20580, "name": "HST", "intl_id": "1990-037B"},
    {"norad_id": 48274, "name": "CSS (TIANHE)", "intl_id": "2021-035A"},
    {"norad_id": 44713, "name": "STARLINK-1007", "intl_id": "2019-074A"},
    {"norad_id": 44714, "name": "STARLINK-1008", "intl_id": "2019-074B"},
]


@pytest.fixture(autouse=True)
def _fixed_index(monkeypatch):
    monkeypatch.setattr(satcat, "_index", lambda: INDEX)


def test_empty_query_returns_count_no_results():
    out = satcat.search("")
    assert out["count"] == 5
    assert out["results"] == []


def test_name_search_prefix_ranks_first():
    results = satcat.search("starlink")["results"]
    assert len(results) == 2
    assert all("STARLINK" in r["name"] for r in results)


def test_name_search_is_case_insensitive_substring():
    results = satcat.search("zarya")["results"]
    assert results[0]["norad_id"] == 25544


def test_numeric_query_matches_norad_id():
    results = satcat.search("205")["results"]
    assert results[0]["norad_id"] == 20580


def test_limit_is_respected():
    assert len(satcat.search("starlink", limit=1)["results"]) == 1

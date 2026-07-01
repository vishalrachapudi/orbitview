"""Tests for constellation propagation (offline; group satellites stubbed)."""

from __future__ import annotations

import pytest

from orbitview import constellations

L1 = "1 25544U 98067A   24160.50000000  .00016717  00000-0  10270-3 0  9007"
L2 = "2 25544  51.6400 208.0000 0006703 130.0000 325.0000 15.50000000    05"


def _triples(n):
    return [(f"SAT-{i}", L1, L2) for i in range(n)]


@pytest.fixture(autouse=True)
def _fixed_group(monkeypatch):
    monkeypatch.setattr(constellations, "_group_elements", lambda group: _triples(5))


def test_list_constellations_has_known_groups():
    ids = {c["id"] for c in constellations.list_constellations()}
    assert {"starlink", "gps", "oneweb", "geo"} <= ids


def test_positions_shape_and_ranges():
    out = constellations.positions("gps")
    assert out["total"] == 5 and out["shown"] == 5
    assert len(out["members"]) == 5
    for m in out["members"]:
        assert m["norad_id"] == 25544
        assert m["name"]
        assert -90 <= m["lat"] <= 90
        assert -180 <= m["lng"] <= 180
        assert m["alt_km"] > 0


def test_positions_caps_and_samples(monkeypatch):
    monkeypatch.setattr(constellations, "_group_elements", lambda group: _triples(50))
    out = constellations.positions("starlink", limit=10)
    assert out["total"] == 50
    assert out["shown"] <= 10


def test_positions_accepts_epoch():
    out = constellations.positions("gps", epoch=2_000_000_000.0)
    assert out["shown"] == 5

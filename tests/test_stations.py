"""Structural tests for the ground-station network."""

from __future__ import annotations

from orbitview.stations import STATIONS, STATIONS_BY_ID, stations_as_dicts


def test_station_ids_unique():
    ids = [s.station_id for s in STATIONS]
    assert len(ids) == len(set(ids))


def test_stations_by_id_complete():
    assert set(STATIONS_BY_ID) == {s.station_id for s in STATIONS}


def test_station_coordinates_valid():
    for s in STATIONS:
        assert -90 <= s.lat <= 90
        assert -180 <= s.lon <= 180
        assert 0 <= s.elevation_mask_deg <= 30


def test_stations_as_dicts_shape():
    rows = stations_as_dicts()
    assert len(rows) == len(STATIONS)
    assert {"station_id", "name", "lat", "lon", "elevation_mask_deg"} <= rows[0].keys()

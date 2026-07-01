"""End-to-end API tests with the propagation layer stubbed (no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orbitview import propagation, server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        propagation,
        "satellite_info",
        lambda norad_id: {
            "norad_id": norad_id,
            "name": "ISS (ZARYA)",
            "category": "Space station",
            "blurb": "test",
            "position": {
                "lat": 1.0, "lng": 2.0, "alt_km": 420.0, "speed_kms": 7.6, "sunlit": True,
            },
            "orbit": {
                "period_min": 92.0, "inclination_deg": 51.6,
                "eccentricity": 0.0006, "revs_per_day": 15.5,
            },
            "tle": {"line1": "1 ...", "line2": "2 ...", "epoch_age_days": 0.5, "fetched_at": 0.0},
            "server_time": "2026-06-16T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        propagation,
        "ground_track",
        lambda norad_id, **kw: {
            "norad_id": norad_id,
            "now": 1000.0,
            "samples": [
                {"t": 999.0, "lat": 0, "lng": 0, "alt_km": 420, "speed_kms": 7.6, "sunlit": True},
            ],
            "subsolar": {"lat": 23.0, "lng": 10.0},
        },
    )
    monkeypatch.setattr(
        propagation,
        "predict_passes",
        lambda norad_id, **kw: [
            {"rise": "2026-06-16T12:00:00+00:00", "peak": "2026-06-16T12:05:00+00:00",
             "set": "2026-06-16T12:10:00+00:00", "peak_elevation_deg": 45.0,
             "peak_azimuth_deg": 180.0, "visible": True, "sat_sunlit": True,
             "observer_sun_alt_deg": -10.0}
        ],
    )
    monkeypatch.setattr(
        propagation, "predict_contacts",
        lambda norad_id, stations, **kw: [
            {"station_id": "svalbard", "station_name": "Svalbard", "aos_utc": "x", "los_utc": "y",
             "aos_epoch": 1.0, "los_epoch": 2.0, "duration_s": 600.0, "max_elevation_deg": 40.0,
             "aos_azimuth_deg": 10.0, "los_azimuth_deg": 200.0, "tca_utc": "z"}
        ],
    )
    monkeypatch.setattr(
        propagation, "contact_profile",
        lambda norad_id, station, **kw: {
            "norad_id": norad_id, "station_id": station.station_id,
            "downlink_hz": kw.get("downlink_hz"),
            "samples": [{"t": 1.0, "alt_deg": 10, "az_deg": 20, "range_km": 2000,
                         "range_rate_kms": -5, "doppler_hz": 1000}],
        },
    )
    monkeypatch.setattr(
        propagation, "eclipse_scan",
        lambda norad_id, **kw: {"norad_id": norad_id, "eph_available": True, "beta_deg": 42.0,
                                "period_min": 92.0, "events": [], "per_orbit": []},
    )
    monkeypatch.setattr(
        propagation, "orbital_elements",
        lambda norad_id: {
            "norad_id": norad_id, "name": "ISS",
            "elements": {"semi_major_axis_km": 6796.0},
        },
    )
    monkeypatch.setattr(
        propagation, "target_access",
        lambda norad_id, **kw: [{"start_epoch": 1.0, "end_epoch": 2.0, "max_elevation_deg": 50.0,
                                 "min_off_nadir_deg": 20.0, "min_slant_range_km": 500.0}],
    )
    monkeypatch.setattr(
        propagation, "event_timeline",
        lambda norad_id, **kw: {
            "norad_id": norad_id, "eph_available": True, "window_hours": 24,
            "events": [{"t": 1.0, "type": "apogee", "label": "Apogee", "detail": {}}],
        },
    )
    return TestClient(server.app)


def test_catalog_lists_iss(client):
    data = client.get("/api/catalog").json()
    assert data["default_norad_id"] == 25544
    assert any(s["norad_id"] == 25544 for s in data["satellites"])


def test_satellite_endpoint(client):
    data = client.get("/api/satellite/25544").json()
    assert data["name"] == "ISS (ZARYA)"
    assert data["orbit"]["period_min"] == 92.0


def test_track_endpoint(client):
    data = client.get("/api/track/25544").json()
    assert data["samples"][0]["alt_km"] == 420
    assert data["subsolar"]["lat"] == 23.0


def test_passes_endpoint_requires_coords(client):
    assert client.get("/api/passes/25544").status_code == 422  # missing lat/lon
    data = client.get("/api/passes/25544?lat=37.7&lon=-122.4").json()
    assert data["passes"][0]["visible"] is True


def test_passes_rejects_out_of_range_lat(client):
    assert client.get("/api/passes/25544?lat=200&lon=0").status_code == 422


def test_track_epoch_param_accepted(client):
    assert client.get("/api/track/25544?epoch=1750000000").status_code == 200


def test_stations_endpoint(client):
    data = client.get("/api/stations").json()
    assert any(s["station_id"] == "svalbard" for s in data["stations"])


def test_contacts_endpoint(client):
    data = client.get("/api/contacts/25544").json()
    assert data["contacts"][0]["station_id"] == "svalbard"
    assert data["stations"]  # defaulted to full network


def test_contacts_rejects_unknown_station(client):
    assert client.get("/api/contacts/25544?station_id=nope").status_code == 422


def test_contacts_rejects_bad_custom_station(client):
    assert client.get("/api/contacts/25544?station=Foo,notnum,0").status_code == 422


def test_contact_profile_requires_station(client):
    assert client.get("/api/contacts/25544/profile?aos=a&los=b").status_code == 422
    ok = client.get("/api/contacts/25544/profile?aos=a&los=b&station_id=svalbard")
    assert ok.status_code == 200
    assert ok.json()["samples"][0]["doppler_hz"] == 1000


def test_eclipses_endpoint(client):
    assert client.get("/api/eclipses/25544").json()["beta_deg"] == 42.0


def test_elements_endpoint(client):
    assert client.get("/api/elements/25544").json()["elements"]["semi_major_axis_km"] == 6796.0


def test_access_endpoint(client):
    assert client.get("/api/access/25544").status_code == 422  # missing lat/lon
    data = client.get("/api/access/25544?lat=40&lon=-75&max_off_nadir_deg=30").json()
    assert data["windows"][0]["min_off_nadir_deg"] == 20.0


def test_events_endpoint(client):
    data = client.get("/api/events/25544?hours=12").json()
    assert data["events"][0]["type"] == "apogee"


def test_search_endpoint(client, monkeypatch):
    from orbitview import satcat
    monkeypatch.setattr(
        satcat, "search",
        lambda q, limit=50: {
            "count": 15000, "query": q,
            "results": [{"norad_id": 25544, "name": "ISS (ZARYA)", "intl_id": "1998-067A"}],
        },
    )
    data = client.get("/api/search?q=iss").json()
    assert data["count"] == 15000
    assert data["results"][0]["norad_id"] == 25544


def test_constellations_list_endpoint(client):
    data = client.get("/api/constellations").json()
    assert any(c["id"] == "starlink" for c in data["constellations"])


def test_constellation_positions_endpoint(client, monkeypatch):
    from orbitview import constellations
    monkeypatch.setattr(
        constellations, "positions",
        lambda cid, **kw: {
            "id": cid, "name": "GPS", "group": "gps-ops", "total": 31, "shown": 31,
            "members": [
                {"norad_id": 44506, "name": "GPS-3", "lat": 1.0, "lng": 2.0, "alt_km": 20180.0},
            ],
        },
    )
    data = client.get("/api/constellation/gps").json()
    assert data["total"] == 31
    assert data["members"][0]["alt_km"] == 20180.0


def test_index_html_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "ORBIT" in res.text

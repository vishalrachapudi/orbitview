"""Tests for the propagation layer using a fixed offline TLE."""

from __future__ import annotations

import time

import pytest

from iss_tracker import propagation, tle
from iss_tracker.stations import GroundStation

# A mid-latitude station that reliably sees the ISS (incl 51.6°) within a day.
TEST_STATION = GroundStation("test", "Test Station", 40.0, -75.0, 0.0, 5.0, "Test")

# A stable, well-formed ISS TLE used so tests never depend on the network.
FIXED_TLE = tle.Tle(
    norad_id=25544,
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   24160.50000000  .00016717  00000-0  10270-3 0  9007",
    line2="2 25544  51.6400 208.0000 0006703 130.0000 325.0000 15.50000000    05",
    fetched_at=time.time(),
)


@pytest.fixture(autouse=True)
def _force_fixed_tle(monkeypatch):
    """Make get_satellite build from FIXED_TLE without touching network/cache."""
    propagation._SAT_CACHE.clear()
    monkeypatch.setattr(tle, "get_tle", lambda norad_id, **_: FIXED_TLE)
    yield
    propagation._SAT_CACHE.clear()


def test_satellite_info_shape_and_ranges():
    info = propagation.satellite_info(25544)
    pos = info["position"]
    assert -90 <= pos["lat"] <= 90
    assert -180 <= pos["lng"] <= 180
    # The ISS orbits ~400-430 km up at ~7.6 km/s.
    assert 300 < pos["alt_km"] < 600
    assert 6 < pos["speed_kms"] < 9
    # Orbital period of the ISS is roughly 90-93 minutes.
    assert 88 < info["orbit"]["period_min"] < 95
    assert 50 < info["orbit"]["inclination_deg"] < 53


def test_ground_track_is_sampled_and_ordered():
    track = propagation.ground_track(
        25544, minutes_before=10, minutes_after=10, step_seconds=30
    )
    samples = track["samples"]
    assert len(samples) > 20
    times = [s["t"] for s in samples]
    assert times == sorted(times)
    for s in samples:
        assert -90 <= s["lat"] <= 90
        assert -180 <= s["lng"] <= 180


def test_ground_track_window_brackets_now():
    track = propagation.ground_track(
        25544, minutes_before=15, minutes_after=20, step_seconds=60
    )
    first, last = track["samples"][0]["t"], track["samples"][-1]["t"]
    assert first < track["now"] < last


def test_predict_passes_returns_well_formed_events():
    passes = propagation.predict_passes(
        25544, lat=37.77, lon=-122.42, days=3.0, min_elevation_deg=10.0
    )
    assert isinstance(passes, list)
    for p in passes:
        assert "peak" in p
        assert 10 <= p["peak_elevation_deg"] <= 90
        assert 0 <= p["peak_azimuth_deg"] <= 360
        assert isinstance(p["visible"], bool)


def test_ground_track_epoch_centers_window():
    epoch = time.time() + 3600.0
    track = propagation.ground_track(
        25544, minutes_before=10, minutes_after=10, step_seconds=30, epoch=epoch
    )
    assert abs(track["center"] - epoch) < 1.0
    assert track["samples"][0]["t"] < epoch < track["samples"][-1]["t"]


def test_orbital_elements_physical():
    el = propagation.orbital_elements(25544)["elements"]
    assert 6700 < el["semi_major_axis_km"] < 6900
    assert el["apogee_alt_km"] > el["perigee_alt_km"]
    assert 0 <= el["true_anomaly_deg"] < 360
    assert 0 <= el["raan_deg"] < 360
    assert 88 < el["period_min"] < 95


def test_beta_angle_in_range():
    beta = propagation.beta_angle(25544)
    assert beta is None or -90.0 <= beta <= 90.0


def test_eclipse_scan_shape():
    data = propagation.eclipse_scan(25544, orbits=2.0, step_seconds=15.0)
    assert "events" in data and "per_orbit" in data
    if data["eph_available"]:
        assert -90.0 <= data["beta_deg"] <= 90.0
        for o in data["per_orbit"]:
            assert 0.0 <= o["sunlit_fraction"] <= 1.0


def test_eclipse_scan_degrades_without_ephemeris(monkeypatch):
    monkeypatch.setattr(propagation, "_ephemeris", lambda: None)
    data = propagation.eclipse_scan(25544, orbits=1.0)
    assert data["eph_available"] is False
    assert data["beta_deg"] is None
    assert data["events"] == []


def test_predict_contacts_well_formed():
    contacts = propagation.predict_contacts(25544, [TEST_STATION], days=2.0)
    assert isinstance(contacts, list)
    assert contacts, "expected at least one ISS contact over a mid-lat station in 2 days"
    for c in contacts:
        assert c["aos_epoch"] < c["los_epoch"]
        assert c["duration_s"] > 0
        assert 5.0 <= c["max_elevation_deg"] <= 90.0
        assert 0 <= c["aos_azimuth_deg"] <= 360
    assert [c["aos_epoch"] for c in contacts] == sorted(c["aos_epoch"] for c in contacts)


def test_contact_profile_geometry():
    c = propagation.predict_contacts(25544, [TEST_STATION], days=2.0)[0]
    prof = propagation.contact_profile(
        25544, TEST_STATION, aos_utc=c["aos_utc"], los_utc=c["los_utc"],
        downlink_hz=2.25e9, step_seconds=5.0,
    )
    samples = prof["samples"]
    assert len(samples) > 2
    ranges = [s["range_km"] for s in samples]
    # Closest approach is in the interior, not at AOS/LOS.
    assert min(ranges) < ranges[0] and min(ranges) < ranges[-1]
    # Doppler present and changes sign across TCA (approaching -> receding).
    dopplers = [s["doppler_hz"] for s in samples]
    assert dopplers[0] is not None
    assert max(dopplers) > 0 and min(dopplers) < 0


def test_target_access_respects_off_nadir():
    windows = propagation.target_access(
        25544, lat=40.0, lon=-75.0, days=2.0, max_off_nadir_deg=30.0
    )
    assert isinstance(windows, list)
    for w in windows:
        assert w["start_epoch"] < w["end_epoch"]
        assert w["min_off_nadir_deg"] <= 31.0  # within sampling tolerance


def test_event_timeline_merged_and_sorted():
    data = propagation.event_timeline(25544, stations=[TEST_STATION], hours=6.0)
    events = data["events"]
    assert events, "expected events over a 6-hour window"
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)
    # ISS is near-circular, so no apsis; with Sun data we get eclipse/terminator.
    types = {e["type"] for e in events}
    known = {"umbra_entry", "umbra_exit", "terminator_day", "terminator_night", "contact_aos"}
    assert types & known
    assert "apogee" not in types and "perigee" not in types  # near-circular orbit

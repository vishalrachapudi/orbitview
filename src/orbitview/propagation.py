"""Skyfield-backed orbital propagation: positions, ground tracks, passes.

All heavy astronomy lives here. The web layer in :mod:`server` is a thin shell
that calls these functions and serializes the results to JSON.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from skyfield.api import EarthSatellite, Loader, wgs84

from . import config, tle
from .catalog import CATALOG_BY_ID
from .stations import GroundStation

# --- Physical constants ------------------------------------------------------

MU_EARTH = 398600.4418  # Earth's gravitational parameter, km^3 / s^2
R_EARTH = 6378.137  # Earth equatorial radius, km
SPEED_OF_LIGHT_KMS = 299792.458

# --- Lazily-initialized, process-wide Skyfield singletons --------------------

_LOCK = threading.Lock()
_LOADER: Loader | None = None
_TIMESCALE = None
_EPHEMERIS = None
_EPHEMERIS_TRIED = False

# norad_id -> (EarthSatellite, Tle)
_SAT_CACHE: dict[int, tuple[EarthSatellite, tle.Tle]] = {}


def _loader() -> Loader:
    global _LOADER
    if _LOADER is None:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _LOADER = Loader(str(config.CACHE_DIR), verbose=False)
    return _LOADER


def _timescale():
    global _TIMESCALE
    if _TIMESCALE is None:
        _TIMESCALE = _loader().timescale()
    return _TIMESCALE


def _ephemeris():
    """Return the Sun/Earth ephemeris, or ``None`` if it cannot be loaded.

    Sun geometry (terminator, sunlit checks, visible passes) degrades
    gracefully to *unavailable* rather than failing the whole request.
    """
    global _EPHEMERIS, _EPHEMERIS_TRIED
    if _EPHEMERIS is None and not _EPHEMERIS_TRIED:
        _EPHEMERIS_TRIED = True
        try:
            _EPHEMERIS = _loader()(config.EPHEMERIS_NAME)
        except Exception:  # noqa: BLE001 — any failure means "no Sun data"
            _EPHEMERIS = None
    return _EPHEMERIS


def get_satellite(norad_id: int, *, force_refresh: bool = False) -> tuple[EarthSatellite, tle.Tle]:
    """Return a cached Skyfield satellite, refreshing its TLE when stale."""
    with _LOCK:
        cached = _SAT_CACHE.get(norad_id)
        if cached is not None and not force_refresh:
            _, cached_tle = cached
            if (time.time() - cached_tle.fetched_at) < config.TLE_MAX_AGE_SECONDS:
                return cached

    parsed = tle.get_tle(norad_id, force_refresh=force_refresh)
    satellite = EarthSatellite(parsed.line1, parsed.line2, parsed.name, _timescale())
    with _LOCK:
        _SAT_CACHE[norad_id] = (satellite, parsed)
    return satellite, parsed


# --- Sun geometry ------------------------------------------------------------


@dataclass(frozen=True)
class SubSolar:
    lat: float
    lon: float


def subsolar_point(t) -> SubSolar | None:
    """Geographic point where the Sun is directly overhead, or ``None``."""
    eph = _ephemeris()
    if eph is None:
        return None
    earth, sun = eph["earth"], eph["sun"]
    astrometric = earth.at(t).observe(sun).apparent()
    ra, dec, _ = astrometric.radec(epoch="date")
    lon = (ra.hours - t.gast) * 15.0
    lon = ((lon + 180.0) % 360.0) - 180.0
    return SubSolar(lat=float(dec.degrees), lon=float(lon))


def _sun_direction_gcrs(t, eph):
    """Unit vector(s) Earth→Sun in the GCRS frame; shape (3,) or (3, N)."""
    sun_pos = eph["earth"].at(t).observe(eph["sun"]).apparent().position.km
    return sun_pos / np.linalg.norm(sun_pos, axis=0)


def _beta_deg(geocentric, t, eph) -> float | None:
    """Beta angle: angle between the Sun line and the orbital plane (deg).

    Computed as asin(n · s), where n is the orbit normal (r×v) and s is the
    Sun direction, both in GCRS. Returns ``None`` without an ephemeris.
    """
    if eph is None:
        return None
    r = geocentric.position.km
    v = geocentric.velocity.km_per_s
    h = np.cross(r, v)
    norm = np.linalg.norm(h)
    if norm == 0:
        return None
    n = h / norm
    s = _sun_direction_gcrs(t, eph)
    sin_beta = float(np.dot(n, s))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_beta))))


def _solve_kepler(mean_anomaly_rad: float, ecc: float, iterations: int = 8) -> float:
    """Solve Kepler's equation M = E − e·sin E for the eccentric anomaly E."""
    m = mean_anomaly_rad
    e = E = m if ecc < 0.8 else math.pi
    for _ in range(iterations):
        E = E - (E - ecc * math.sin(E) - m) / (1.0 - ecc * math.cos(E))
    del e
    return E


def _topocentric_series(satellite, observer, t) -> dict:
    """Vectorized alt/az/range/range-rate of a satellite from an observer."""
    topo = (satellite - observer).at(t)
    alt, az, distance = topo.altaz()
    range_km = np.atleast_1d(distance.km).astype(float)
    r = topo.position.km.reshape(3, -1)
    v = topo.velocity.km_per_s.reshape(3, -1)
    range_rate = np.sum(r * v, axis=0) / range_km  # d|r|/dt = (r·v)/|r|
    return {
        "alt_deg": np.atleast_1d(alt.degrees).astype(float),
        "az_deg": np.atleast_1d(az.degrees).astype(float),
        "range_km": range_km,
        "range_rate_kms": np.atleast_1d(range_rate).astype(float),
    }


# --- Current state and ground track -----------------------------------------


def _now_time():
    return _timescale().from_datetime(datetime.now(UTC))


def satellite_info(norad_id: int) -> dict:
    """Return metadata, current position, and orbital parameters."""
    satellite, parsed = get_satellite(norad_id)
    t = _now_time()

    geocentric = satellite.at(t)
    subpoint = wgs84.subpoint(geocentric)
    speed = float(np.linalg.norm(geocentric.velocity.km_per_s))

    eph = _ephemeris()
    sunlit = bool(geocentric.is_sunlit(eph)) if eph is not None else None

    model = satellite.model
    mean_motion_rad_min = model.no_kozai  # radians / minute
    period_min = (2.0 * math.pi / mean_motion_rad_min) if mean_motion_rad_min else None
    epoch_age_days = float(t - satellite.epoch)

    catalog_entry = CATALOG_BY_ID.get(norad_id)
    return {
        "norad_id": norad_id,
        "name": parsed.name,
        "category": catalog_entry.category if catalog_entry else "Satellite",
        "blurb": catalog_entry.blurb if catalog_entry else "",
        "position": {
            "lat": float(subpoint.latitude.degrees),
            "lng": float(subpoint.longitude.degrees),
            "alt_km": float(subpoint.elevation.km),
            "speed_kms": speed,
            "sunlit": sunlit,
        },
        "orbit": {
            "period_min": period_min,
            "inclination_deg": math.degrees(model.inclo),
            "eccentricity": float(model.ecco),
            "revs_per_day": mean_motion_rad_min * 1440.0 / (2.0 * math.pi)
            if mean_motion_rad_min
            else None,
            "beta_deg": _beta_deg(geocentric, t, eph),
        },
        "tle": {
            "line1": parsed.line1,
            "line2": parsed.line2,
            "epoch_age_days": epoch_age_days,
            "fetched_at": parsed.fetched_at,
        },
        "server_time": datetime.now(UTC).isoformat(),
    }


def ground_track(
    norad_id: int,
    *,
    minutes_before: float,
    minutes_after: float,
    step_seconds: float,
    epoch: float | None = None,
) -> dict:
    """Compute a sampled ground track.

    Centered on the current time, or on ``epoch`` (UTC epoch seconds) when
    given — the latter powers the time machine. The returned ``subsolar`` point
    is computed at the window center so a scrubbed terminator stays correct.
    """
    satellite, _ = get_satellite(norad_id)
    ts = _timescale()
    now = datetime.now(UTC)
    center = datetime.fromtimestamp(epoch, UTC) if epoch is not None else now

    start = center - timedelta(minutes=minutes_before)
    total_seconds = (minutes_before + minutes_after) * 60.0
    count = max(2, int(total_seconds / step_seconds) + 1)
    offsets = np.arange(count) * step_seconds
    times = [start + timedelta(seconds=float(s)) for s in offsets]
    t = ts.from_datetimes(times)

    geocentric = satellite.at(t)
    subpoint = wgs84.subpoint(geocentric)
    lats = subpoint.latitude.degrees
    lngs = subpoint.longitude.degrees
    alts = subpoint.elevation.km
    speeds = np.linalg.norm(geocentric.velocity.km_per_s, axis=0)

    eph = _ephemeris()
    if eph is not None:
        sunlit = np.atleast_1d(geocentric.is_sunlit(eph)).astype(bool)
    else:
        sunlit = np.full(count, True)

    samples = [
        {
            "t": times[i].timestamp(),
            "lat": float(lats[i]),
            "lng": float(lngs[i]),
            "alt_km": float(alts[i]),
            "speed_kms": float(speeds[i]),
            "sunlit": bool(sunlit[i]),
        }
        for i in range(count)
    ]

    sub = subsolar_point(ts.from_datetime(center))
    return {
        "norad_id": norad_id,
        "now": now.timestamp(),  # real wall clock — drives the client clock offset
        "center": center.timestamp(),
        "samples": samples,
        "subsolar": ({"lat": sub.lat, "lng": sub.lon} if sub else None),
        "subsolar_epoch": center.timestamp(),
    }


# --- Pass prediction ---------------------------------------------------------


def predict_passes(
    norad_id: int,
    *,
    lat: float,
    lon: float,
    elevation_m: float = 0.0,
    days: float = 3.0,
    min_elevation_deg: float = 10.0,
) -> list[dict]:
    """Predict upcoming visible-from-the-ground passes over an observer.

    Each pass reports rise/peak/set times, peak elevation and azimuth, and a
    ``visible`` flag (satellite sunlit at peak while the observer is in
    darkness) when Sun data is available.
    """
    satellite, _ = get_satellite(norad_id)
    ts = _timescale()
    observer = wgs84.latlon(lat, lon, elevation_m)
    eph = _ephemeris()

    t0 = _now_time()
    t1 = ts.from_datetime(datetime.now(UTC) + timedelta(days=days))
    times, events = satellite.find_events(
        observer, t0, t1, altitude_degrees=min_elevation_deg
    )

    difference = satellite - observer

    def _altaz(t):
        alt, az, _ = difference.at(t).altaz()
        return alt.degrees, az.degrees

    def _observer_sun_alt(t) -> float | None:
        if eph is None:
            return None
        sun = eph["sun"]
        alt, _, _ = (eph["earth"] + observer).at(t).observe(sun).apparent().altaz()
        return float(alt.degrees)

    def _sat_sunlit(t) -> bool | None:
        if eph is None:
            return None
        return bool(satellite.at(t).is_sunlit(eph))

    passes: list[dict] = []
    current: dict | None = None

    for t, event in zip(times, events, strict=True):
        when = t.utc_datetime().isoformat()
        if event == 0:  # rise
            current = {"rise": when}
        elif event == 1:  # culmination
            if current is None:
                current = {}
            alt, az = _altaz(t)
            current["peak"] = when
            current["peak_elevation_deg"] = float(alt)
            current["peak_azimuth_deg"] = float(az)
            sun_alt = _observer_sun_alt(t)
            sat_lit = _sat_sunlit(t)
            current["observer_sun_alt_deg"] = sun_alt
            current["sat_sunlit"] = sat_lit
            current["visible"] = bool(
                sat_lit and sun_alt is not None and sun_alt < -6.0
            )
        elif event == 2:  # set
            if current is None:
                current = {}
            current["set"] = when
            if "peak" in current:
                passes.append(current)
            current = None

    return passes


# --- Orbital state -----------------------------------------------------------


def beta_angle(norad_id: int, t=None) -> float | None:
    """Sun/orbit-plane (beta) angle in degrees, or ``None`` without Sun data."""
    eph = _ephemeris()
    if eph is None:
        return None
    satellite, _ = get_satellite(norad_id)
    if t is None:
        t = _now_time()
    return _beta_deg(satellite.at(t), t, eph)


def orbital_elements(norad_id: int) -> dict:
    """Full Keplerian element set derived from the SGP4 mean elements."""
    satellite, parsed = get_satellite(norad_id)
    t = _now_time()
    model = satellite.model

    n_rad_min = model.no_kozai
    n_rad_s = n_rad_min / 60.0
    period_min = (2.0 * math.pi / n_rad_min) if n_rad_min else None
    a_km = (MU_EARTH / n_rad_s**2) ** (1.0 / 3.0) if n_rad_s else None
    ecc = float(model.ecco)

    mean_anom = model.mo  # radians
    ecc_anom = _solve_kepler(mean_anom, ecc)
    true_anom = 2.0 * math.atan2(
        math.sqrt(1.0 + ecc) * math.sin(ecc_anom / 2.0),
        math.sqrt(1.0 - ecc) * math.cos(ecc_anom / 2.0),
    )

    revs_per_day = n_rad_min * 1440.0 / (2.0 * math.pi) if n_rad_min else None
    try:
        rev_at_epoch: int | None = int(parsed.line2[63:68])
    except (ValueError, IndexError):
        rev_at_epoch = None
    rev_number = None
    if rev_at_epoch is not None and revs_per_day is not None:
        rev_number = rev_at_epoch + int(float(t - satellite.epoch) * revs_per_day)

    return {
        "norad_id": norad_id,
        "name": parsed.name,
        "epoch_utc": satellite.epoch.utc_datetime().isoformat(),
        "epoch_age_days": float(t - satellite.epoch),
        "elements": {
            "semi_major_axis_km": a_km,
            "eccentricity": ecc,
            "inclination_deg": math.degrees(model.inclo),
            "raan_deg": math.degrees(model.nodeo) % 360.0,
            "arg_perigee_deg": math.degrees(model.argpo) % 360.0,
            "true_anomaly_deg": math.degrees(true_anom) % 360.0,
            "mean_anomaly_deg": math.degrees(mean_anom) % 360.0,
            "apogee_alt_km": (a_km * (1.0 + ecc) - R_EARTH) if a_km else None,
            "perigee_alt_km": (a_km * (1.0 - ecc) - R_EARTH) if a_km else None,
            "mean_motion_rev_per_day": revs_per_day,
            "period_min": period_min,
            "rev_at_epoch": rev_at_epoch,
            "rev_number": rev_number,
        },
    }


def eclipse_scan(norad_id: int, *, orbits: float = 3.0, step_seconds: float = 10.0) -> dict:
    """Umbra entry/exit times, per-orbit eclipse duration, and beta angle."""
    satellite, _ = get_satellite(norad_id)
    ts = _timescale()
    model = satellite.model
    period_min = (2.0 * math.pi / model.no_kozai) if model.no_kozai else 92.0
    eph = _ephemeris()

    base = {
        "norad_id": norad_id,
        "period_min": period_min,
        "orbits_scanned": orbits,
        "step_seconds": step_seconds,
    }
    if eph is None:
        return {**base, "eph_available": False, "beta_deg": None, "events": [], "per_orbit": []}

    now = datetime.now(UTC)
    total_seconds = orbits * period_min * 60.0
    count = max(2, int(total_seconds / step_seconds) + 1)
    offsets = np.arange(count) * step_seconds
    times = [now + timedelta(seconds=float(s)) for s in offsets]
    sample_ts = np.array([tt.timestamp() for tt in times])
    t = ts.from_datetimes(times)

    geo = satellite.at(t)
    lit = np.atleast_1d(geo.is_sunlit(eph)).astype(int)

    events = []
    for i in np.where(np.diff(lit) != 0)[0]:
        mid = times[i] + (times[i + 1] - times[i]) / 2
        entry = lit[i + 1] < lit[i]
        events.append(
            {
                "type": "umbra_entry" if entry else "umbra_exit",
                "utc": mid.astimezone(UTC).isoformat(),
                "t": mid.timestamp(),
            }
        )

    per_orbit = []
    orbit_sec = period_min * 60.0
    k = 0
    while k < 33:
        s0 = sample_ts[0] + k * orbit_sec
        if s0 >= sample_ts[-1]:
            break
        mask = (sample_ts >= s0) & (sample_ts < s0 + orbit_sec)
        if not mask.any():
            break
        lit_k = lit[mask].astype(bool)
        dark = int((~lit_k).sum())
        per_orbit.append(
            {
                "orbit_index": k,
                "eclipse_s": dark * step_seconds,
                "sunlit_s": int(lit_k.sum()) * step_seconds,
                "sunlit_fraction": float(lit_k.mean()),
            }
        )
        k += 1

    return {
        **base,
        "eph_available": True,
        "beta_deg": _beta_deg(satellite.at(ts.from_datetime(now)), ts.from_datetime(now), eph),
        "events": events,
        "per_orbit": per_orbit,
    }


# --- Ground-station contacts -------------------------------------------------


def predict_contacts(
    norad_id: int, stations: Sequence[GroundStation], *, days: float = 1.0
) -> list[dict]:
    """Upcoming RF contacts (AOS/LOS) across a set of ground stations."""
    satellite, _ = get_satellite(norad_id)
    ts = _timescale()
    t0 = _now_time()
    t1 = ts.from_datetime(datetime.now(UTC) + timedelta(days=days))

    contacts: list[dict] = []
    for st in stations:
        observer = wgs84.latlon(st.lat, st.lon, st.elevation_m)
        difference = satellite - observer
        times, events = satellite.find_events(
            observer, t0, t1, altitude_degrees=st.elevation_mask_deg
        )
        current: dict | None = None
        for tt, ev in zip(times, events, strict=True):
            alt, az, _ = difference.at(tt).altaz()
            if ev == 0:  # AOS
                current = {"aos_t": tt, "aos_az": float(az.degrees)}
            elif ev == 1:  # culmination / TCA
                if current is None:
                    current = {}
                current["tca_t"] = tt
                current["max_elevation_deg"] = float(alt.degrees)
            elif ev == 2:  # LOS
                if current is None:
                    current = {}
                if "aos_t" in current and "max_elevation_deg" in current:
                    aos = current["aos_t"].utc_datetime()
                    los = tt.utc_datetime()
                    contacts.append(
                        {
                            "station_id": st.station_id,
                            "station_name": st.name,
                            "aos_utc": aos.isoformat(),
                            "los_utc": los.isoformat(),
                            "aos_epoch": aos.timestamp(),
                            "los_epoch": los.timestamp(),
                            "duration_s": (los - aos).total_seconds(),
                            "max_elevation_deg": current["max_elevation_deg"],
                            "aos_azimuth_deg": current.get("aos_az"),
                            "los_azimuth_deg": float(az.degrees),
                            "tca_utc": current["tca_t"].utc_datetime().isoformat()
                            if "tca_t" in current
                            else None,
                        }
                    )
                current = None
    contacts.sort(key=lambda c: c["aos_epoch"])
    return contacts


def contact_profile(
    norad_id: int,
    station: GroundStation,
    *,
    aos_utc: str,
    los_utc: str,
    downlink_hz: float | None = None,
    step_seconds: float = 5.0,
) -> dict:
    """Az/el/range/range-rate (and Doppler) sampled over a single contact."""
    satellite, _ = get_satellite(norad_id)
    ts = _timescale()
    observer = wgs84.latlon(station.lat, station.lon, station.elevation_m)

    aos = _parse_iso(aos_utc)
    los = _parse_iso(los_utc)
    total = max(1.0, (los - aos).total_seconds())
    count = max(2, int(total / step_seconds) + 1)
    offsets = np.arange(count) * step_seconds
    times = [aos + timedelta(seconds=float(s)) for s in offsets]
    series = _topocentric_series(satellite, observer, ts.from_datetimes(times))

    samples = []
    for i in range(count):
        rr = float(series["range_rate_kms"][i])
        doppler = (-rr / SPEED_OF_LIGHT_KMS * downlink_hz) if downlink_hz else None
        samples.append(
            {
                "t": times[i].timestamp(),
                "alt_deg": float(series["alt_deg"][i]),
                "az_deg": float(series["az_deg"][i]),
                "range_km": float(series["range_km"][i]),
                "range_rate_kms": rr,
                "doppler_hz": doppler,
            }
        )
    return {
        "norad_id": norad_id,
        "station_id": station.station_id,
        "downlink_hz": downlink_hz,
        "samples": samples,
    }


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


# --- Target access -----------------------------------------------------------


def target_access(
    norad_id: int,
    *,
    lat: float,
    lon: float,
    elevation_m: float = 0.0,
    days: float = 1.0,
    min_elevation_deg: float | None = None,
    max_off_nadir_deg: float | None = None,
) -> list[dict]:
    """Windows when the satellite can view a ground target.

    Constraint is either a minimum ground elevation or a maximum sensor
    off-nadir angle (converted to a ground elevation using the satellite's
    current altitude, spherical-Earth approximation).
    """
    satellite, _ = get_satellite(norad_id)
    ts = _timescale()

    if min_elevation_deg is not None:
        threshold = min_elevation_deg
    elif max_off_nadir_deg is not None:
        h = float(wgs84.subpoint(satellite.at(_now_time())).elevation.km)
        arg = math.sin(math.radians(max_off_nadir_deg)) * (R_EARTH + h) / R_EARTH
        threshold = math.degrees(math.acos(arg)) if arg < 1.0 else 0.0
    else:
        threshold = 0.0

    observer = wgs84.latlon(lat, lon, elevation_m)
    difference = satellite - observer
    t0 = _now_time()
    t1 = ts.from_datetime(datetime.now(UTC) + timedelta(days=days))
    times, events = satellite.find_events(observer, t0, t1, altitude_degrees=threshold)

    windows: list[dict] = []
    current: dict | None = None
    for tt, ev in zip(times, events, strict=True):
        if ev == 0:
            current = {"start_t": tt}
        elif ev == 1:
            if current is None:
                current = {}
            alt, _, _ = difference.at(tt).altaz()
            current["peak_t"] = tt
            current["max_elevation_deg"] = float(alt.degrees)
        elif ev == 2:
            if current is None:
                current = {}
            if "start_t" in current and "max_elevation_deg" in current:
                start = current["start_t"].utc_datetime()
                end = tt.utc_datetime()
                n = 12
                sub_times = [start + (end - start) * k / n for k in range(n + 1)]
                st = ts.from_datetimes(sub_times)
                series = _topocentric_series(satellite, observer, st)
                sat_alt = np.atleast_1d(wgs84.subpoint(satellite.at(st)).elevation.km)
                sin_eta = np.cos(np.radians(series["alt_deg"])) * R_EARTH / (R_EARTH + sat_alt)
                eta = np.degrees(np.arcsin(np.clip(sin_eta, -1.0, 1.0)))
                windows.append(
                    {
                        "start_utc": start.isoformat(),
                        "peak_utc": current["peak_t"].utc_datetime().isoformat(),
                        "end_utc": end.isoformat(),
                        "start_epoch": start.timestamp(),
                        "end_epoch": end.timestamp(),
                        "duration_s": (end - start).total_seconds(),
                        "max_elevation_deg": current["max_elevation_deg"],
                        "min_off_nadir_deg": float(np.min(eta)),
                        "min_slant_range_km": float(np.min(series["range_km"])),
                    }
                )
            current = None
    return windows


# --- Event timeline ----------------------------------------------------------


def event_timeline(
    norad_id: int,
    *,
    stations: Sequence[GroundStation],
    hours: float = 24.0,
    include: set[str] | None = None,
) -> dict:
    """Merge contacts, eclipse, terminator and apsis events into one feed."""
    include = include or {"contacts", "eclipse", "terminator", "apsis"}
    eph = _ephemeris()
    events: list[dict] = []

    if "contacts" in include and stations:
        for c in predict_contacts(norad_id, stations, days=hours / 24.0):
            events.append(
                {
                    "t": c["aos_epoch"],
                    "utc": c["aos_utc"],
                    "type": "contact_aos",
                    "label": f"AOS {c['station_name']}",
                    "detail": {
                        "station_id": c["station_id"],
                        "max_elevation_deg": c["max_elevation_deg"],
                        "duration_s": c["duration_s"],
                        "end_utc": c["los_utc"],
                    },
                }
            )
            events.append(
                {
                    "t": c["los_epoch"],
                    "utc": c["los_utc"],
                    "type": "contact_los",
                    "label": f"LOS {c['station_name']}",
                    "detail": {"station_id": c["station_id"]},
                }
            )

    if include & {"eclipse", "terminator", "apsis"}:
        satellite, _ = get_satellite(norad_id)
        ts = _timescale()
        now = datetime.now(UTC)
        step = 30.0
        count = max(2, int(hours * 3600.0 / step) + 1)
        offsets = np.arange(count) * step
        times = [now + timedelta(seconds=float(s)) for s in offsets]
        sample_ts = np.array([tt.timestamp() for tt in times])
        t = ts.from_datetimes(times)
        geo = satellite.at(t)
        sat_pos = geo.position.km.reshape(3, -1)
        radius_km = np.linalg.norm(sat_pos, axis=0)

        if "eclipse" in include and eph is not None:
            lit = np.atleast_1d(geo.is_sunlit(eph)).astype(int)
            for i in np.where(np.diff(lit) != 0)[0]:
                mid = times[i] + (times[i + 1] - times[i]) / 2
                entry = lit[i + 1] < lit[i]
                events.append(
                    {
                        "t": mid.timestamp(),
                        "utc": mid.astimezone(UTC).isoformat(),
                        "type": "umbra_entry" if entry else "umbra_exit",
                        "label": "Eclipse entry" if entry else "Eclipse exit",
                        "detail": {},
                    }
                )

        if "terminator" in include and eph is not None:
            sun_unit = _sun_direction_gcrs(t, eph).reshape(3, -1)
            sat_unit = sat_pos / np.linalg.norm(sat_pos, axis=0)
            ground_lit = (np.sum(sat_unit * sun_unit, axis=0) > 0).astype(int)
            for i in np.where(np.diff(ground_lit) != 0)[0]:
                mid = times[i] + (times[i + 1] - times[i]) / 2
                day = ground_lit[i + 1] > ground_lit[i]
                events.append(
                    {
                        "t": mid.timestamp(),
                        "utc": mid.astimezone(UTC).isoformat(),
                        "type": "terminator_day" if day else "terminator_night",
                        "label": "Subpoint sunrise" if day else "Subpoint sunset",
                        "detail": {},
                    }
                )

        # Apogee/perigee from geocentric radius extrema — only meaningful for a
        # non-circular orbit (a near-circular orbit has no useful apsis, and the
        # tiny radius wobble would otherwise spam spurious events).
        if "apsis" in include and radius_km.size > 2 and satellite.model.ecco > 0.001:
            sign = np.sign(np.diff(radius_km))
            for i in np.where(np.diff(sign) != 0)[0]:
                idx = i + 1
                apogee = sign[i] > 0  # rising then falling
                events.append(
                    {
                        "t": sample_ts[idx],
                        "utc": times[idx].astimezone(UTC).isoformat(),
                        "type": "apogee" if apogee else "perigee",
                        "label": "Apogee" if apogee else "Perigee",
                        "detail": {"alt_km": float(radius_km[idx] - R_EARTH)},
                    }
                )

    events.sort(key=lambda e: e["t"])
    return {
        "norad_id": norad_id,
        "window_hours": hours,
        "eph_available": eph is not None,
        "events": events,
    }

"""FastAPI application exposing the orbital backend and serving the web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import constellations, imagery, propagation, satcat
from .catalog import ISS_NORAD_ID, catalog_as_dicts
from .stations import STATIONS, STATIONS_BY_ID, GroundStation, stations_as_dicts
from .tle import TleError

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ISS Tracker", version="1.0.0")


@app.get("/api/catalog")
def get_catalog() -> dict:
    """Curated list of satellites offered in the picker by default."""
    return {"default_norad_id": ISS_NORAD_ID, "satellites": catalog_as_dicts()}


@app.get("/api/search")
def search_catalog(
    q: str = Query("", max_length=64),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """Search the full active-satellite catalog by name or NORAD/international id."""
    try:
        return satcat.search(q, limit=limit)
    except satcat.CatalogError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/constellations")
def get_constellations() -> dict:
    """Constellations (CelesTrak groups) that can be added in one click."""
    return {"constellations": constellations.list_constellations()}


@app.get("/api/constellation/search")
def search_constellation(
    q: str = Query("", max_length=64),
    epoch: float | None = Query(None),
    limit: int = Query(constellations.MAX_MEMBERS, ge=10, le=4000),
) -> dict:
    """Search CelesTrak by satellite name substring and return positions."""
    try:
        return constellations.positions_by_name(q, epoch=epoch, limit=limit)
    except constellations.ConstellationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/constellation/{constellation_id}")
def get_constellation(
    constellation_id: str,
    epoch: float | None = Query(None),
    limit: int = Query(constellations.MAX_MEMBERS, ge=10, le=4000),
) -> dict:
    """Current member positions of a constellation (capped/sampled for size)."""
    try:
        return constellations.positions(constellation_id, epoch=epoch, limit=limit)
    except constellations.ConstellationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/satellite/{norad_id}")
def get_satellite(norad_id: int) -> dict:
    """Metadata, current position, and orbital parameters for one satellite."""
    try:
        return propagation.satellite_info(norad_id)
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/track/{norad_id}")
def get_track(
    norad_id: int,
    minutes_before: float = Query(45.0, ge=1.0, le=720.0),
    minutes_after: float = Query(60.0, ge=1.0, le=1440.0),
    step_seconds: float = Query(8.0, ge=1.0, le=120.0),
    epoch: float | None = Query(None),
) -> dict:
    """Sampled ground track for smooth animation; centered on ``epoch`` if given."""
    try:
        return propagation.ground_track(
            norad_id,
            minutes_before=minutes_before,
            minutes_after=minutes_after,
            step_seconds=step_seconds,
            epoch=epoch,
        )
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/passes/{norad_id}")
def get_passes(
    norad_id: int,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    elevation_m: float = Query(0.0),
    days: float = Query(3.0, ge=0.5, le=10.0),
    min_elevation_deg: float = Query(10.0, ge=0.0, le=60.0),
) -> dict:
    """Upcoming overhead passes for an observer location."""
    try:
        passes = propagation.predict_passes(
            norad_id,
            lat=lat,
            lon=lon,
            elevation_m=elevation_m,
            days=days,
            min_elevation_deg=min_elevation_deg,
        )
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"norad_id": norad_id, "passes": passes}


def _parse_stations(station_id: str, station: str) -> list[GroundStation]:
    """Resolve query params into GroundStation objects (default: full network).

    ``station_id`` is a comma-separated list of built-in ids; ``station`` is a
    semicolon-separated list of custom ``NAME,LAT,LON[,ELEV[,MASK]]`` specs.
    """
    ids = [s for s in station_id.split(",") if s]
    specs = [s for s in station.split(";") if s]
    if not ids and not specs:
        return list(STATIONS)
    resolved: list[GroundStation] = []
    for sid in ids:
        if sid not in STATIONS_BY_ID:
            raise HTTPException(status_code=422, detail=f"Unknown station id: {sid}")
        resolved.append(STATIONS_BY_ID[sid])
    for spec in specs:
        parts = spec.split(",")
        if len(parts) < 3:
            raise HTTPException(
                status_code=422,
                detail=f"Bad station spec '{spec}'; expected NAME,LAT,LON[,ELEV[,MASK]]",
            )
        try:
            name, lat, lon = parts[0], float(parts[1]), float(parts[2])
            elev = float(parts[3]) if len(parts) > 3 else 0.0
            mask = float(parts[4]) if len(parts) > 4 else 5.0
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Bad station spec '{spec}'") from exc
        resolved.append(
            GroundStation(name.lower().replace(" ", "_"), name, lat, lon, elev, mask, "Custom")
        )
    return resolved


@app.get("/api/stations")
def get_stations() -> dict:
    """The built-in ground-station network."""
    return {"stations": stations_as_dicts()}


@app.get("/api/contacts/{norad_id}")
def get_contacts(
    norad_id: int,
    days: float = Query(1.0, ge=0.25, le=10.0),
    station_id: str = Query(""),
    station: str = Query(""),
) -> dict:
    """RF contact schedule over the selected ground stations."""
    stations = _parse_stations(station_id, station)
    try:
        contacts = propagation.predict_contacts(norad_id, stations, days=days)
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "norad_id": norad_id,
        "stations": [s.station_id for s in stations],
        "contacts": contacts,
    }


@app.get("/api/contacts/{norad_id}/profile")
def get_contact_profile(
    norad_id: int,
    aos: str = Query(...),
    los: str = Query(...),
    station_id: str | None = Query(None),
    lat: float | None = Query(None, ge=-90.0, le=90.0),
    lon: float | None = Query(None, ge=-180.0, le=180.0),
    elevation_m: float = Query(0.0),
    mask_deg: float = Query(5.0, ge=0.0, le=30.0),
    downlink_hz: float | None = Query(None, gt=0.0),
    step_seconds: float = Query(5.0, ge=1.0, le=60.0),
) -> dict:
    """Az/el/range/range-rate + Doppler sampled across one contact window."""
    if station_id is not None:
        if station_id not in STATIONS_BY_ID:
            raise HTTPException(status_code=422, detail=f"Unknown station id: {station_id}")
        st = STATIONS_BY_ID[station_id]
    elif lat is not None and lon is not None:
        st = GroundStation("custom", "Custom", lat, lon, elevation_m, mask_deg, "Custom")
    else:
        raise HTTPException(status_code=422, detail="Provide station_id or lat & lon")
    try:
        return propagation.contact_profile(
            norad_id, st, aos_utc=aos, los_utc=los,
            downlink_hz=downlink_hz, step_seconds=step_seconds,
        )
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/eclipses/{norad_id}")
def get_eclipses(
    norad_id: int,
    orbits: float = Query(3.0, ge=0.5, le=16.0),
    step_seconds: float = Query(10.0, ge=1.0, le=60.0),
) -> dict:
    """Umbra entry/exit, per-orbit eclipse duration, and beta angle."""
    try:
        return propagation.eclipse_scan(norad_id, orbits=orbits, step_seconds=step_seconds)
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/elements/{norad_id}")
def get_elements(norad_id: int) -> dict:
    """Full Keplerian orbital element set."""
    try:
        return propagation.orbital_elements(norad_id)
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/access/{norad_id}")
def get_access(
    norad_id: int,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    elevation_m: float = Query(0.0),
    days: float = Query(1.0, ge=0.5, le=10.0),
    min_elevation_deg: float | None = Query(None, ge=0.0, le=90.0),
    max_off_nadir_deg: float | None = Query(None, ge=0.0, le=85.0),
) -> dict:
    """Windows when the satellite can view a ground target."""
    try:
        windows = propagation.target_access(
            norad_id, lat=lat, lon=lon, elevation_m=elevation_m, days=days,
            min_elevation_deg=min_elevation_deg, max_off_nadir_deg=max_off_nadir_deg,
        )
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"norad_id": norad_id, "target": {"lat": lat, "lon": lon}, "windows": windows}


@app.get("/api/events/{norad_id}")
def get_events(
    norad_id: int,
    hours: float = Query(24.0, ge=1.0, le=168.0),
    station_id: str = Query(""),
    station: str = Query(""),
    include: str = Query("contacts,eclipse,terminator,apsis"),
) -> dict:
    """Unified, time-sorted event feed for the focused satellite."""
    stations = _parse_stations(station_id, station)
    include_set = {s.strip() for s in include.split(",") if s.strip()}
    try:
        return propagation.event_timeline(
            norad_id, stations=stations, hours=hours, include=include_set
        )
    except TleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/earth-texture")
def get_earth_texture() -> FileResponse:
    """Serve the cached equirectangular Blue Marble day texture."""
    try:
        path = imagery.earth_texture_path()
    except imagery.ImageryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/earth-night-texture")
def get_earth_night_texture() -> FileResponse:
    """Serve the cached Black Marble night-lights texture."""
    try:
        path = imagery.earth_night_texture_path()
    except imagery.ImageryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/sky-texture")
def get_sky_texture() -> FileResponse:
    """Serve the cached Milky Way / starfield sky panorama."""
    try:
        path = imagery.sky_texture_path()
    except imagery.ImageryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# Mount the SPA last so API routes take precedence. ``html=True`` serves
# index.html at "/" and falls back to it for client-side routing.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

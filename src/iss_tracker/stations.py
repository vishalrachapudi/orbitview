"""A built-in network of ground stations for contact prediction.

Mirrors :mod:`catalog.py`. The list seeds the station picker; operators can
also pass custom stations to the contact endpoints via query parameters.
Coordinates are approximate real sites; elevation masks are typical values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundStation:
    """A ground station / antenna site."""

    station_id: str
    name: str
    lat: float
    lon: float
    elevation_m: float = 0.0
    elevation_mask_deg: float = 5.0  # minimum usable elevation (horizon mask)
    network: str = "default"


# A representative polar/equatorial mix — good for LEO contact coverage.
STATIONS: tuple[GroundStation, ...] = (
    GroundStation("svalbard", "Svalbard (SvalSat)", 78.23, 15.39, 450.0, 3.0, "Polar"),
    GroundStation("troll", "Troll, Antarctica (TrollSat)", -72.01, 2.53, 1270.0, 3.0, "Polar"),
    GroundStation("fairbanks", "Fairbanks, Alaska", 64.97, -147.51, 130.0, 5.0, "Polar"),
    GroundStation("mcmurdo", "McMurdo, Antarctica", -77.85, 166.67, 100.0, 5.0, "Polar"),
    GroundStation("wallops", "Wallops Island, Virginia", 37.94, -75.46, 12.0, 5.0, "Mid-lat"),
    GroundStation("kourou", "Kourou, French Guiana", 5.25, -52.80, 15.0, 5.0, "Equatorial"),
    GroundStation("kokee", "Kokee Park, Hawaii", 22.13, -159.66, 1100.0, 5.0, "Mid-lat"),
    GroundStation("dongara", "Dongara, Australia", -29.05, 115.35, 50.0, 5.0, "Mid-lat"),
)

STATIONS_BY_ID = {s.station_id: s for s in STATIONS}


def stations_as_dicts() -> list[dict]:
    """Return the station network as JSON-serializable dictionaries."""
    return [
        {
            "station_id": s.station_id,
            "name": s.name,
            "lat": s.lat,
            "lon": s.lon,
            "elevation_m": s.elevation_m,
            "elevation_mask_deg": s.elevation_mask_deg,
            "network": s.network,
        }
        for s in STATIONS
    ]

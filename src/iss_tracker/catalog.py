"""A curated catalog of trackable satellites.

The catalog seeds the satellite picker with well-known objects. Any NORAD
catalog number can still be tracked on demand, even if it is not listed here.
"""

from __future__ import annotations

from dataclasses import dataclass

ISS_NORAD_ID = 25544


@dataclass(frozen=True)
class CatalogEntry:
    """A satellite the UI offers by default."""

    norad_id: int
    name: str
    category: str
    blurb: str


# Ordered roughly by how interesting they are to a casual observer. Low-Earth
# orbiters move visibly across the globe; the geostationary entry stays put,
# which is a nice contrast.
CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        25544, "ISS (ZARYA)", "Space station",
        "The International Space Station — humanity's crewed outpost in low Earth orbit.",
    ),
    CatalogEntry(
        48274, "CSS (Tiangong)", "Space station",
        "China's Tiangong space station, crewed and continuously inhabited.",
    ),
    CatalogEntry(
        20580, "Hubble Space Telescope", "Observatory",
        "NASA/ESA's flagship optical space telescope, in orbit since 1990.",
    ),
    CatalogEntry(
        43013, "NOAA-20 (JPSS-1)", "Weather",
        "Polar-orbiting weather satellite imaging the whole planet twice a day.",
    ),
    CatalogEntry(
        25994, "Terra (EOS AM-1)", "Earth science",
        "NASA Earth-observing satellite carrying the MODIS imager.",
    ),
    CatalogEntry(
        49260, "Landsat 9", "Earth science",
        "Latest in the Landsat series, mapping land cover and change.",
    ),
    CatalogEntry(
        51850, "GOES-18 (GOES-West)", "Weather (GEO)",
        "Geostationary weather satellite — appears nearly fixed over the Pacific.",
    ),
)

CATALOG_BY_ID = {entry.norad_id: entry for entry in CATALOG}


def catalog_as_dicts() -> list[dict]:
    """Return the catalog as JSON-serializable dictionaries."""
    return [
        {
            "norad_id": entry.norad_id,
            "name": entry.name,
            "category": entry.category,
            "blurb": entry.blurb,
        }
        for entry in CATALOG
    ]

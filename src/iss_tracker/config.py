"""Runtime configuration and shared constants."""

from __future__ import annotations

import os
from pathlib import Path

# --- Caching -----------------------------------------------------------------

CACHE_DIR = Path(
    os.environ.get("ISS_TRACKER_CACHE_DIR", Path.home() / ".cache" / "iss-tracker")
)

# How long a cached TLE is considered fresh before we re-download it.
TLE_MAX_AGE_SECONDS = 6 * 60 * 60  # 6 hours

# --- Remote data sources -----------------------------------------------------

# CelesTrak "general perturbations" endpoint, queried by NORAD catalog number.
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"

# Equirectangular Earth imagery used as the globe day texture. The first
# reachable URL wins; the result is cached locally and reused thereafter.
EARTH_IMAGE_URLS = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74117/"
    "world.200408.3x5400x2700.jpg",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/"
    "land_shallow_topo_2048.jpg",
    "https://raw.githubusercontent.com/turban/webgl-earth/master/images/"
    "2_no_clouds_4k.jpg",
)

# NASA "Black Marble" night-lights, used to render city lights on the dark side.
EARTH_NIGHT_IMAGE_URLS = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79765/"
    "dnb_land_ocean_ice.2012.3600x1800.jpg",
    "https://unpkg.com/three-globe/example/img/earth-night.jpg",
)

# Equirectangular Milky Way panorama for the sky background (ESO 360° pano,
# with a starfield fallback). Downscaled and cached like the Earth textures.
SKY_IMAGE_URLS = (
    "https://cdn.eso.org/images/large/eso0932a.jpg",
    "https://www.eso.org/public/archives/images/wallpaper3/eso0932a.jpg",
    "https://cdn.jsdelivr.net/npm/three-globe/example/img/night-sky.png",
)
EARTH_DISPLAY_WIDTH = 4096

# JPL ephemeris used for Sun geometry (sunlit checks, the day/night terminator,
# and visible-pass classification). Downloaded once, ~16 MB, then cached.
EPHEMERIS_NAME = "de421.bsp"

USER_AGENT = "iss-tracker/1.0 (+https://github.com/local/iss-tracker)"

# --- Networking timeouts -----------------------------------------------------

HTTP_TIMEOUT_SECONDS = 30

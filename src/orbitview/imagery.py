"""Download and cache the equirectangular Earth textures for the globe.

Two textures are served: the daytime Blue Marble and the night-time Black
Marble (city lights). Each is downloaded once, downscaled, and cached.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from . import config


class ImageryError(RuntimeError):
    """Raised when Earth imagery cannot be obtained."""


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def _cached_texture(name: str, urls: Sequence[str]) -> Path:
    """Return a local, downscaled JPEG for one of ``urls``, caching the result."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    source_path = config.CACHE_DIR / f"{name}_source.jpg"
    display_path = config.CACHE_DIR / f"{name}_display.jpg"

    if display_path.exists():
        return display_path

    if not source_path.exists():
        last_exc: Exception | None = None
        for url in urls:
            try:
                _download(url, source_path)
                break
            except (OSError, urllib.error.URLError) as exc:
                last_exc = exc
        else:
            raise ImageryError(
                f"Could not download {name} imagery and no cached copy is available."
            ) from last_exc

    with Image.open(source_path) as image:
        image = image.convert("RGB")
        if image.width > config.EARTH_DISPLAY_WIDTH:
            ratio = config.EARTH_DISPLAY_WIDTH / image.width
            height = max(1, int(image.height * ratio))
            image = image.resize(
                (config.EARTH_DISPLAY_WIDTH, height), Image.Resampling.LANCZOS
            )
        image.save(display_path, "JPEG", quality=88)

    return display_path


def earth_texture_path() -> Path:
    """Local path to the cached daytime Blue Marble texture."""
    return _cached_texture("earth_day", config.EARTH_IMAGE_URLS)


def earth_night_texture_path() -> Path:
    """Local path to the cached night-time Black Marble (city lights) texture."""
    return _cached_texture("earth_night", config.EARTH_NIGHT_IMAGE_URLS)


def sky_texture_path() -> Path:
    """Local path to the cached Milky Way / starfield sky panorama."""
    return _cached_texture("sky", config.SKY_IMAGE_URLS)

# OrbitView — live 3D satellite tracker

A browser-based satellite tracker with an interactive 3D globe. Watch the ISS
(or any satellite) orbit in real time over a NASA Blue Marble Earth, read live
telemetry, see the day/night terminator, and get visible-pass predictions and
alerts for your own location.

Positions come from the latest TLE (Two-Line Element) sets via SGP4 propagation
through [Skyfield](https://rhodesmill.org/skyfield/), served by a small FastAPI
backend. The front end is a single page built on
[globe.gl](https://github.com/vasturiano/globe.gl) — no build step required.

![OrbitView](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/status-1.0-brightgreen)

## Features

- **Live 3D globe with real lighting** — a custom shader lights the Earth from
  the true Sun direction: daylight on one side, **city lights** (NASA Black
  Marble) glowing on the night side, and a soft natural terminator. The Sun
  itself sits in the scene in the correct subsolar direction.
- **2D map or 3D globe** — toggle between the 3D globe and a 2D equirectangular
  map; both keep the day/night textures, the live terminator, markers and tracks.
- **Search every active satellite** — a searchable catalog of ~16,000 on-orbit
  satellites (CelesTrak), by name or NORAD id, on top of the curated quick-list.
- **Whole constellations in one click** — add Starlink, OneWeb, GPS, Galileo,
  Iridium, etc. as a live points cloud (sampled/capped for huge groups), in both
  the 3D and 2D views. **Click any member** to promote it to a fully-tracked
  satellite with its trajectory, telemetry and orbital elements.
- **Track many objects at once** — every tracked satellite gets its own colour
  and marker; the focused one draws its past (solid) / predicted (dashed) track.
  Click a chip to focus it; markers switch instantly with no glide.
- **Rich telemetry** — latitude, longitude, altitude, speed, orbital period,
  inclination, sunlit/in-shadow status, and TLE freshness for the focused object.
- **Multi-satellite catalog** — ISS, Tiangong, Hubble, NOAA-20, Terra,
  Landsat 9, GOES-18 — or track *any* NORAD catalog ID.
- **Pass predictions** — upcoming overhead passes for your location with rise
  time, peak elevation, compass heading, and a *visible to the naked eye* flag
  (satellite sunlit while you're in darkness).
- **Alerts** — opt-in banner, sound, and browser notification before a pass.

### Mission-operations layer

- **Time machine** — scrub/propagate the whole scene to any epoch; play
  forward/back at 1×–600×, jump ±1 orbit, jump to the next event, or snap back
  to live. The globe, markers, terminator, Sun and telemetry all follow.
- **Ground-station contacts** — RF contact schedule over a built-in station
  network (Svalbard, Troll, Fairbanks, Wallops, Kourou, …) or custom stations:
  AOS/LOS, duration, max elevation, AOS→LOS azimuth. Click a contact for an
  **az/el polar plot** (antenna pointing) and a **Doppler-shift** curve.
- **Power/thermal** — eclipse (umbra) entry/exit and duration per orbit,
  sunlit fraction, and **beta angle**.
- **Full orbital state** — SMA, eccentricity, inclination, RAAN, argument of
  perigee, true/mean anomaly, apogee/perigee altitude, and revolution number.
- **Target access** — drop a target (type a lat/lon or pick on the globe) and
  get access windows within a max off-nadir / elevation constraint; click a
  window to fly the time machine to it and watch the overpass.
- **Event timeline + export** — one chronological feed of contacts, eclipse,
  terminator and apsis events, exportable to **CSV** and **iCalendar (.ics)**.

## Requirements

- Python 3.11+
- Network access on first run (downloads TLEs from CelesTrak, the Blue Marble
  texture, and the `de421` ephemeris ~16 MB for Sun geometry). Everything is
  cached under `~/.cache/iss-tracker` and reused thereafter.
- A modern browser with WebGL.

## Setup

```bash
cd ~/Projects/iss-tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
iss-tracker            # serve on http://127.0.0.1:8000 and open the browser
# or
python -m iss_tracker
```

Options:

```bash
python -m iss_tracker --port 9000     # choose a port
python -m iss_tracker --no-open       # serve only, don't open a browser
python -m iss_tracker --host 0.0.0.0  # expose on the local network
python -m iss_tracker --reload        # auto-reload for development
```

## How it works

The backend (`src/iss_tracker/`) is the source of truth for all astronomy:

| Module | Responsibility |
| --- | --- |
| `tle.py` | Fetch & cache TLEs per NORAD id; fall back to a stale cache offline |
| `propagation.py` | Skyfield SGP4 — positions, ground tracks, subsolar point, passes |
| `imagery.py` | Download & downscale the equirectangular Earth texture |
| `catalog.py` | The curated default satellite list |
| `server.py` | FastAPI routes + static file serving |

The browser fetches a ground track sampled every few seconds and interpolates
between samples by wall-clock time, so the marker glides smoothly at the display
frame rate while the server stays authoritative. It refetches the track every
~45 s (which also picks up refreshed TLEs).

### API

| Endpoint | Description |
| --- | --- |
| `GET /api/catalog` | Curated default satellite list |
| `GET /api/search?q=` | Search the full active catalog (~16k satellites) by name or id |
| `GET /api/constellations` | Constellations (CelesTrak groups) available to add |
| `GET /api/constellation/{id}` | Member positions of a constellation (capped/sampled) |
| `GET /api/satellite/{norad_id}` | Metadata, current position, orbital params |
| `GET /api/track/{norad_id}` | Sampled ground track (`minutes_before`, `minutes_after`, `step_seconds`) |
| `GET /api/passes/{norad_id}?lat=&lon=` | Upcoming passes (`days`, `min_elevation_deg`) |
| `GET /api/earth-texture` | Cached Blue Marble (day) JPEG |
| `GET /api/earth-night-texture` | Cached Black Marble (night lights) JPEG |
| `GET /api/sky-texture` | Cached Milky Way panorama (sky background) |
| `GET /api/track/{id}?epoch=` | Ground track centered on an arbitrary epoch (time machine) |
| `GET /api/stations` | Built-in ground-station network |
| `GET /api/contacts/{id}` | Contact schedule (`station_id=`, `station=`, `days`) |
| `GET /api/contacts/{id}/profile` | Az/el/range/range-rate + Doppler over one contact |
| `GET /api/eclipses/{id}` | Umbra entry/exit, eclipse duration, beta angle |
| `GET /api/elements/{id}` | Full Keplerian element set |
| `GET /api/access/{id}?lat=&lon=` | Target access windows (`max_off_nadir_deg`) |
| `GET /api/events/{id}` | Merged event timeline (contacts/eclipse/terminator/apsis) |

The browser renders the globe with three.js + globe.gl loaded as ES modules
(via an import map from esm.sh); the day/night blend is a small custom GLSL
shader fed the subsolar direction the backend computes each refresh.

## Development

```bash
pip install -e ".[dev]"
pytest          # 16 tests, no network needed (TLEs are stubbed/fixed)
ruff check src tests
```

## Legacy desktop viewer

The original matplotlib/tkinter desktop version lives in
[`legacy/iss_tracker_desktop.py`](legacy/iss_tracker_desktop.py). To run it:

```bash
pip install -e ".[desktop]"
python legacy/iss_tracker_desktop.py
```

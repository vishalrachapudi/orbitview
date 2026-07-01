#!/usr/bin/env python3
"""Interactive ISS location and trajectory viewer with time scrubbing."""

from __future__ import annotations

import argparse
import calendar
import sys
import tkinter as tk
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider
from PIL import Image
from skyfield.api import EarthSatellite, Loader, wgs84
from skyfield.timelib import Time

ISS_NORAD_ID = 25544
CELESTRAK_TLE_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
)
EARTH_IMAGE_URLS = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74117/"
    "world.200408.3x5400x2700.jpg",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/"
    "land_shallow_topo_2048.jpg",
    "https://raw.githubusercontent.com/turban/webgl-earth/master/images/"
    "2_no_clouds_4k.jpg",
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "orbitview"
DEFAULT_HOURS_EACH_WAY = 6.0
MIN_HOURS_EACH_WAY = 1.0
MAX_HOURS_EACH_WAY = 168.0  # 7 days each way
DEFAULT_STEP_SECONDS = 30
MAX_TRAJECTORY_SAMPLES = 15000
ORBIT_PERIOD_MINUTES = 92.0
LIVE_UPDATE_INTERVAL_MS = 1000
TRAJECTORY_RECENTER_SECONDS = 120
TLE_REFRESH_SECONDS = 6 * 60 * 60
EARTH_DISPLAY_WIDTH = 4096
CENTER_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class IssSample:
    """A single propagated ISS state sample."""

    time: datetime
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    speed_km_s: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Show ISS location and scrub through its predicted trajectory.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS_EACH_WAY,
        help="Initial hours of trajectory to show before and after the reference time "
        f"(default: {DEFAULT_HOURS_EACH_WAY}, max: {MAX_HOURS_EACH_WAY}).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP_SECONDS,
        help=f"Seconds between trajectory samples (default: {DEFAULT_STEP_SECONDS}).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="now",
        help='Reference UTC time as "now" or an ISO timestamp like '
        '"2026-06-16T12:00:00Z".',
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable automatic live tracking; stay on the initial reference time.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached TLE and imagery files.",
    )
    return parser.parse_args()


def parse_reference_time(reference: str) -> datetime:
    """Parse the reference time argument into a timezone-aware UTC datetime."""
    if reference.lower() == "now":
        return datetime.now(timezone.utc)

    return parse_center_time(reference)


def parse_center_time(value: str) -> datetime:
    """Parse a user-entered UTC timestamp."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("Time is empty.")
    if normalized.lower() == "now":
        return datetime.now(timezone.utc)

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            'Use UTC like "2026-06-16 18:30:00" or "2026-06-16T18:30:00Z".',
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_center_time(value: datetime) -> str:
    """Format a UTC datetime for the center-time input."""
    return value.astimezone(timezone.utc).strftime(CENTER_TIME_FORMAT)


def download_file(url: str, destination: Path) -> None:
    """Download a remote file to a local path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "orbitview/0.2 (+local earth imagery cache)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def load_earth_texture(cache_dir: Path) -> np.ndarray:
    """Load a high-resolution equirectangular Earth image, cached locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = cache_dir / "earth_blue_marble.jpg"

    if not image_path.exists():
        last_exc: Exception | None = None
        for url in EARTH_IMAGE_URLS:
            try:
                download_file(url, image_path)
                break
            except (OSError, urllib.error.URLError) as exc:
                last_exc = exc
        else:
            raise RuntimeError(
                "Could not download Earth imagery and no cached copy is available.",
            ) from last_exc

    with Image.open(image_path) as image:
        if image.width > EARTH_DISPLAY_WIDTH:
            ratio = EARTH_DISPLAY_WIDTH / image.width
            display_height = max(1, int(image.height * ratio))
            image = image.resize(
                (EARTH_DISPLAY_WIDTH, display_height),
                Image.Resampling.LANCZOS,
            )
        return np.asarray(image)


def load_iss_satellite(cache_dir: Path, force_refresh: bool = False):
    """Load the latest ISS TLE and return a Skyfield EarthSatellite."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    loader = Loader(str(cache_dir))
    tle_path = cache_dir / "iss.tle"

    if force_refresh or not tle_path.exists():
        try:
            tle_bytes = loader.open(CELESTRAK_TLE_URL).read()
            tle_path.write_bytes(tle_bytes)
        except OSError as exc:
            if not tle_path.exists():
                raise RuntimeError(
                    "Could not download ISS TLE data and no cached copy is available.",
                ) from exc

    tle_text = tle_path.read_text(encoding="utf-8").strip().splitlines()
    if len(tle_text) < 3:
        raise RuntimeError("ISS TLE file is incomplete.")

    name, line1, line2 = tle_text[0], tle_text[1], tle_text[2]
    timescale = loader.timescale()
    return EarthSatellite(line1, line2, name.strip(), timescale), timescale


def propagate_iss(
    satellite,
    timescale,
    start: datetime,
    end: datetime,
    step_seconds: int,
) -> list[IssSample]:
    """Propagate ISS states from start through end inclusive."""
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive.")

    step = timedelta(seconds=step_seconds)
    samples: list[IssSample] = []
    current = start
    while current <= end:
        skyfield_time: Time = timescale.from_datetime(current)
        geocentric = satellite.at(skyfield_time)
        subpoint = wgs84.subpoint(geocentric)
        samples.append(
            IssSample(
                time=current,
                latitude_deg=float(subpoint.latitude.degrees),
                longitude_deg=float(subpoint.longitude.degrees),
                altitude_km=float(subpoint.elevation.km),
                speed_km_s=float(np.linalg.norm(geocentric.velocity.km_per_s)),
            ),
        )
        current += step

    return samples


def split_longitude_segments(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a ground track at the antimeridian so lines do not wrap the map."""
    if len(longitudes) == 0:
        return []

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    segment_lons: list[float] = [float(longitudes[0])]
    segment_lats: list[float] = [float(latitudes[0])]

    for index in range(1, len(longitudes)):
        previous_lon = float(longitudes[index - 1])
        current_lon = float(longitudes[index])
        if abs(current_lon - previous_lon) > 180.0:
            segments.append((np.array(segment_lons), np.array(segment_lats)))
            segment_lons = [current_lon]
            segment_lats = [float(latitudes[index])]
        else:
            segment_lons.append(current_lon)
            segment_lats.append(float(latitudes[index]))

    segments.append((np.array(segment_lons), np.array(segment_lats)))
    return segments


def build_colored_segments(
    samples: list[IssSample],
    reference_index: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Build past and future ground-track segments relative to the reference index."""
    longitudes = np.array([sample.longitude_deg for sample in samples])
    latitudes = np.array([sample.latitude_deg for sample in samples])

    past_segments = split_longitude_segments(
        longitudes[: reference_index + 1],
        latitudes[: reference_index + 1],
    )
    future_segments = split_longitude_segments(
        longitudes[reference_index:],
        latitudes[reference_index:],
    )
    return past_segments, future_segments


def closest_sample_index(samples: list[IssSample], target: datetime) -> int:
    """Return the sample index closest to the target UTC time."""
    return min(
        range(len(samples)),
        key=lambda index: abs((samples[index].time - target).total_seconds()),
    )


def _format_offset(offset: timedelta) -> str:
    """Format a timedelta relative to the reference time."""
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_window_label(hours_each_way: float) -> str:
    """Format the ± window size for display."""
    if hours_each_way >= 24:
        days = hours_each_way / 24
        if abs(days - round(days)) < 0.05:
            return f"±{round(days):g} d"
        return f"±{days:.1f} d"
    if abs(hours_each_way - round(hours_each_way)) < 0.05:
        return f"±{round(hours_each_way):g} h"
    return f"±{hours_each_way:.1f} h"


def effective_step_seconds(hours_each_way: float, requested_step: int) -> int:
    """Pick a step size that keeps long windows responsive."""
    span_seconds = hours_each_way * 2 * 3600
    if span_seconds <= 0:
        return requested_step

    min_step_for_cap = int(np.ceil(span_seconds / MAX_TRAJECTORY_SAMPLES))
    return max(requested_step, min_step_for_cap)


def clamp_hours_each_way(hours: float) -> float:
    """Clamp the visible window to supported bounds."""
    return float(np.clip(hours, MIN_HOURS_EACH_WAY, MAX_HOURS_EACH_WAY))


class CenterTimeDropdownPanel:
    """UTC date/time picker built from dropdown menus."""

    def __init__(
        self,
        parent: tk.Misc,
        initial: datetime,
        on_center,
    ) -> None:
        self.on_center = on_center
        self.syncing = False

        self.frame = tk.Frame(parent, bg="#0a1020", padx=10, pady=8)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)

        title = tk.Label(
            self.frame,
            text="Center at UTC",
            bg="#0a1020",
            fg="#b8c7de",
            font=("", 11, "bold"),
        )
        title.pack(side=tk.LEFT, padx=(0, 12))

        fields = tk.Frame(self.frame, bg="#0a1020")
        fields.pack(side=tk.LEFT, fill=tk.X, expand=True)

        now = datetime.now(timezone.utc)
        self.year_var = tk.StringVar()
        self.month_var = tk.StringVar()
        self.day_var = tk.StringVar()
        self.hour_var = tk.StringVar()
        self.minute_var = tk.StringVar()
        self.second_var = tk.StringVar()

        year_values = [str(year) for year in range(now.year - 5, now.year + 6)]
        month_values = [f"{month:02d}" for month in range(1, 13)]
        hour_values = [f"{hour:02d}" for hour in range(24)]
        minute_values = [f"{minute:02d}" for minute in range(60)]
        second_values = [f"{second:02d}" for second in range(60)]

        self.year_combo = self._add_dropdown(fields, "Year", self.year_var, year_values, 6)
        self._add_separator(fields, "-")
        self.month_combo = self._add_dropdown(fields, "Month", self.month_var, month_values, 4)
        self._add_separator(fields, "-")
        self.day_combo = self._add_dropdown(fields, "Day", self.day_var, [], 4)
        self._add_separator(fields, " ")
        self.hour_combo = self._add_dropdown(fields, "Hour", self.hour_var, hour_values, 4)
        self._add_separator(fields, ":")
        self.minute_combo = self._add_dropdown(fields, "Min", self.minute_var, minute_values, 4)
        self._add_separator(fields, ":")
        self.second_combo = self._add_dropdown(fields, "Sec", self.second_var, second_values, 4)

        self.center_button = tk.Button(
            self.frame,
            text="Center",
            command=self.submit,
            padx=12,
            pady=4,
        )
        self.center_button.pack(side=tk.LEFT, padx=(12, 8))

        self.error_label = tk.Label(
            self.frame,
            text="",
            bg="#0a1020",
            fg="#ff8a8a",
            font=("", 10),
        )
        self.error_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.year_combo.bind("<<ComboboxSelected>>", self._on_date_part_changed)
        self.month_combo.bind("<<ComboboxSelected>>", self._on_date_part_changed)

        self.set_time(initial)

    def _add_dropdown(
        self,
        parent: tk.Frame,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        width: int,
    ) -> ttk.Combobox:
        """Add a labeled read-only combobox."""
        container = tk.Frame(parent, bg="#0a1020")
        container.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(
            container,
            text=label,
            bg="#0a1020",
            fg="#8ea6c8",
            font=("", 9),
        ).pack(anchor="w")
        combo = ttk.Combobox(
            container,
            textvariable=variable,
            values=values,
            width=width,
            state="readonly",
        )
        combo.pack()
        return combo

    @staticmethod
    def _add_separator(parent: tk.Frame, text: str) -> None:
        """Add a small separator label between dropdown groups."""
        tk.Label(
            parent,
            text=text,
            bg="#0a1020",
            fg="#8ea6c8",
            font=("", 12),
        ).pack(side=tk.LEFT, padx=(0, 4), pady=(12, 0))

    def _on_date_part_changed(self, _event=None) -> None:
        """Keep the day dropdown valid for the selected month."""
        if self.syncing:
            return
        self._refresh_day_values()

    def _refresh_day_values(self) -> None:
        """Update day choices for the selected year and month."""
        year = int(self.year_var.get())
        month = int(self.month_var.get())
        days_in_month = calendar.monthrange(year, month)[1]
        day_values = [f"{day:02d}" for day in range(1, days_in_month + 1)]
        self.day_combo["values"] = day_values

        current_day = int(self.day_var.get() or "1")
        if current_day > days_in_month:
            self.day_var.set(f"{days_in_month:02d}")
        elif self.day_var.get() not in day_values:
            self.day_var.set(f"{min(current_day, days_in_month):02d}")

    def set_time(self, value: datetime) -> None:
        """Set all dropdowns to a UTC datetime."""
        self.syncing = True
        utc_time = value.astimezone(timezone.utc)
        self.year_var.set(str(utc_time.year))
        self.month_var.set(f"{utc_time.month:02d}")
        self._refresh_day_values()
        self.day_var.set(f"{utc_time.day:02d}")
        self.hour_var.set(f"{utc_time.hour:02d}")
        self.minute_var.set(f"{utc_time.minute:02d}")
        self.second_var.set(f"{utc_time.second:02d}")
        self.syncing = False

    def get_time(self) -> datetime:
        """Read the selected UTC datetime from the dropdowns."""
        return datetime(
            int(self.year_var.get()),
            int(self.month_var.get()),
            int(self.day_var.get()),
            int(self.hour_var.get()),
            int(self.minute_var.get()),
            int(self.second_var.get()),
            tzinfo=timezone.utc,
        )

    def set_error(self, message: str) -> None:
        """Show a validation error."""
        self.error_label.config(text=message)

    def clear_error(self) -> None:
        """Clear the validation error."""
        self.error_label.config(text="")

    def submit(self) -> None:
        """Validate the dropdown selection and center the view."""
        try:
            target_time = self.get_time()
        except ValueError as exc:
            self.set_error(f"Invalid date/time: [{exc}].")
            return

        self.clear_error()
        self.on_center(target_time)


class IssTrackerApp:
    """Matplotlib application for live and scrubbed ISS trajectory viewing."""

    def __init__(
        self,
        satellite,
        timescale,
        cache_dir: Path,
        samples: list[IssSample],
        reference_index: int,
        hours_each_way: float,
        step_seconds: int,
        live_mode: bool,
    ) -> None:
        self.satellite = satellite
        self.timescale = timescale
        self.cache_dir = cache_dir
        self.hours_each_way = clamp_hours_each_way(hours_each_way)
        self.base_step_seconds = step_seconds
        self.step_seconds = effective_step_seconds(self.hours_each_way, step_seconds)
        self.samples = samples
        self.reference_index = reference_index
        self.window_center = samples[reference_index].time
        self.current_index = reference_index
        self.live_mode = live_mode
        self.playing = False
        self.play_timer = None
        self.live_timer = None
        self.last_tle_refresh = datetime.now(timezone.utc)
        self.syncing_center_time_box = False
        self.earth_image = load_earth_texture(cache_dir)

        self.figure = plt.figure(figsize=(14, 8), facecolor="#0a1020")
        self.figure.canvas.manager.set_window_title("ISS Trajectory Viewer")
        self._build_layout()

        self.past_lines: list = []
        self.future_lines: list = []
        self.current_marker = self.map_axes.plot(
            [],
            [],
            "o",
            color="#ff3355",
            markersize=11,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=5,
        )[0]
        self.current_halo = self.map_axes.plot(
            [],
            [],
            "o",
            color="#ff3355",
            markersize=22,
            alpha=0.35,
            zorder=4,
        )[0]

        self.time_slider.on_changed(self.on_slider_changed)
        self.span_slider.on_changed(self.on_span_changed)
        self.play_button.on_clicked(self.toggle_play)
        self.live_button.on_clicked(self.toggle_live)
        self.orbit_back_button.on_clicked(lambda _event: self.jump_orbits(-1))
        self.orbit_forward_button.on_clicked(lambda _event: self.jump_orbits(1))

        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.figure.canvas.mpl_connect("resize_event", self.on_resize)

        self.center_panel = CenterTimeDropdownPanel(
            parent=self.figure.canvas.manager.window,
            initial=self.window_center,
            on_center=self.center_on_time,
        )

        self._refresh_header_text()
        self.update_view(self.current_index, draw_slider=False)
        if self.live_mode:
            self.start_live_updates()

    def _build_layout(self) -> None:
        """Create a responsive layout that scales with the window."""
        grid = GridSpec(
            nrows=26,
            ncols=24,
            figure=self.figure,
            left=0.04,
            right=0.98,
            top=0.95,
            bottom=0.10,
            hspace=0.55,
            wspace=0.2,
        )

        self.map_axes = self.figure.add_subplot(grid[0:16, 0:24])
        self._draw_world_map(self.map_axes)

        self.header_text = self.figure.text(
            0.04,
            0.975,
            "",
            ha="left",
            va="top",
            fontsize=10,
            color="#b8c7de",
        )
        self.info_text = self.figure.text(
            0.04,
            0.935,
            "",
            ha="left",
            va="top",
            fontsize=11,
            family="monospace",
            color="#f2f6ff",
        )

        slider_ax = self.figure.add_subplot(grid[17:18, 1:23])
        self.time_slider = Slider(
            slider_ax,
            "Time",
            0,
            max(1, len(self.samples) - 1),
            valinit=self.current_index,
            valstep=1,
            color="#4da3ff",
        )

        span_ax = self.figure.add_subplot(grid[19:20, 1:23])
        self.span_slider = Slider(
            span_ax,
            "Window",
            MIN_HOURS_EACH_WAY,
            MAX_HOURS_EACH_WAY,
            valinit=self.hours_each_way,
            valstep=1,
            color="#66d9a8",
        )
        self._update_span_slider_label(self.hours_each_way)

        button_row = grid[21:23, 1:23].subgridspec(1, 5, wspace=0.25)
        self.live_button = Button(
            self.figure.add_subplot(button_row[0, 0]),
            "Live" if not self.live_mode else "Live ●",
        )
        self.play_button = Button(self.figure.add_subplot(button_row[0, 1]), "Play")
        self.orbit_back_button = Button(
            self.figure.add_subplot(button_row[0, 2]),
            "−1 orbit",
        )
        self.orbit_forward_button = Button(
            self.figure.add_subplot(button_row[0, 3]),
            "+1 orbit",
        )
        self.now_button = Button(self.figure.add_subplot(button_row[0, 4]), "Now")

        self.now_button.on_clicked(self.jump_to_now)

    def _draw_world_map(self, axes: plt.Axes) -> None:
        """Draw the high-resolution Earth background."""
        axes.imshow(
            self.earth_image,
            extent=(-180, 180, -90, 90),
            origin="upper",
            interpolation="bilinear",
            zorder=0,
        )
        axes.set_xlim(-180, 180)
        axes.set_ylim(-90, 90)
        axes.set_aspect("equal", adjustable="box")
        axes.set_facecolor("#020611")
        axes.set_xlabel("Longitude (°)", color="#b8c7de", labelpad=8)
        axes.set_ylabel("Latitude (°)", color="#b8c7de", labelpad=8)
        axes.tick_params(colors="#b8c7de", labelsize=9)
        for spine in axes.spines.values():
            spine.set_color("#30415f")
        axes.grid(color="#ffffff", linestyle=":", linewidth=0.5, alpha=0.25)

    def _update_span_slider_label(self, hours_each_way: float) -> None:
        """Show the window slider value in hours or days."""
        self.span_slider.label.set_text(f"Window ({_format_window_label(hours_each_way)})")

    def _refresh_header_text(self) -> None:
        """Update the status line above the map."""
        mode = "LIVE" if self.live_mode else "SCRUB"
        step_note = ""
        if self.step_seconds > self.base_step_seconds:
            step_note = f"  |  Sample step: {self.step_seconds}s"
        self.header_text.set_text(
            f"Mode: {mode}  |  Window: {_format_window_label(self.hours_each_way)}  |  "
            f"Range: {self.samples[0].time.strftime('%Y-%m-%d %H:%M')} – "
            f"{self.samples[-1].time.strftime('%Y-%m-%d %H:%M')} UTC{step_note}  |  "
            "Use the UTC dropdowns below to center on a specific time",
        )

    def on_resize(self, _event) -> None:
        """Keep the map aspect ratio stable when the window is resized."""
        self.map_axes.set_aspect("equal", adjustable="box")
        self.figure.canvas.draw_idle()

    def on_slider_changed(self, value: float) -> None:
        """Handle manual slider movement."""
        self.stop_play()
        if self.live_mode:
            self.set_live_mode(False)
        self.update_view(int(value), draw_slider=False)

    def on_span_changed(self, value: float) -> None:
        """Rebuild the trajectory for a new visible time window."""
        hours_each_way = clamp_hours_each_way(float(value))
        if abs(hours_each_way - self.hours_each_way) < 0.5:
            return

        self.hours_each_way = hours_each_way
        self.step_seconds = effective_step_seconds(
            self.hours_each_way,
            self.base_step_seconds,
        )
        self._update_span_slider_label(self.hours_each_way)

        center_time = self.samples[self.current_index].time
        self.rebuild_trajectory(center_time, preserve_time=center_time)
        self._refresh_header_text()
        self.figure.canvas.draw_idle()

    def rebuild_trajectory(
        self,
        center_time: datetime,
        preserve_time: datetime | None = None,
    ) -> None:
        """Recompute samples for the current window size."""
        start = center_time - timedelta(hours=self.hours_each_way)
        end = center_time + timedelta(hours=self.hours_each_way)
        self.samples = propagate_iss(
            self.satellite,
            self.timescale,
            start,
            end,
            self.step_seconds,
        )
        self.window_center = center_time
        self.reference_index = closest_sample_index(self.samples, center_time)

        target_time = preserve_time if preserve_time is not None else center_time
        target_index = closest_sample_index(self.samples, target_time)

        self.time_slider.valmin = 0
        self.time_slider.valmax = max(1, len(self.samples) - 1)
        self.time_slider.ax.set_xlim(self.time_slider.valmin, self.time_slider.valmax)
        self.update_view(target_index, draw_slider=True)

    def center_from_dropdowns(self) -> None:
        """Center the view using the current dropdown selection."""
        self.center_panel.submit()

    def center_on_time(self, target_time: datetime) -> None:
        """Rebuild the trajectory window centered on a specific UTC time."""
        self.stop_play()
        if self.live_mode:
            self.set_live_mode(False)

        self.center_panel.clear_error()
        self.rebuild_trajectory(target_time, preserve_time=target_time)
        self._set_center_time_selector(target_time)
        self._refresh_header_text()
        self.figure.canvas.draw_idle()

    def _set_center_time_selector(self, target_time: datetime) -> None:
        """Update the dropdown picker to match the selected time."""
        if self.syncing_center_time_box:
            return
        self.syncing_center_time_box = True
        self.center_panel.set_time(target_time)
        self.syncing_center_time_box = False

    def on_key_press(self, event) -> None:
        """Handle keyboard scrubbing."""
        if event.key in {"left", "down"}:
            self.stop_play()
            if self.live_mode:
                self.set_live_mode(False)
            self.update_view(max(0, self.current_index - 1))
        elif event.key in {"right", "up"}:
            self.stop_play()
            if self.live_mode:
                self.set_live_mode(False)
            self.update_view(min(len(self.samples) - 1, self.current_index + 1))
        elif event.key == " ":
            self.toggle_play(None)
        elif event.key in {"home", "n"}:
            self.jump_to_now(None)
        elif event.key == "[":
            self.adjust_window_hours(-6)
        elif event.key == "]":
            self.adjust_window_hours(6)
        elif event.key == "{":
            self.adjust_window_hours(-24)
        elif event.key == "}":
            self.adjust_window_hours(24)
        elif event.key == "c":
            self.center_from_dropdowns()

    def adjust_window_hours(self, delta_hours: float) -> None:
        """Shift the visible window size by a fixed number of hours."""
        new_hours = clamp_hours_each_way(self.hours_each_way + delta_hours)
        if abs(new_hours - self.hours_each_way) < 0.5:
            return

        self.span_slider.set_val(new_hours)

    def toggle_play(self, _event) -> None:
        """Start or stop automatic time playback."""
        if self.playing:
            self.stop_play()
        else:
            if self.live_mode:
                self.set_live_mode(False)
            self.start_play()

    def toggle_live(self, _event) -> None:
        """Toggle live tracking mode."""
        self.set_live_mode(not self.live_mode)

    def set_live_mode(self, enabled: bool) -> None:
        """Enable or disable live tracking."""
        self.live_mode = enabled
        self.live_button.label.set_text("Live ●" if enabled else "Live")
        self._refresh_header_text()
        if enabled:
            self.stop_play()
            self.start_live_updates()
            self.refresh_trajectory(datetime.now(timezone.utc), force_recenter=True)
        else:
            self.stop_live_updates()
        self.figure.canvas.draw_idle()

    def jump_to_now(self, _event) -> None:
        """Jump to the current UTC time and optionally re-enable live mode."""
        self.stop_play()
        now = datetime.now(timezone.utc)
        self.refresh_trajectory(now, force_recenter=True)
        self.set_live_mode(True)

    def start_play(self) -> None:
        """Advance time automatically through the loaded samples."""
        self.playing = True
        self.play_button.label.set_text("Pause")
        self.figure.canvas.draw_idle()

        def advance(_timer) -> None:
            if not self.playing:
                return
            next_index = self.current_index + 1
            if next_index >= len(self.samples):
                self.stop_play()
                return
            self.update_view(next_index)

        interval_ms = max(20, self.step_seconds * 25)
        self.play_timer = self.figure.canvas.new_timer(interval=interval_ms)
        self.play_timer.add_callback(advance)
        self.play_timer.start()

    def stop_play(self) -> None:
        """Stop automatic playback."""
        self.playing = False
        self.play_button.label.set_text("Play")
        if self.play_timer is not None:
            self.play_timer.stop()
            self.play_timer = None
        self.figure.canvas.draw_idle()

    def start_live_updates(self) -> None:
        """Continuously advance the view with real time."""
        self.stop_live_updates()

        def tick(_timer) -> None:
            if not self.live_mode:
                return
            now = datetime.now(timezone.utc)
            self.refresh_trajectory(now)
            index = closest_sample_index(self.samples, now)
            self.update_view(index, draw_slider=True)

        self.live_timer = self.figure.canvas.new_timer(interval=LIVE_UPDATE_INTERVAL_MS)
        self.live_timer.add_callback(tick)
        self.live_timer.start()

    def stop_live_updates(self) -> None:
        """Stop the live update timer."""
        if self.live_timer is not None:
            self.live_timer.stop()
            self.live_timer = None

    def refresh_trajectory(self, center_time: datetime, force_recenter: bool = False) -> None:
        """Refresh TLE data and rebuild the trajectory window when needed."""
        now = datetime.now(timezone.utc)
        if (now - self.last_tle_refresh).total_seconds() >= TLE_REFRESH_SECONDS:
            try:
                self.satellite, self.timescale = load_iss_satellite(
                    self.cache_dir,
                    force_refresh=True,
                )
                self.last_tle_refresh = now
            except RuntimeError:
                pass

        elapsed = abs((center_time - self.window_center).total_seconds())
        if not force_recenter and elapsed < TRAJECTORY_RECENTER_SECONDS:
            return

        self.rebuild_trajectory(center_time, preserve_time=center_time)
        self._refresh_header_text()

    def jump_orbits(self, direction: int) -> None:
        """Jump forward or backward by roughly one ISS orbit."""
        if self.live_mode:
            self.set_live_mode(False)
        orbit_steps = int((ORBIT_PERIOD_MINUTES * 60) / self.step_seconds)
        target = self.current_index + direction * orbit_steps
        target = max(0, min(len(self.samples) - 1, target))
        self.stop_play()
        self.update_view(target)

    def update_view(self, index: int, draw_slider: bool = True) -> None:
        """Refresh map markers and info for the selected time index."""
        if not self.samples:
            return

        index = max(0, min(len(self.samples) - 1, index))
        self.current_index = index
        sample = self.samples[index]

        for line in self.past_lines + self.future_lines:
            line.remove()
        self.past_lines.clear()
        self.future_lines.clear()

        past_segments, future_segments = build_colored_segments(self.samples, index)
        line_width = max(1.8, min(3.2, self.figure.get_figwidth() * 0.22))

        for segment_lons, segment_lats in past_segments:
            line = self.map_axes.plot(
                segment_lons,
                segment_lats,
                color="#7fd4ff",
                linewidth=line_width,
                alpha=0.92,
                solid_capstyle="round",
                zorder=2,
            )[0]
            self.past_lines.append(line)

        for segment_lons, segment_lats in future_segments:
            line = self.map_axes.plot(
                segment_lons,
                segment_lats,
                color="#ffd166",
                linewidth=line_width,
                alpha=0.92,
                solid_capstyle="round",
                zorder=2,
            )[0]
            self.future_lines.append(line)

        self.current_marker.set_data([sample.longitude_deg], [sample.latitude_deg])
        self.current_halo.set_data([sample.longitude_deg], [sample.latitude_deg])

        offset = sample.time - self.samples[self.reference_index].time
        offset_label = _format_offset(offset)
        live_suffix = "   LIVE" if self.live_mode else ""
        self.info_text.set_text(
            f"UTC: {sample.time.strftime('%Y-%m-%d %H:%M:%S')}{live_suffix}   "
            f"Offset: {offset_label}\n"
            f"Lat: {sample.latitude_deg:+.4f}°   Lon: {sample.longitude_deg:+.4f}°   "
            f"Alt: {sample.altitude_km:.1f} km   Speed: {sample.speed_km_s:.2f} km/s",
        )

        if draw_slider:
            self.time_slider.set_val(index)

        if not self.syncing_center_time_box:
            self._set_center_time_selector(sample.time)

        self.figure.canvas.draw_idle()

    def run(self) -> None:
        """Show the interactive viewer."""
        plt.show()
        self.stop_live_updates()
        self.stop_play()


def main() -> int:
    """Entry point."""
    args = parse_args()
    if args.hours <= 0:
        print("Error: --hours must be positive.", file=sys.stderr)
        return 1
    if args.hours > MAX_HOURS_EACH_WAY:
        print(
            f"Error: --hours cannot exceed {MAX_HOURS_EACH_WAY} "
            f"({MAX_HOURS_EACH_WAY / 24:g} days each way).",
            file=sys.stderr,
        )
        return 1

    hours_each_way = clamp_hours_each_way(args.hours)
    step_seconds = effective_step_seconds(hours_each_way, args.step)

    try:
        reference_time = parse_reference_time(args.reference)
    except ValueError as exc:
        print(f"Error: invalid --reference time: {exc}", file=sys.stderr)
        return 1

    try:
        load_earth_texture(args.cache_dir)
        satellite, timescale = load_iss_satellite(args.cache_dir)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    start = reference_time - timedelta(hours=hours_each_way)
    end = reference_time + timedelta(hours=hours_each_way)
    samples = propagate_iss(
        satellite,
        timescale,
        start,
        end,
        step_seconds,
    )
    if not samples:
        print("Error: no trajectory samples were generated.", file=sys.stderr)
        return 1

    reference_index = closest_sample_index(samples, reference_time)
    live_mode = not args.no_live and args.reference.lower() == "now"

    app = IssTrackerApp(
        satellite=satellite,
        timescale=timescale,
        cache_dir=args.cache_dir,
        samples=samples,
        reference_index=reference_index,
        hours_each_way=hours_each_way,
        step_seconds=args.step,
        live_mode=live_mode,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

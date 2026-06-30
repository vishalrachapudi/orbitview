import Globe from "globe.gl";
import * as THREE from "three";

/* ============================================================================
 * ISS Tracker — front-end controller
 *
 * The FastAPI backend (Skyfield) is the source of truth: it serves each
 * satellite's ground track sampled every few seconds, plus orbital params, the
 * subsolar point and pass predictions. This client interpolates between samples
 * by wall clock to animate markers smoothly, lights the globe with a custom
 * day/night shader, and can track many objects at once.
 * ========================================================================== */

const EARTH_RADIUS_KM = 6371.0;
const TRACK_REFRESH_MS = 45_000;
const INFO_REFRESH_MS = 60_000;
const PASS_REFRESH_MS = 5 * 60_000;
const ALERT_LEAD_MIN = 10;
const MARKER_FPS = 30;
const SCRUB_BACK_HOURS = 6;    // transport scrubber reaches this far into the past
const SCRUB_FWD_HOURS = 24;    // ...and this far into the future

// Scene scale: globe.gl uses a globe radius of 100 units.
const SUN_DISTANCE = 2300;
const SUN_RADIUS = 120;
const SUN_GLOW = 820;
const SKY_RADIUS = 9000;   // Milky Way sphere, well beyond the Sun, inside camera far

const PALETTE = ["#38e8ff", "#ffce5a", "#54e6a0", "#c792ff", "#ff8d6b", "#6ab7ff", "#ff5d97", "#9be15d"];
const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];

const state = {
  tracked: new Map(),   // norad_id -> Sat
  order: [],            // norad_ids in insertion order
  focusId: null,
  clockOffset: 0,       // server_epoch - client_epoch (seconds)
  subsolar: null,       // shared { lat, lng }
  subsolarEpoch: null,  // epoch the subsolar point was computed at
  location: null,
  passes: [],
  alertsOn: false,
  alertedRise: null,
  events: [],           // unified event feed (populated in the timeline phase)
  eclipse: null,        // /api/eclipses payload for the focused satellite
  stations: { list: [], selected: [] },  // ground-station network + selection
  contacts: [],         // /api/contacts for the focused satellite
  target: null,         // { lat, lon, windows, marker, ring } observation target
  view: "3d",           // "3d" globe | "2d" map
  constellations: new Map(),  // id -> { id, name, color, total, shown, members, points3d }
  constLayerDirty: false,     // 2D constellation layer needs a redraw
  timezone: "local",    // "local" | "utc"
  time: {
    mode: "live",       // 'live' | 'scrub'
    scrubEpoch: null,   // absolute epoch the scrubber points at
    rate: 1,            // playback multiplier
    playing: false,
    anchorWall: null,   // wall clock captured when playback (re)started
    anchorScrub: null,  // scrubEpoch at that instant
    lastRecenterEpoch: null,
    recentering: false,
  },
};

const $ = (id) => document.getElementById(id);

// Real wall clock, server-corrected. Drives alerts, countdowns and live mode.
const nowEpoch = () => Date.now() / 1000 + state.clockOffset;

// The time the whole scene is rendered at — "now" in live mode, or the
// scrubbed/playing epoch in the time machine. Everything visual keys off this.
function displayEpoch() {
  const t = state.time;
  if (t.mode === "live") return nowEpoch();
  if (t.playing) return t.anchorScrub + (Date.now() / 1000 - t.anchorWall) * t.rate;
  return t.scrubEpoch;
}

/* --- tiny helpers --------------------------------------------------------- */
async function api(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function toast(msg, ms = 3200) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), ms);
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function formatTime(date, opts = {}) {
  const options = { weekday: "short", hour: "2-digit", minute: "2-digit", ...opts };
  if (state.timezone === "utc") {
    return new Intl.DateTimeFormat([], { ...options, timeZone: "UTC" }).format(date);
  }
  return date.toLocaleString([], options);
}
const rgba = (rgb, a) => `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;

function nextColor() {
  const used = new Set([...state.tracked.values()].map((s) => s.color));
  return PALETTE.find((c) => !used.has(c)) || PALETTE[state.order.length % PALETTE.length];
}

/* ============================================================================
 * Globe, lighting and the Sun
 * ========================================================================== */
let world;

const DAY_NIGHT_VERT = `
  varying vec3 vNormal;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    // World-space normal: the globe mesh carries an internal rotation, so we
    // must match the frame of the (world-space) sun direction or the lit
    // hemisphere ends up rotated.
    vNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const DAY_NIGHT_FRAG = `
  uniform sampler2D dayTexture;
  uniform sampler2D nightTexture;
  uniform vec3 sunDirection;
  varying vec3 vNormal;
  varying vec2 vUv;
  void main() {
    float intensity = dot(normalize(vNormal), normalize(sunDirection));
    vec3 day = texture2D(dayTexture, vUv).rgb;
    vec3 night = texture2D(nightTexture, vUv).rgb * 1.4;   // lift the city lights
    float t = smoothstep(-0.12, 0.22, intensity);          // soft terminator
    vec3 color = mix(night, day, t);
    gl_FragColor = vec4(color, 1.0);
  }
`;

function buildGlobeMaterial() {
  const loader = new THREE.TextureLoader();
  const material = new THREE.ShaderMaterial({
    uniforms: {
      dayTexture: { value: loader.load("/api/earth-texture") },
      nightTexture: { value: loader.load("/api/earth-night-texture") },
      sunDirection: { value: new THREE.Vector3(1, 0, 0) },
    },
    vertexShader: DAY_NIGHT_VERT,
    fragmentShader: DAY_NIGHT_FRAG,
  });
  return material;
}

function radialGlowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0.0, "rgba(255,248,220,1)");
  g.addColorStop(0.22, "rgba(255,216,110,0.9)");
  g.addColorStop(0.5, "rgba(255,170,60,0.28)");
  g.addColorStop(1.0, "rgba(255,150,40,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

// A large inward-facing sphere carrying the Milky Way panorama — our own
// skybox (globe.gl's backgroundImageUrl doesn't render reliably here).
function buildSky() {
  const tex = new THREE.TextureLoader().load("/api/sky-texture");
  tex.colorSpace = THREE.SRGBColorSpace;
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(SKY_RADIUS, 64, 40),
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, depthWrite: false, fog: false }),
  );
  sky.renderOrder = -1;   // draw behind everything
  world.scene().add(sky);
  return sky;
}

function buildSun() {
  const group = new THREE.Group();
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(SUN_RADIUS, 36, 24),
    new THREE.MeshBasicMaterial({ color: 0xfff3c4 }),
  );
  group.add(core);
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: radialGlowTexture(),
    color: 0xffd56b,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  }));
  glow.scale.set(SUN_GLOW, SUN_GLOW, 1);
  group.add(glow);
  world.scene().add(group);
  return group;
}

function sunVector(sub) {
  if (world.getCoords) {
    const c = world.getCoords(sub.lat, sub.lng, 0);
    return new THREE.Vector3(c.x, c.y, c.z).normalize();
  }
  const phi = ((90 - sub.lat) * Math.PI) / 180;
  const theta = ((90 - sub.lng) * Math.PI) / 180;
  return new THREE.Vector3(
    Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta),
  ).normalize();
}

// Subsolar point advanced to the current display time. The point is computed
// server-side at state.subsolarEpoch; we glide its longitude (~15°/hr) between
// refetches so the terminator follows the clock (and the time machine).
function currentSubsolar() {
  if (!state.subsolar) return null;
  if (state.subsolarEpoch == null) return state.subsolar;
  const dLng = (displayEpoch() - state.subsolarEpoch) * 360 / 86400;
  return { lat: state.subsolar.lat, lng: ((state.subsolar.lng - dLng + 540) % 360) - 180 };
}

function updateSun() {
  if (!world._mat) return;
  const sub = currentSubsolar();
  if (!sub) return;
  const v = sunVector(sub);
  world._mat.uniforms.sunDirection.value.copy(v);
  world._sun.position.copy(v).multiplyScalar(SUN_DISTANCE);
}

function initGlobe() {
  world = Globe()(document.getElementById("globe"))
    .width(window.innerWidth)
    .height(window.innerHeight)
    .backgroundColor("#04060f")
    .showAtmosphere(true)
    .atmosphereColor("#5fa8ff")
    .atmosphereAltitude(0.18)
    .pathPoints((d) => d.pts)
    .pathPointLat((p) => p[0])
    .pathPointLng((p) => p[1])
    .pathPointAlt((p) => p[2])
    .pathColor((d) => d.color)
    .pathStroke((d) => d.stroke)
    .pathDashLength((d) => (d.dash ? 0.5 : 0))
    .pathDashGap((d) => (d.dash ? 0.25 : 0))
    .pathDashAnimateTime(0)   // static dashes — calmer than flowing
    .pathTransitionDuration(0)
    .ringLat((d) => d.lat)
    .ringLng((d) => d.lng)
    .ringColor((d) => (t) => rgba(d.rgb, 1 - t))
    .ringMaxRadius(3.5)
    .ringPropagationSpeed(1.8)
    .ringRepeatPeriod(1100)
    .htmlLat((d) => d.lat)
    .htmlLng((d) => d.lng)
    .htmlAltitude((d) => d.alt)
    .htmlTransitionDuration(0)   // markers jump instantly, no glide
    .htmlElement((d) => {
      if (!d.__el) d.__el = makeMarkerEl(d);
      return d.__el;
    });

  world._mat = buildGlobeMaterial();
  world.globeMaterial(world._mat);
  world._sky = buildSky();
  world._sun = buildSun();

  const cam = world.camera();
  cam.far = 30000;
  cam.updateProjectionMatrix();

  world.pointOfView({ lat: 20, lng: 0, altitude: 2.6 });
  const controls = world.controls();
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.18;
  controls.enableDamping = true;
  controls.addEventListener("start", () => { controls.autoRotate = false; });

  // Click a constellation dot on the globe → track that satellite.
  const dom = world.renderer().domElement;
  let downX = 0, downY = 0;
  dom.addEventListener("pointerdown", (e) => { downX = e.clientX; downY = e.clientY; });
  dom.addEventListener("pointerup", (e) => {
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) return;   // a drag, ignore
    if (targetPickArmed || !state.constellations.size) return;
    const r = dom.getBoundingClientRect();
    trackMember(pickMember3d(e.clientX - r.left, e.clientY - r.top));
  });

  window.addEventListener("resize", () => {
    world.width(window.innerWidth).height(window.innerHeight);
    resizeMap2d();
  });
}

function makeMarkerEl(d) {
  const el = document.createElement("div");
  if (d.isTarget) {
    el.className = "target-marker";
    el.innerHTML = `<div class="tgt-cross"></div><div class="tag">${d.name}</div>`;
    return el;
  }
  el.className = "sat-marker" + (d.satId !== state.focusId ? " dim" : "");
  el.style.setProperty("--c", d.color);
  el.innerHTML = `<div class="ring"></div><div class="core"></div><div class="tag">${d.name}</div>`;
  return el;
}

// Markers/rings for all tracked satellites, plus the observation target if set.
function allMarkers() {
  const m = [...state.tracked.values()].map((s) => s.marker);
  if (state.target) m.push(state.target.marker);
  return m;
}
function allRings() {
  const r = [...state.tracked.values()].map((s) => s.ring);
  if (state.target) r.push(state.target.ring);
  return r;
}

function updateMarkerStyles() {
  for (const s of state.tracked.values()) {
    if (s.marker.__el) s.marker.__el.classList.toggle("dim", s.id !== state.focusId);
  }
}

/* ============================================================================
 * Interpolation along a satellite's ground track
 * ========================================================================== */
function interpolate(sat, epoch) {
  const s = sat.samples;
  if (!s || s.length === 0) return null;
  if (epoch <= s[0].t) return s[0];
  if (epoch >= s[s.length - 1].t) return s[s.length - 1];

  let lo = 0, hi = s.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (s[mid].t <= epoch) lo = mid; else hi = mid;
  }
  const a = s[lo], b = s[hi];
  const f = (epoch - a.t) / (b.t - a.t || 1);
  let dlng = b.lng - a.lng;
  if (dlng > 180) dlng -= 360;
  if (dlng < -180) dlng += 360;
  let lng = a.lng + dlng * f;
  lng = ((lng + 540) % 360) - 180;
  return {
    lat: a.lat + (b.lat - a.lat) * f,
    lng,
    alt_km: a.alt_km + (b.alt_km - a.alt_km) * f,
    speed_kms: a.speed_kms + (b.speed_kms - a.speed_kms) * f,
    sunlit: f < 0.5 ? a.sunlit : b.sunlit,
  };
}

const toPt = (p) => [p.lat, p.lng, Math.max(0.005, p.alt_km / EARTH_RADIUS_KM)];

function buildPaths() {
  // Keep it simple: only the focused satellite draws a ground track. Other
  // tracked objects show just their marker + ring, so the globe stays clean.
  const epoch = displayEpoch();
  const s = state.tracked.get(state.focusId);
  if (!s || !s.samples.length) return [];

  const here = interpolate(s, epoch);
  const split = here ? toPt(here) : null;
  const past = [], future = [];
  for (const p of s.samples) (p.t <= epoch ? past : future).push(toPt(p));
  if (split) { past.push(split); future.unshift(split); }

  const paths = [
    { pts: past, color: rgba(s.rgb, 0.95), stroke: 2.2, dash: 0 },     // travelled (solid)
    { pts: future, color: rgba(s.rgb, 0.5), stroke: 1.6, dash: 1 },    // predicted (dashed)
  ];

  // Highlight the stretch within a target access window (shows when scrubbed there).
  if (state.target && state.target.windows.length) {
    const hi = s.samples
      .filter((p) => state.target.windows.some((w) => p.t >= w.start_epoch && p.t <= w.end_epoch))
      .map(toPt);
    if (hi.length > 1) paths.push({ pts: hi, color: "rgba(84,230,160,0.9)", stroke: 3.5, dash: 0 });
  }
  return paths;
}

function rebuildLayers() {
  world.htmlElementsData(allMarkers()).ringsData(allRings()).pathsData(buildPaths());
}

/* ============================================================================
 * Animation loop
 * ========================================================================== */
let lastDraw = 0;

function animate(ts) {
  requestAnimationFrame(animate);
  if (ts - lastDraw < 1000 / MARKER_FPS) return;
  lastDraw = ts;
  if (state.time.mode === "scrub") maybeRecenter();
  if (state.view === "2d") { drawMap2d(); return; }
  if (!world || !state.tracked.size) return;

  const epoch = displayEpoch();
  for (const s of state.tracked.values()) {
    const pos = interpolate(s, epoch);
    if (!pos) continue;
    s.marker.lat = pos.lat;
    s.marker.lng = pos.lng;
    s.marker.alt = Math.max(0.02, pos.alt_km / EARTH_RADIUS_KM);
    s.ring.lat = pos.lat;
    s.ring.lng = pos.lng;
    if (s.id === state.focusId) renderLiveTelemetry(pos);
  }
  updateSun();   // glide the terminator with the display time
  world.htmlElementsData(allMarkers()).ringsData(allRings());
}

function tickClock() {
  const d = new Date(displayEpoch() * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  $("utc-clock").textContent =
    `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  if (world && state.tracked.size && state.view === "3d") world.pathsData(buildPaths());
  updateTransport();
  renderEclipse();   // keep the next-eclipse countdown live
}

/* ============================================================================
 * Telemetry
 * ========================================================================== */
function renderLiveTelemetry(pos) {
  $("m-lat").textContent = `${Math.abs(pos.lat).toFixed(2)}° ${pos.lat >= 0 ? "N" : "S"}`;
  $("m-lng").textContent = `${Math.abs(pos.lng).toFixed(2)}° ${pos.lng >= 0 ? "E" : "W"}`;
  $("m-alt").textContent = `${pos.alt_km.toFixed(0)} km`;
  $("m-speed").textContent = `${pos.speed_kms.toFixed(2)} km/s`;
  const sun = $("b-sunlit");
  if (pos.sunlit === null || pos.sunlit === undefined) {
    sun.textContent = "Sun data N/A"; sun.className = "badge subtle";
  } else if (pos.sunlit) {
    sun.textContent = "☀ In sunlight"; sun.className = "badge";
  } else {
    sun.textContent = "🌑 In Earth's shadow"; sun.className = "badge shadow";
  }
}

function renderOrbitTelemetry(info) {
  const o = info.orbit;
  $("m-period").textContent = o.period_min ? `${o.period_min.toFixed(1)} min` : "—";
  $("m-incl").textContent = o.inclination_deg != null ? `${o.inclination_deg.toFixed(1)}°` : "—";
  $("sat-blurb").textContent = info.blurb || "";
  const age = info.tle.epoch_age_days;
  $("b-tle").textContent = `TLE ${age < 1 ? `${(age * 24).toFixed(0)} h` : `${age.toFixed(1)} d`} old`;
}

function renderEclipse() {
  const d = state.eclipse;
  const beta = $("e-beta"), frac = $("e-frac"), next = $("e-next");
  if (!d) {
    beta.textContent = "β —"; frac.textContent = "Sunlit —"; next.textContent = "Eclipse —";
    return;
  }
  beta.textContent = d.beta_deg == null ? "β N/A" : `β ${d.beta_deg.toFixed(1)}°`;
  if (d.per_orbit && d.per_orbit.length) {
    const f = d.per_orbit.reduce((a, o) => a + o.sunlit_fraction, 0) / d.per_orbit.length;
    frac.textContent = `Sunlit ${(f * 100).toFixed(0)}%`;
  } else {
    frac.textContent = "Sunlit —";
  }
  const ev = (d.events || []).find((e) => e.t > displayEpoch());
  if (!ev) {
    next.textContent = d.eph_available ? "No eclipse soon" : "Eclipse —";
    next.className = "badge subtle";
  } else {
    const mins = Math.max(0, Math.round((ev.t - displayEpoch()) / 60));
    const entry = ev.type === "umbra_entry";
    next.textContent = entry ? `Eclipse in ${mins}m` : `Sunlight in ${mins}m`;
    next.className = entry ? "badge shadow" : "badge";
  }
}

function renderElements(data) {
  const e = data.elements;
  const set = (id, v) => { $(id).textContent = v; };
  const deg = (v) => (v != null ? `${v.toFixed(1)}°` : "—");
  const km = (v) => (v != null ? `${v.toFixed(0)} km` : "—");
  set("el-sma", km(e.semi_major_axis_km));
  set("el-ecc", e.eccentricity != null ? e.eccentricity.toFixed(5) : "—");
  set("el-raan", deg(e.raan_deg));
  set("el-argp", deg(e.arg_perigee_deg));
  set("el-nu", deg(e.true_anomaly_deg));
  set("el-ma", deg(e.mean_anomaly_deg));
  set("el-apo", km(e.apogee_alt_km));
  set("el-per", km(e.perigee_alt_km));
  set("el-rev", e.rev_number != null ? String(e.rev_number) : "—");
}

async function refreshEclipse() {
  if (!state.focusId) return;
  try { state.eclipse = await api(`/api/eclipses/${state.focusId}?orbits=3`); renderEclipse(); }
  catch { /* keep last good values */ }
}

async function refreshElements() {
  if (!state.focusId) return;
  try { renderElements(await api(`/api/elements/${state.focusId}`)); }
  catch { /* keep last good values */ }
}

function clearFocusUI() {
  $("tele-name").textContent = "—";
  $("brand-sub").textContent = "Live satellite tracking";
  for (const id of ["m-lat", "m-lng", "m-alt", "m-speed", "m-period", "m-incl",
                    "el-sma", "el-ecc", "el-raan", "el-argp", "el-nu", "el-ma",
                    "el-apo", "el-per", "el-rev"]) $(id).textContent = "—";
  $("b-sunlit").textContent = "—"; $("b-sunlit").className = "badge subtle";
  $("sat-blurb").textContent = "";
  state.passes = [];
  state.eclipse = null;
  state.contacts = [];
  state.events = [];
  renderEclipse();
  renderPasses();
  renderContacts();
  renderTimeline();
  renderTicks();
}

/* ============================================================================
 * Tracked-satellite management
 * ========================================================================== */
async function fetchTrack(sat, epoch = null) {
  const fetchedAt = Date.now() / 1000;
  // Widen the look-ahead window at high playback rates so we refetch less often.
  const after = epoch == null ? 70 : Math.min(360, 70 * Math.max(1, state.time.rate / 10));
  const q = epoch == null
    ? "minutes_before=45&minutes_after=70&step_seconds=8"
    : `epoch=${epoch}&minutes_before=45&minutes_after=${after}&step_seconds=8`;
  const track = await api(`/api/track/${sat.id}?${q}`);
  if (epoch == null) state.clockOffset = track.now - fetchedAt;  // only live fetch sets the clock
  state.subsolar = track.subsolar;
  state.subsolarEpoch = track.subsolar_epoch ?? track.center ?? null;
  sat.samples = track.samples;
}

async function addSatellite(noradId) {
  noradId = Number(noradId);
  if (state.tracked.has(noradId)) { focusSatellite(noradId, { recenter: true }); return; }

  let info;
  try {
    info = await api(`/api/satellite/${noradId}`);
  } catch (e) {
    toast(`Couldn't track ${noradId}: ${e.message}`);
    return;
  }

  const color = nextColor();
  const sat = {
    id: noradId, name: info.name, color, rgb: hexToRgb(color), info, samples: [],
    marker: { satId: noradId, name: info.name, color, lat: 0, lng: 0, alt: 0.06, __el: null },
    ring: { lat: 0, lng: 0, rgb: hexToRgb(color) },
  };
  state.tracked.set(noradId, sat);
  state.order.push(noradId);
  saveTracked();

  try { await fetchTrack(sat); } catch (e) { toast(`Track failed: ${e.message}`); }
  updateSun();
  rebuildLayers();
  renderChips();
  $("transport")?.classList.remove("hidden");
  focusSatellite(noradId, { recenter: true });
}

function removeSatellite(noradId) {
  state.tracked.delete(noradId);
  state.order = state.order.filter((x) => x !== noradId);
  saveTracked();
  if (state.focusId === noradId) state.focusId = state.order[0] ?? null;
  rebuildLayers();
  renderChips();
  if (state.focusId) focusSatellite(state.focusId);
  else clearFocusUI();
}

function focusSatellite(noradId, { recenter = false } = {}) {
  const sat = state.tracked.get(noradId);
  if (!sat) return;
  state.focusId = noradId;
  $("tele-name").textContent = sat.name;
  $("brand-sub").textContent =
    state.tracked.size > 1 ? `${sat.name}  ·  +${state.tracked.size - 1} more` : sat.name;
  renderOrbitTelemetry(sat.info);
  refreshEclipse();
  refreshElements();
  renderChips();
  updateMarkerStyles();
  world.pathsData(buildPaths());

  if (recenter) {
    const p = interpolate(sat, displayEpoch());
    if (p) world.pointOfView({ lat: p.lat, lng: p.lng, altitude: 2.4 }, 900);
  }
  refreshPasses();
  refreshContacts();
  refreshAccess();
  refreshEvents();
}

function renderChips() {
  const ul = $("tracked-list");
  ul.innerHTML = "";
  for (const id of state.order) {
    const s = state.tracked.get(id);
    const li = document.createElement("li");
    li.className = "chip" + (id === state.focusId ? " focus" : "");
    li.style.setProperty("--c", s.color);
    li.innerHTML =
      `<span class="swatch-dot"></span><span class="chip-name">${s.name}</span>` +
      `<button class="chip-remove" title="Stop tracking">✕</button>`;
    li.addEventListener("click", (e) => {
      if (e.target.classList.contains("chip-remove")) { e.stopPropagation(); removeSatellite(id); }
      else focusSatellite(id, { recenter: true });
    });
    ul.appendChild(li);
  }
}

async function refreshAllTracks() {
  if (state.time.mode === "scrub") return;  // the recenter watchdog owns fetching while scrubbing
  await Promise.all([...state.tracked.values()].map((s) =>
    fetchTrack(s).catch(() => {})));
  updateSun();
  if (world && state.tracked.size) world.pathsData(buildPaths());
}

async function refreshFocusInfo() {
  const sat = state.tracked.get(state.focusId);
  if (!sat) return;
  try {
    sat.info = await api(`/api/satellite/${sat.id}`);
    renderOrbitTelemetry(sat.info);
  } catch { /* keep last good values */ }
  refreshEclipse();
  refreshElements();
}

function saveTracked() {
  localStorage.setItem("iss.tracked", JSON.stringify(state.order));
}

/* ============================================================================
 * Observer location & passes (for the focused satellite)
 * ========================================================================== */
function setLocation(lat, lon, { silent = false } = {}) {
  state.location = { lat, lon };
  localStorage.setItem("iss.location", JSON.stringify(state.location));
  $("loc-lat").value = lat.toFixed(4);
  $("loc-lon").value = lon.toFixed(4);
  $("loc-status").textContent =
    `Observing from ${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}, ` +
    `${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}.`;
  if (!silent) refreshPasses();
}

const compass = (az) => COMPASS[Math.round(az / 22.5) % 16];

function relTime(date) {
  const mins = Math.round((date.getTime() - Date.now()) / 60000);
  if (mins < 0) return "now";
  if (mins < 60) return `in ${mins} min`;
  return `in ${Math.floor(mins / 60)}h ${mins % 60}m`;
}

async function refreshPasses() {
  if (!state.focusId || !state.location) { renderPasses(); return; }
  const { lat, lon } = state.location;
  try {
    const data = await api(
      `/api/passes/${state.focusId}?lat=${lat}&lon=${lon}&days=3&min_elevation_deg=10`,
    );
    state.passes = data.passes || [];
  } catch (e) {
    toast(`Pass prediction failed: ${e.message}`);
    state.passes = [];
  }
  renderPasses();
}

function renderPasses() {
  const list = $("pass-list");
  list.innerHTML = "";
  if (!state.focusId) { list.innerHTML = `<li class="pass-empty">No satellite selected.</li>`; return; }
  if (!state.location) { list.innerHTML = `<li class="pass-empty">No location set yet.</li>`; return; }
  if (state.passes.length === 0) {
    list.innerHTML = `<li class="pass-empty">No passes above 10° in the next 3 days.</li>`;
    return;
  }
  const now = Date.now();
  for (const p of state.passes.slice(0, 8)) {
    const rise = new Date(p.rise || p.peak);
    const li = document.createElement("li");
    li.className = "pass";
    if (rise.getTime() - now < ALERT_LEAD_MIN * 60000 && rise.getTime() > now) li.classList.add("soon");
    const visClass = p.visible ? "visible" : "daylight";
    const visText = p.visible ? "VISIBLE" : (p.sat_sunlit === null ? "—" : "DAYLIGHT");
    const time = formatTime(rise);
    li.innerHTML =
      `<span class="when">${time}</span><span class="rel">${relTime(rise)}</span>` +
      `<span class="detail">Peak ${p.peak_elevation_deg.toFixed(0)}° · ${compass(p.peak_azimuth_deg)}</span>` +
      `<span class="vis ${visClass}">${visText}</span>`;
    list.appendChild(li);
  }
}

/* ============================================================================
 * Alerts (focused satellite)
 * ========================================================================== */
function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.connect(g); g.connect(ctx.destination);
    o.type = "sine"; o.frequency.value = 880;
    g.gain.setValueAtTime(0.0001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.25, ctx.currentTime + 0.05);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.6);
    o.start(); o.stop(ctx.currentTime + 0.6);
  } catch { /* no audio */ }
}

function showAlert(pass) {
  const rise = new Date(pass.rise || pass.peak);
  const name = state.tracked.get(state.focusId)?.name || "Satellite";
  $("alert-title").textContent = `${name} pass ${relTime(rise)}`;
  $("alert-sub").textContent =
    `Peak ${pass.peak_elevation_deg.toFixed(0)}° to the ${compass(pass.peak_azimuth_deg)}` +
    (pass.visible ? " · visible to the naked eye" : "");
  $("alert-banner").classList.remove("hidden");
  beep();
  if (state.alertsOn && "Notification" in window && Notification.permission === "granted") {
    new Notification("🛰️ Satellite pass incoming", {
      body: `${name} ${relTime(rise)} · peak ${pass.peak_elevation_deg.toFixed(0)}°`,
    });
  }
}

function checkAlerts() {
  if (!state.alertsOn || state.passes.length === 0) return;
  const now = Date.now();
  for (const p of state.passes) {
    const rise = new Date(p.rise || p.peak).getTime();
    const lead = rise - now;
    if (lead > 0 && lead <= ALERT_LEAD_MIN * 60000) {
      const id = p.rise || p.peak;
      if (state.alertedRise !== id) { state.alertedRise = id; showAlert(p); }
      break;
    }
  }
}

/* ============================================================================
 * Ground-station contacts + az/el polar plot
 * ========================================================================== */
async function loadStations() {
  try {
    const data = await api("/api/stations");
    state.stations.list = data.stations;
    // Select all by default — coverage works regardless of orbit inclination.
    state.stations.selected = data.stations.map((s) => s.station_id);
    renderStationChips();
  } catch { /* stations optional */ }
}

function renderStationChips() {
  const box = $("station-net");
  if (!box) return;
  box.innerHTML = "";
  for (const s of state.stations.list) {
    const on = state.stations.selected.includes(s.station_id);
    const b = document.createElement("button");
    b.className = "st-chip" + (on ? " on" : "");
    b.textContent = s.name.split(" (")[0];
    b.title = `${s.name} · ${s.lat.toFixed(1)}, ${s.lon.toFixed(1)} · mask ${s.elevation_mask_deg}°`;
    b.addEventListener("click", () => toggleStation(s.station_id));
    box.appendChild(b);
  }
}

function toggleStation(id) {
  const sel = state.stations.selected;
  const i = sel.indexOf(id);
  if (i >= 0) sel.splice(i, 1); else sel.push(id);
  renderStationChips();
  refreshContacts();
  refreshEvents();
}

async function refreshContacts() {
  if (!state.focusId || !state.stations.selected.length) {
    state.contacts = []; renderContacts(); return;
  }
  try {
    const ids = state.stations.selected.join(",");
    const data = await api(`/api/contacts/${state.focusId}?days=2&station_id=${ids}`);
    state.contacts = data.contacts || [];
  } catch (e) {
    toast(`Contact prediction failed: ${e.message}`);
    state.contacts = [];
  }
  renderContacts();
}

function renderContacts() {
  const list = $("contact-list");
  if (!list) return;
  list.innerHTML = "";
  if (!state.focusId) { list.innerHTML = `<li class="pass-empty">No satellite selected.</li>`; return; }
  if (!state.stations.selected.length) {
    list.innerHTML = `<li class="pass-empty">Select ground stations above.</li>`; return;
  }
  if (!state.contacts.length) {
    list.innerHTML = `<li class="pass-empty">No contacts in the next 2 days.</li>`; return;
  }
  const now = Date.now();
  for (const c of state.contacts.slice(0, 14)) {
    const aos = new Date(c.aos_utc);
    const li = document.createElement("li");
    li.className = "pass contact";
    if (aos.getTime() - now < ALERT_LEAD_MIN * 60000 && aos.getTime() > now) li.classList.add("soon");
    const time = formatTime(aos);
    li.innerHTML =
      `<span class="when">${time}</span><span class="rel">${relTime(aos)}</span>` +
      `<span class="detail">${c.station_name.split(" (")[0]} · ` +
      `${(c.duration_s / 60).toFixed(1)} min · peak ${c.max_elevation_deg.toFixed(0)}°</span>` +
      `<span class="vis az">${compass(c.aos_azimuth_deg)}→${compass(c.los_azimuth_deg)}</span>`;
    li.addEventListener("click", () => openContactDetail(c));
    list.appendChild(li);
  }
}

async function openContactDetail(c) {
  const freq = Number($("contact-freq").value) * 1e6 || null;
  $("cd-title").textContent = `${state.tracked.get(state.focusId)?.name || ""} · ${c.station_name}`;
  $("cd-polar").innerHTML = `<div class="cd-loading">Computing pass geometry…</div>`;
  $("cd-doppler").innerHTML = "";
  $("cd-meta").innerHTML = "";
  $("contact-detail").classList.remove("hidden");
  try {
    const q = `station_id=${c.station_id}&aos=${encodeURIComponent(c.aos_utc)}` +
              `&los=${encodeURIComponent(c.los_utc)}` + (freq ? `&downlink_hz=${freq}` : "");
    const prof = await api(`/api/contacts/${state.focusId}/profile?${q}`);
    drawPolar(prof.samples);
    drawDoppler(prof.samples);
    const peak = Math.max(...prof.samples.map((s) => s.alt_deg));
    $("cd-meta").textContent =
      `${(c.duration_s / 60).toFixed(1)} min · peak ${peak.toFixed(0)}° · ` +
      `${compass(c.aos_azimuth_deg)} → ${compass(c.los_azimuth_deg)}`;
  } catch (e) {
    $("cd-polar").innerHTML = `<div class="cd-loading">Failed: ${e.message}</div>`;
  }
}

// Az/el polar plot: elevation rings (90° at centre), pass arc AOS→LOS.
function drawPolar(samples) {
  const R = 90, cx = 100, cy = 105;
  const pt = (az, el) => {
    const r = (R * (90 - el)) / 90;
    const a = (az * Math.PI) / 180;
    return [cx + r * Math.sin(a), cy - r * Math.cos(a)];
  };
  let rings = "";
  for (const el of [0, 30, 60]) {
    const r = (R * (90 - el)) / 90;
    rings += `<circle cx="${cx}" cy="${cy}" r="${r.toFixed(1)}" class="pp-ring"/>`;
    rings += `<text x="${cx + 3}" y="${(cy - r + 11).toFixed(1)}" class="pp-rlabel">${el}°</text>`;
  }
  const cross =
    `<line x1="${cx}" y1="${cy - R}" x2="${cx}" y2="${cy + R}" class="pp-ax"/>` +
    `<line x1="${cx - R}" y1="${cy}" x2="${cx + R}" y2="${cy}" class="pp-ax"/>`;
  const labels =
    `<text x="${cx}" y="${cy - R - 5}" class="pp-card">N</text>` +
    `<text x="${cx + R + 4}" y="${cy + 4}" class="pp-card">E</text>` +
    `<text x="${cx}" y="${cy + R + 14}" class="pp-card">S</text>` +
    `<text x="${cx - R - 10}" y="${cy + 4}" class="pp-card">W</text>`;
  const pts = samples.filter((s) => s.alt_deg >= 0).map((s) => pt(s.az_deg, s.alt_deg));
  const poly = pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ");
  let ends = "";
  if (pts.length) {
    ends += `<circle cx="${pts[0][0].toFixed(1)}" cy="${pts[0][1].toFixed(1)}" r="3.5" class="pp-aos"/>`;
    ends += `<circle cx="${pts.at(-1)[0].toFixed(1)}" cy="${pts.at(-1)[1].toFixed(1)}" r="3.5" class="pp-los"/>`;
  }
  $("cd-polar").innerHTML =
    `<svg viewBox="0 0 200 215" class="polar">${rings}${cross}${labels}` +
    `<polyline points="${poly}" class="pp-arc"/>${ends}</svg>`;
}

function drawDoppler(samples) {
  const haveDoppler = samples.some((s) => s.doppler_hz != null);
  const key = haveDoppler ? "doppler_hz" : "range_rate_kms";
  const max = Math.max(...samples.map((s) => Math.abs(s[key]))) || 1;
  const W = 240, H = 64, pad = 4;
  const x = (i) => pad + (i / (samples.length - 1)) * (W - 2 * pad);
  const y = (v) => H / 2 - (v / max) * (H / 2 - pad);
  const poly = samples.map((s, i) => `${x(i).toFixed(1)},${y(s[key]).toFixed(1)}`).join(" ");
  const unit = haveDoppler ? `±${(max / 1000).toFixed(1)} kHz` : `±${max.toFixed(1)} km/s`;
  $("cd-doppler").innerHTML =
    `<div class="dop-title">${haveDoppler ? "Doppler shift" : "Range rate"} · ${unit}</div>` +
    `<svg viewBox="0 0 ${W} ${H}" class="doppler">` +
    `<line x1="${pad}" y1="${H / 2}" x2="${W - pad}" y2="${H / 2}" class="dop-zero"/>` +
    `<polyline points="${poly}" class="dop-line"/></svg>`;
}

function wireTabs() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      const name = tab.dataset.tab;
      for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t === tab);
      for (const pane of document.querySelectorAll(".tab-pane"))
        pane.classList.toggle("hidden", pane.id !== `tab-${name}`);
      if (name === "contacts") refreshContacts();
      if (name === "timeline") refreshEvents();
    });
  }
  $("cd-close").addEventListener("click", () => $("contact-detail").classList.add("hidden"));
}

/* ============================================================================
 * Target access (EO tasking)
 * ========================================================================== */
function setTarget(lat, lon) {
  const rgb = [84, 230, 160];
  state.target = {
    lat, lon, windows: [],
    marker: { satId: "target", name: "Target", isTarget: true, lat, lng: lon, alt: 0.001, __el: null },
    ring: { lat, lng: lon, rgb },
  };
  $("tgt-lat").value = lat.toFixed(3);
  $("tgt-lon").value = lon.toFixed(3);
  rebuildLayers();
  refreshAccess();
}

async function refreshAccess() {
  if (!state.target || !state.focusId) return;
  const off = Number($("tgt-offnadir").value) || 45;
  try {
    const data = await api(
      `/api/access/${state.focusId}?lat=${state.target.lat}&lon=${state.target.lon}` +
      `&days=2&max_off_nadir_deg=${off}`,
    );
    state.target.windows = data.windows || [];
  } catch (e) {
    toast(`Access prediction failed: ${e.message}`);
    state.target.windows = [];
  }
  renderAccess();
  if (world && state.tracked.size) world.pathsData(buildPaths());
}

function renderAccess() {
  const list = $("access-list");
  if (!list) return;
  if (!state.target) { list.innerHTML = `<li class="pass-empty">Set a target.</li>`; return; }
  if (!state.target.windows.length) {
    list.innerHTML =
      `<li class="pass-empty">No access within ${$("tgt-offnadir").value}° off-nadir in 2 days.</li>`;
    return;
  }
  list.innerHTML = "";
  for (const w of state.target.windows.slice(0, 12)) {
    const start = new Date(w.start_utc);
    const li = document.createElement("li");
    li.className = "pass contact";
    const time = formatTime(start);
    li.innerHTML =
      `<span class="when">${time}</span><span class="rel">${relTime(start)}</span>` +
      `<span class="detail">${(w.duration_s / 60).toFixed(1)} min · off-nadir ` +
      `${w.min_off_nadir_deg.toFixed(0)}° · ${w.min_slant_range_km.toFixed(0)} km</span>` +
      `<span class="vis az">peak ${w.max_elevation_deg.toFixed(0)}°</span>`;
    li.title = "Jump the time machine to this access window";
    li.addEventListener("click", () => enterScrub(w.start_epoch));
    list.appendChild(li);
  }
}

function wireTarget() {
  $("target-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const lat = parseFloat($("tgt-lat").value), lon = parseFloat($("tgt-lon").value);
    if (Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
      setTarget(lat, lon);
    } else {
      toast("Enter a valid target latitude/longitude.");
    }
  });
  $("tgt-offnadir").addEventListener("change", refreshAccess);

  const pick = $("tgt-pick");
  pick.addEventListener("click", () => {
    targetPickArmed = !targetPickArmed;
    pick.classList.toggle("active", targetPickArmed);
    pick.textContent = targetPickArmed ? "⊕ Click the map…" : "⊕ Pick on globe";
  });
  if (world && world.onGlobeClick) {
    world.onGlobeClick(({ lat, lng }) => {
      if (targetPickArmed) pickTarget(lat, lng);
    });
  }
}

// Shared between the 3D globe click and the 2D map click.
let targetPickArmed = false;

function pickTarget(lat, lon) {
  targetPickArmed = false;
  const pick = $("tgt-pick");
  if (pick) { pick.classList.remove("active"); pick.textContent = "⊕ Pick on globe"; }
  setTarget(lat, lon);
}

/* ============================================================================
 * 2D map view — equirectangular canvas with day/night textures + terminator
 * ========================================================================== */
const map2d = {
  canvas: null, ctx: null, base: null, baseCtx: null,
  dayImg: null, nightImg: null, ready: false,
  W: 0, H: 0, scale: 4, worldWidth: 720, panX: null, baseSubLng: null,
};

function initMap2d() {
  map2d.canvas = $("map2d");
  map2d.ctx = map2d.canvas.getContext("2d");
  map2d.base = document.createElement("canvas");
  map2d.baseCtx = map2d.base.getContext("2d");
  const load = (src) => { const im = new Image(); im.src = src; return im; };
  map2d.dayImg = load("/api/earth-texture");
  map2d.nightImg = load("/api/earth-night-texture");
  let loaded = 0;
  const done = () => { if (++loaded >= 2) { map2d.ready = true; buildMapBase(); } };
  map2d.dayImg.onload = done;
  map2d.nightImg.onload = done;
  resizeMap2d();
  wireMapDrag();
}

const wrapPan = (p) => { const w = map2d.worldWidth; return ((p % w) + w) % w; };

// Drag to scroll longitude infinitely; the map wraps seamlessly.
function wireMapDrag() {
  const c = map2d.canvas;
  let dragging = false, startX = 0, startPan = 0, moved = 0;
  c.style.cursor = "grab";
  c.addEventListener("pointerdown", (e) => {
    dragging = true; startX = e.clientX; startPan = map2d.panX; moved = 0;
    c.setPointerCapture(e.pointerId); c.style.cursor = "grabbing";
  });
  c.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    moved = Math.max(moved, Math.abs(dx));
    map2d.panX = wrapPan(startPan + dx);
  });
  const end = (e) => {
    if (!dragging) return;
    dragging = false; c.style.cursor = "grab";
    if (moved >= 4) return;                          // a drag, not a click
    const r = c.getBoundingClientRect();
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    if (targetPickArmed) { pickTarget(y2lat(cy), x2lon(cx)); return; }
    trackMember(pickConstellationMember(cx, cy));    // click a constellation dot → track it
  };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", () => { dragging = false; c.style.cursor = "grab"; });
}

function resizeMap2d() {
  if (!map2d.canvas) return;
  const r = map2d.canvas.getBoundingClientRect();
  map2d.W = map2d.canvas.width = Math.max(2, Math.round(r.width));
  map2d.H = map2d.canvas.height = Math.max(2, Math.round(r.height));
  // Fixed, undistorted scale: latitude fills the height (true 2:1 world).
  // A wider monitor therefore shows MORE longitude (wider FOV), never stretched.
  map2d.scale = map2d.H / 180;
  map2d.worldWidth = 2 * map2d.H;                  // = 360 * scale, exact integer
  map2d.panX = map2d.panX == null
    ? wrapPan(map2d.W / 2 - map2d.worldWidth / 2)   // center the world initially
    : wrapPan(map2d.panX);
  map2d.baseSubLng = null;
  if (map2d.ready) buildMapBase();
}

// Vertical mapping is fixed; horizontal is panned + wrapped.
const lat2y = (lat) => (90 - lat) * map2d.scale;
const baseX = (lon) => (lon + 180) * map2d.scale;   // 0 … worldWidth
const y2lat = (y) => 90 - y / map2d.scale;
const x2lon = (x) => {
  let lon = (x - map2d.panX) / map2d.scale - 180;
  lon = ((lon % 360) + 360) % 360;
  return lon > 180 ? lon - 360 : lon;
};
// Every on-screen x for a base x position (handles infinite wrap).
function wrapXs(screenX) {
  const w = map2d.worldWidth, xs = [];
  let x = ((screenX % w) + w) % w;
  for (let xi = x - w; xi < map2d.W + 80; xi += w) if (xi > -80) xs.push(xi);
  return xs;
}

// Composite day + night with a soft, subsolar-driven terminator into the base.
function buildMapBase() {
  if (!map2d.ready) return;
  const W = map2d.worldWidth, H = map2d.H;   // base holds the full 2:1 world
  map2d.base.width = W; map2d.base.height = H;
  const ctx = map2d.baseCtx;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(map2d.dayImg, 0, 0, W, H);

  const sub = currentSubsolar();
  if (!sub) return;
  // Low-res night-amount mask (feathered when scaled up = soft terminator).
  const MW = 240, MH = 120;
  const mask = document.createElement("canvas");
  mask.width = MW; mask.height = MH;
  const mctx = mask.getContext("2d");
  const data = mctx.createImageData(MW, MH);
  const sl = (sub.lat * Math.PI) / 180, sg = (sub.lng * Math.PI) / 180;
  const sinSl = Math.sin(sl), cosSl = Math.cos(sl);
  for (let j = 0; j < MH; j++) {
    const lat = ((90 - ((j + 0.5) / MH) * 180) * Math.PI) / 180;
    const sinLat = Math.sin(lat), cosLat = Math.cos(lat);
    for (let i = 0; i < MW; i++) {
      const lon = (((i + 0.5) / MW) * 360 - 180) * Math.PI / 180;
      const e = sinLat * sinSl + cosLat * cosSl * Math.cos(lon - sg);
      let t = (0.12 - e) / 0.24;          // 1 = night, 0 = day
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const k = (j * MW + i) * 4;
      data.data[k] = data.data[k + 1] = data.data[k + 2] = 255;
      data.data[k + 3] = Math.round(t * t * (3 - 2 * t) * 255);  // smoothstep
    }
  }
  mctx.putImageData(data, 0, 0);

  const nl = document.createElement("canvas");
  nl.width = W; nl.height = H;
  const nlctx = nl.getContext("2d");
  nlctx.drawImage(map2d.nightImg, 0, 0, W, H);
  nlctx.globalCompositeOperation = "destination-in";
  nlctx.drawImage(mask, 0, 0, W, H);     // scaled-up mask feathers the terminator
  ctx.drawImage(nl, 0, 0);
  map2d.baseSubLng = sub.lng;
}

function drawMap2d() {
  const { ctx, W, H } = map2d;
  if (!ctx) return;
  if (!map2d.ready) { ctx.fillStyle = "#04060f"; ctx.fillRect(0, 0, W, H); return; }

  // Rebuild the day/night base when the terminator has moved enough.
  const sub = currentSubsolar();
  if (sub && (map2d.baseSubLng == null ||
      Math.abs(((sub.lng - map2d.baseSubLng + 540) % 360) - 180) > 0.5)) {
    buildMapBase();
  }
  ctx.fillStyle = "#04060f"; ctx.fillRect(0, 0, W, H);
  // Tile the full-world base across the canvas for seamless infinite scroll.
  const ww = map2d.worldWidth;
  let x0 = Math.round(map2d.panX) % ww; if (x0 > 0) x0 -= ww;
  for (let x = x0; x < W; x += ww) ctx.drawImage(map2d.base, x, 0, ww, H);

  // Constellation clouds (tiled like the base for seamless scroll).
  if (state.constellations.size) {
    if (state.constLayerDirty || !map2d.constLayer || map2d.constLayer.width !== ww) buildConstLayer();
    for (let x = x0; x < W; x += ww) ctx.drawImage(map2d.constLayer, x, 0, ww, H);
  }

  ctx.strokeStyle = "rgba(120,160,220,0.12)"; ctx.lineWidth = 1; ctx.beginPath();
  for (let lat = -60; lat <= 60; lat += 30) { const y = lat2y(lat); ctx.moveTo(0, y); ctx.lineTo(W, y); }
  for (let lon = -180; lon < 180; lon += 30) {
    for (const x of wrapXs(map2d.panX + baseX(lon))) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
  }
  ctx.stroke();

  const epoch = displayEpoch();
  const focus = state.tracked.get(state.focusId);
  if (focus && focus.samples.length) drawTrack2d(focus, epoch);
  for (const s of state.tracked.values()) {
    const pos = interpolate(s, epoch);
    if (!pos) continue;
    drawMarker2d(pos.lng, pos.lat, s.color, s.name, s.id === state.focusId);
    if (s.id === state.focusId) renderLiveTelemetry(pos);   // keep telemetry live in 2D too
  }
  if (state.target) drawTarget2d(state.target.lon, state.target.lat);
}

function drawTrack2d(sat, epoch) {
  const ctx = map2d.ctx, ww = map2d.worldWidth, W = map2d.W;
  const here = interpolate(sat, epoch);
  const past = [], future = [];
  for (const p of sat.samples) (p.t <= epoch ? past : future).push(p);
  if (here) { past.push(here); future.unshift(here); }
  const seg = (pts, color, dash) => {
    if (pts.length < 2) return;
    // Continuous (unwrapped) longitude so the line never jumps at the dateline.
    const sx = [], sy = [];
    let prev = null, cont = 0;
    for (const p of pts) {
      if (prev == null) cont = p.lng;
      else { let d = p.lng - prev; if (d > 180) d -= 360; if (d < -180) d += 360; cont += d; }
      prev = p.lng;
      sx.push(map2d.panX + baseX(cont)); sy.push(lat2y(p.lat));
    }
    let minX = Infinity, maxX = -Infinity;
    for (const x of sx) { if (x < minX) minX = x; if (x > maxX) maxX = x; }
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash ? [6, 5] : []);
    for (let k = Math.floor(-maxX / ww) - 1; k <= Math.ceil((W - minX) / ww) + 1; k++) {
      const off = k * ww;
      ctx.beginPath(); ctx.moveTo(sx[0] + off, sy[0]);
      for (let i = 1; i < sx.length; i++) ctx.lineTo(sx[i] + off, sy[i]);
      ctx.stroke();
    }
    ctx.restore();
  };
  seg(past, rgba(sat.rgb, 0.95), false);
  seg(future, rgba(sat.rgb, 0.5), true);
}

function drawMarker2d(lon, lat, color, name, focus) {
  const ctx = map2d.ctx, y = lat2y(lat);
  for (const x of wrapXs(map2d.panX + baseX(lon))) {
    ctx.save();
    ctx.beginPath(); ctx.arc(x, y, focus ? 5 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.fill();
    ctx.shadowBlur = 0;
    ctx.font = `${focus ? 700 : 600} ${focus ? 12 : 11}px Inter, sans-serif`;
    ctx.fillStyle = "#eaf2ff"; ctx.globalAlpha = focus ? 1 : 0.75;
    ctx.fillText(name, x + 9, y + 4);
    ctx.restore();
  }
}

function drawTarget2d(lon, lat) {
  const ctx = map2d.ctx, y = lat2y(lat);
  for (const x of wrapXs(map2d.panX + baseX(lon))) {
    ctx.save();
    ctx.strokeStyle = "#54e6a0"; ctx.lineWidth = 2; ctx.shadowColor = "#54e6a0"; ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y - 11); ctx.lineTo(x, y + 11); ctx.moveTo(x - 11, y); ctx.lineTo(x + 11, y);
    ctx.stroke(); ctx.shadowBlur = 0;
    ctx.fillStyle = "#54e6a0"; ctx.font = "700 11px Inter, sans-serif";
    ctx.fillText("Target", x + 11, y + 4);
    ctx.restore();
  }
}

function setView(view) {
  state.view = view;
  const is2d = view === "2d";
  $("globe").classList.toggle("hidden", is2d);
  $("map2d").classList.toggle("hidden", !is2d);
  $("view-toggle").textContent = is2d ? "◍ 3D Globe" : "▦ 2D Map";
  if (is2d) {
    if (world && world.pauseAnimation) world.pauseAnimation();
    resizeMap2d();
  } else if (world && world.resumeAnimation) {
    world.resumeAnimation();
  }
  localStorage.setItem("iss.view", view);
}

function wireViewToggle() {
  $("view-toggle").addEventListener("click", () => setView(state.view === "3d" ? "2d" : "3d"));
}

function toggleTimezone() {
  state.timezone = state.timezone === "local" ? "utc" : "local";
  const btn = $("tz-toggle");
  btn.textContent = state.timezone === "utc" ? "🕐 UTC" : "🕐 Local";
  localStorage.setItem("iss-tracker-timezone", state.timezone);
  renderPasses(); renderContacts(); renderTarget(); renderTimeline();
}

function loadTimezonePreference() {
  const saved = localStorage.getItem("iss-tracker-timezone");
  if (saved) state.timezone = saved;
  const btn = $("tz-toggle");
  btn.textContent = state.timezone === "utc" ? "🕐 UTC" : "🕐 Local";
  $("tz-toggle").addEventListener("click", toggleTimezone);
}

/* ============================================================================
 * Constellations — add a whole CelesTrak group as a points cloud
 * ========================================================================== */
const CONST_PALETTE = ["#ff5d73", "#ffce5a", "#54e6a0", "#c792ff", "#6ab7ff",
                       "#ff8d6b", "#9be15d", "#5fd0ff", "#ff5d97", "#e0e0e0"];
let constColorIdx = 0;

async function loadConstellations() {
  try {
    const data = await api("/api/constellations");
    const picker = $("const-picker");
    for (const c of data.constellations) {
      const opt = document.createElement("option");
      opt.value = c.id; opt.textContent = c.name;
      picker.appendChild(opt);
    }
  } catch { /* constellations optional */ }
}

async function addConstellation(id) {
  if (state.constellations.has(id)) return;
  const color = CONST_PALETTE[constColorIdx++ % CONST_PALETTE.length];
  const entry = { id, name: id, color, total: 0, shown: 0, members: [], points3d: null, loading: true };
  state.constellations.set(id, entry);
  renderConstChips();
  try {
    const data = await api(`/api/constellation/${id}?epoch=${Math.round(displayEpoch())}`);
    entry.name = data.name; entry.total = data.total; entry.shown = data.shown; entry.members = data.members;
  } catch (e) {
    toast(`Couldn't load ${id}: ${e.message}`);
    state.constellations.delete(id); renderConstChips(); return;
  }
  entry.loading = false;
  buildConstPoints(entry);
  state.constLayerDirty = true;
  renderConstChips();
}

async function addConstellationByName(query) {
  const id = "name:" + query.trim().toLowerCase();
  if (state.constellations.has(id)) { toast(`"${query}" already shown`); return; }
  const color = CONST_PALETTE[constColorIdx++ % CONST_PALETTE.length];
  const entry = { id, name: query, color, total: 0, shown: 0, members: [], points3d: null, loading: true };
  state.constellations.set(id, entry);
  renderConstChips();
  try {
    const data = await api(`/api/constellation/search?q=${encodeURIComponent(query)}&epoch=${Math.round(displayEpoch())}`);
    if (data.total === 0) {
      toast(`No satellites found for "${query}"`);
      state.constellations.delete(id); renderConstChips(); return;
    }
    entry.name = data.name; entry.total = data.total; entry.shown = data.shown; entry.members = data.members;
  } catch (e) {
    toast(`Couldn't search "${query}": ${e.message}`);
    state.constellations.delete(id); renderConstChips(); return;
  }
  entry.loading = false;
  buildConstPoints(entry);
  state.constLayerDirty = true;
  renderConstChips();
}

function removeConstellation(id) {
  const e = state.constellations.get(id);
  if (e && e.points3d && world) {
    world.scene().remove(e.points3d);
    e.points3d.geometry.dispose(); e.points3d.material.dispose();
  }
  state.constellations.delete(id);
  state.constLayerDirty = true;
  renderConstChips();
}

// 3D: one THREE.Points cloud per constellation (constant-size GL points).
function buildConstPoints(entry) {
  if (!world) return;
  if (entry.points3d) {
    world.scene().remove(entry.points3d);
    entry.points3d.geometry.dispose(); entry.points3d.material.dispose();
  }
  const pos = new Float32Array(entry.members.length * 3);
  entry.members.forEach((m, i) => {
    const c = world.getCoords(m.lat, m.lng, m.alt_km / EARTH_RADIUS_KM);
    pos[i * 3] = c.x; pos[i * 3 + 1] = c.y; pos[i * 3 + 2] = c.z;
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({
    color: entry.color, size: 2.6, sizeAttenuation: false, transparent: true, opacity: 0.9,
  });
  entry.points3d = new THREE.Points(geo, mat);
  entry.points3d.renderOrder = 2;
  world.scene().add(entry.points3d);
}

// 2D: draw every constellation member onto a world-space layer (then tiled).
function buildConstLayer() {
  const ww = map2d.worldWidth, H = map2d.H;
  if (!map2d.constLayer) map2d.constLayer = document.createElement("canvas");
  map2d.constLayer.width = ww; map2d.constLayer.height = H;
  const ctx = map2d.constLayer.getContext("2d");
  ctx.clearRect(0, 0, ww, H);
  for (const e of state.constellations.values()) {
    ctx.fillStyle = e.color;
    for (const m of e.members) {
      ctx.fillRect((m.lng + 180) * map2d.scale - 1, (90 - m.lat) * map2d.scale - 1, 2.4, 2.4);
    }
  }
  state.constLayerDirty = false;
}

async function refreshConstellations() {
  if (!state.constellations.size) return;
  const epoch = Math.round(displayEpoch());
  await Promise.all([...state.constellations.values()].map(async (e) => {
    try {
      const data = await api(`/api/constellation/${e.id}?epoch=${epoch}`);
      e.members = data.members; e.total = data.total; e.shown = data.shown;
      buildConstPoints(e);
    } catch { /* keep last positions */ }
  }));
  state.constLayerDirty = true;
}

// Hit-test a screen click against constellation members → nearest member.
function pickMember2d(x, y) {
  let best = null, bestD = 14 * 14;
  for (const e of state.constellations.values()) {
    for (const m of e.members) {
      const sy = (90 - m.lat) * map2d.scale;
      if (Math.abs(sy - y) > 14) continue;
      for (const sx of wrapXs(map2d.panX + baseX(m.lng))) {
        const d = (sx - x) ** 2 + (sy - y) ** 2;
        if (d < bestD) { bestD = d; best = m; }
      }
    }
  }
  return best;
}

function pickMember3d(x, y) {
  if (!world) return null;
  const cam = world.camera(), cp = cam.position;
  const dom = world.renderer().domElement;
  const W = dom.clientWidth, H = dom.clientHeight;
  const v = new THREE.Vector3();
  let best = null, bestD = 14 * 14;
  for (const e of state.constellations.values()) {
    for (const m of e.members) {
      const c = world.getCoords(m.lat, m.lng, m.alt_km / EARTH_RADIUS_KM);
      if (c.x * cp.x + c.y * cp.y + c.z * cp.z <= 0) continue;   // far side → occluded
      v.set(c.x, c.y, c.z).project(cam);
      if (v.z > 1) continue;
      const sx = (v.x * 0.5 + 0.5) * W, sy = (-v.y * 0.5 + 0.5) * H;
      const d = (sx - x) ** 2 + (sy - y) ** 2;
      if (d < bestD) { bestD = d; best = m; }
    }
  }
  return best;
}

function pickConstellationMember(x, y) {
  return state.view === "2d" ? pickMember2d(x, y) : pickMember3d(x, y);
}

// Promote a clicked constellation member to a fully-tracked satellite.
function trackMember(m) {
  if (m && m.norad_id) { addSatellite(m.norad_id); toast(`Tracking ${m.name}`); }
}

function renderConstChips() {
  const ul = $("const-list");
  ul.innerHTML = "";
  for (const e of state.constellations.values()) {
    const li = document.createElement("li");
    li.className = "chip const-chip";
    li.style.setProperty("--c", e.color);
    const count = e.loading ? "…" : (e.shown < e.total ? `${e.shown}/${e.total}` : `${e.total}`);
    li.innerHTML =
      `<span class="swatch-dot"></span><span class="chip-name">${e.name}</span>` +
      `<span class="const-count">${count}</span>` +
      `<button class="chip-remove" title="Remove constellation">✕</button>`;
    li.querySelector(".chip-remove").addEventListener("click", (ev) => {
      ev.stopPropagation(); removeConstellation(e.id);
    });
    ul.appendChild(li);
  }
}

/* ============================================================================
 * Event timeline + export
 * ========================================================================== */
const EVENT_ICONS = {
  contact_aos: "📡", contact_los: "📡", umbra_entry: "🌑", umbra_exit: "☀",
  terminator_day: "🌅", terminator_night: "🌆", apogee: "▲", perigee: "▼", access: "🎯",
};

async function refreshEvents() {
  if (!state.focusId) { state.events = []; renderTimeline(); return; }
  const ids = state.stations.selected.join(",");
  try {
    const data = await api(`/api/events/${state.focusId}?hours=24&station_id=${ids}`);
    const evs = data.events || [];
    if (state.target) {
      for (const w of state.target.windows) {
        evs.push({
          t: w.start_epoch, utc: w.start_utc, type: "access", label: "Target access",
          detail: { duration_s: w.duration_s, max_elevation_deg: w.max_elevation_deg, end_utc: w.end_utc },
        });
      }
    }
    evs.sort((a, b) => a.t - b.t);
    state.events = evs;
  } catch { state.events = []; }
  renderTimeline();
  renderTicks();
}

function tickClass(type) {
  if (type.startsWith("contact")) return "tk-contact";
  if (type.startsWith("umbra")) return "tk-eclipse";
  if (type.startsWith("terminator")) return "tk-term";
  if (type === "access") return "tk-access";
  return "tk-apsis";
}

// Place event marks along the transport scrubber's [now−6h, now+24h] window.
function renderTicks() {
  const box = $("t-ticks");
  if (!box) return;
  const { start, end } = transportWindow();
  const span = end - start || 1;
  box.innerHTML = "";
  for (const e of state.events) {
    const f = (e.t - start) / span;
    if (f < 0 || f > 1) continue;
    const tick = document.createElement("div");
    tick.className = `t-tick ${tickClass(e.type)}`;
    tick.style.left = `${(f * 100).toFixed(2)}%`;
    tick.title = e.label;
    box.appendChild(tick);
  }
}

function renderTimeline() {
  const list = $("event-list");
  if (!list) return;
  if (!state.focusId) { list.innerHTML = `<li class="pass-empty">No satellite selected.</li>`; return; }
  const upcoming = state.events.filter((e) => e.t > displayEpoch() - 1800);
  if (!upcoming.length) { list.innerHTML = `<li class="pass-empty">No events in the next 24 hours.</li>`; return; }
  list.innerHTML = "";
  for (const e of upcoming.slice(0, 40)) {
    const d = new Date(e.t * 1000);
    const li = document.createElement("li");
    li.className = "pass evt";
    const time = formatTime(d);
    li.innerHTML =
      `<span class="when">${EVENT_ICONS[e.type] || "•"} ${e.label}</span>` +
      `<span class="rel">${time} · ${relTime(d)}</span>`;
    li.title = "Jump the time machine here";
    li.addEventListener("click", () => enterScrub(e.t));
    list.appendChild(li);
  }
}

function download(filename, mime, text) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const csvCell = (v) => {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

function eventsToCSV(events) {
  const rows = [["type", "label", "utc", "epoch", "detail"]];
  for (const e of events) {
    rows.push([e.type, e.label, new Date(e.t * 1000).toISOString(), e.t.toFixed(0),
               JSON.stringify(e.detail || {})]);
  }
  return rows.map((r) => r.map(csvCell).join(",")).join("\n");
}

const icsTime = (epoch) =>
  new Date(epoch * 1000).toISOString().replace(/[-:]/g, "").replace(/\.\d+/, "");

const icsFold = (s) => (s.length <= 73 ? s : s.match(/.{1,73}/g).join("\r\n "));

function eventsToICS(events) {
  const name = state.tracked.get(state.focusId)?.name || "Satellite";
  const stamp = icsTime(nowEpoch());
  const lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Orbital Track//EN", "CALSCALE:GREGORIAN"];
  for (const e of events) {
    const end = e.detail?.duration_s
      ? e.t + e.detail.duration_s
      : (e.detail?.end_utc ? Date.parse(e.detail.end_utc) / 1000 : e.t + 60);
    lines.push(
      "BEGIN:VEVENT",
      `UID:${name.replace(/\s/g, "")}-${e.type}-${Math.round(e.t)}@orbital-track`,
      `DTSTAMP:${stamp}`,
      `DTSTART:${icsTime(e.t)}`,
      `DTEND:${icsTime(end)}`,
      icsFold(`SUMMARY:${name} — ${e.label}`),
      "END:VEVENT",
    );
  }
  lines.push("END:VCALENDAR");
  return lines.join("\r\n");
}

function wireTimeline() {
  $("tl-csv").addEventListener("click", () => {
    if (!state.events.length) return toast("No events to export.");
    download("orbital-track-events.csv", "text/csv", eventsToCSV(state.events));
  });
  $("tl-ics").addEventListener("click", () => {
    if (!state.events.length) return toast("No events to export.");
    download("orbital-track-events.ics", "text/calendar", eventsToICS(state.events));
  });
}

/* ============================================================================
 * Time machine — transport bar
 * ========================================================================== */
let lastRecenterWall = 0;

function transportWindow() {
  const now = nowEpoch();
  return { start: now - SCRUB_BACK_HOURS * 3600, end: now + SCRUB_FWD_HOURS * 3600 };
}

function enterScrub(epoch) {
  const t = state.time;
  t.mode = "scrub";
  t.playing = false;
  t.scrubEpoch = epoch;
  t.lastRecenterEpoch = null;   // force the watchdog to refetch around the new epoch
  if (world) world.controls().autoRotate = false;
  updateTransport();
}

function setPlaying(playing) {
  const t = state.time;
  if (playing) {
    if (t.mode === "live") { t.mode = "scrub"; t.scrubEpoch = nowEpoch(); }
    t.anchorScrub = t.scrubEpoch;
    t.anchorWall = Date.now() / 1000;
    t.playing = true;
    if (world) world.controls().autoRotate = false;
  } else {
    t.scrubEpoch = displayEpoch();   // freeze where we are
    t.playing = false;
  }
  updateTransport();
}

function setRate(rate) {
  const t = state.time;
  if (t.playing) { t.anchorScrub = displayEpoch(); t.anchorWall = Date.now() / 1000; }
  t.rate = rate;
  updateTransport();
}

function jumpOrbit(direction) {
  const sat = state.tracked.get(state.focusId);
  const periodMin = sat?.info?.orbit?.period_min || 92.0;
  enterScrub(displayEpoch() + direction * periodMin * 60);
}

function jumpNextEvent() {
  const next = state.events.find((e) => e.t > displayEpoch() + 1);
  if (next) enterScrub(next.t);
  else toast("No upcoming events in the timeline.");
}

function returnToLive() {
  const t = state.time;
  t.mode = "live";
  t.playing = false;
  t.scrubEpoch = null;
  t.lastRecenterEpoch = null;
  refreshAllTracks();   // restore the now-centered window
  if (state.focusId) focusSatellite(state.focusId, { recenter: true });
  updateTransport();
}

async function recenterNow() {
  const t = state.time;
  if (t.recentering) return;
  t.recentering = true;
  lastRecenterWall = Date.now() / 1000;
  const epoch = displayEpoch();
  try {
    await Promise.all([...state.tracked.values()].map((s) => fetchTrack(s, epoch).catch(() => {})));
    t.lastRecenterEpoch = epoch;
    if (world && state.tracked.size) world.pathsData(buildPaths());
  } finally {
    t.recentering = false;
  }
}

function maybeRecenter() {
  const t = state.time;
  if (t.recentering) return;
  const epoch = displayEpoch();
  const rate = Math.max(1, t.rate);
  const threshold = Math.min(40 * 60, Math.max(8, (40 * 60) / rate));  // sim seconds of drift
  if (t.lastRecenterEpoch != null && Math.abs(epoch - t.lastRecenterEpoch) < threshold) return;
  if (Date.now() / 1000 - lastRecenterWall < 1.5) return;              // real-time debounce
  recenterNow();
}

function formatOffset(sec) {
  const sign = sec < 0 ? "−" : "+";
  sec = Math.abs(Math.round(sec));
  const pad = (n) => String(n).padStart(2, "0");
  return `${sign}${pad(Math.floor(sec / 3600))}:${pad(Math.floor((sec % 3600) / 60))}:${pad(sec % 60)}`;
}

function updateTransport() {
  const bar = $("transport");
  if (!bar) return;
  const t = state.time;
  $("t-live").classList.toggle("active", t.mode === "live");
  $("t-play").textContent = t.playing ? "⏸" : "▶";
  for (const b of document.querySelectorAll(".t-rate button"))
    b.classList.toggle("active", Number(b.dataset.rate) === t.rate);

  const { start, end } = transportWindow();
  const epoch = displayEpoch();
  const range = $("t-range");
  if (range && document.activeElement !== range) {
    range.value = String(Math.round(((epoch - start) / (end - start)) * 1000));
  }
  const iso = new Date(epoch * 1000).toISOString().slice(0, 19).replace("T", " ");
  $("t-readout").textContent = `${iso} UTC  ·  ${formatOffset(epoch - nowEpoch())}`;
}

function wireTransport() {
  $("t-live").addEventListener("click", returnToLive);
  $("t-play").addEventListener("click", () => setPlaying(!state.time.playing));
  $("t-orbit-back").addEventListener("click", () => jumpOrbit(-1));
  $("t-orbit-fwd").addEventListener("click", () => jumpOrbit(1));
  $("t-next-event").addEventListener("click", jumpNextEvent);
  for (const b of document.querySelectorAll(".t-rate button"))
    b.addEventListener("click", () => setRate(Number(b.dataset.rate)));
  $("t-range").addEventListener("input", () => {
    const { start, end } = transportWindow();
    enterScrub(start + (Number($("t-range").value) / 1000) * (end - start));
  });
}

/* ============================================================================
 * Wiring & init
 * ========================================================================== */
async function loadCatalog() {
  const data = await api("/api/catalog");
  const picker = $("sat-picker");
  for (const sat of data.satellites) {
    const opt = document.createElement("option");
    opt.value = sat.norad_id;
    opt.textContent = `${sat.name} · ${sat.category}`;
    picker.appendChild(opt);
  }
  return data.default_norad_id;
}

/* --- Full-catalog search -------------------------------------------------- */
let searchTimer = null;

async function runSearch(query) {
  try {
    const data = await api(`/api/search?q=${encodeURIComponent(query)}&limit=30`);
    renderSearchResults(data.results || []);
  } catch {
    renderSearchResults([]);
  }
}

function renderSearchResults(results) {
  const box = $("search-results");
  box.innerHTML = "";
  if (!results.length) { box.classList.add("hidden"); return; }
  for (const r of results) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="sr-name">${r.name}</span><span class="sr-id">#${r.norad_id}</span>`;
    // mousedown fires before the input's blur, so the click isn't lost.
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      addSatellite(r.norad_id);
      $("sat-search").value = "";
      box.classList.add("hidden");
    });
    box.appendChild(li);
  }
  box.classList.remove("hidden");
}

function wireSearch() {
  const input = $("sat-search");
  const box = $("search-results");
  input.addEventListener("input", () => {
    const q = input.value.trim();
    clearTimeout(searchTimer);
    if (q.length < 2) { box.classList.add("hidden"); return; }
    searchTimer = setTimeout(() => runSearch(q), 250);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const q = input.value.trim();
      const first = box.querySelector("li");
      if (/^\d+$/.test(q)) { addSatellite(Number(q)); input.value = ""; box.classList.add("hidden"); }
      else if (first) first.dispatchEvent(new MouseEvent("mousedown"));
    } else if (e.key === "Escape") {
      box.classList.add("hidden");
    }
  });
  input.addEventListener("blur", () => setTimeout(() => box.classList.add("hidden"), 150));
  input.addEventListener("focus", () => { if (box.children.length) box.classList.remove("hidden"); });
}

function wireEvents() {
  $("sat-picker").addEventListener("change", (e) => {
    const id = Number(e.target.value);
    if (id > 0) addSatellite(id);
    e.target.selectedIndex = 0;
  });

  $("const-picker").addEventListener("change", (e) => {
    if (e.target.value) addConstellation(e.target.value);
    e.target.selectedIndex = 0;
  });

  $("const-search").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const q = e.target.value.trim();
    if (q) { addConstellationByName(q); e.target.value = ""; }
  });

  $("loc-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const lat = parseFloat($("loc-lat").value), lon = parseFloat($("loc-lon").value);
    if (Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
      setLocation(lat, lon);
    } else {
      toast("Enter a valid latitude (−90…90) and longitude (−180…180).");
    }
  });

  $("locate-btn").addEventListener("click", () => {
    if (!navigator.geolocation) return toast("Geolocation isn't available in this browser.");
    $("loc-status").textContent = "Locating…";
    navigator.geolocation.getCurrentPosition(
      (pos) => setLocation(pos.coords.latitude, pos.coords.longitude),
      () => toast("Couldn't get your location — enter it manually."),
      { enableHighAccuracy: false, timeout: 10000 },
    );
  });

  $("alerts-toggle").addEventListener("change", async (e) => {
    state.alertsOn = e.target.checked;
    localStorage.setItem("iss.alerts", state.alertsOn ? "1" : "0");
    if (state.alertsOn && "Notification" in window && Notification.permission === "default") {
      await Notification.requestPermission();
    }
    toast(state.alertsOn ? "Pass alerts on." : "Pass alerts off.");
  });

  $("alert-dismiss").addEventListener("click", () => $("alert-banner").classList.add("hidden"));
}

async function init() {
  try {
  initGlobe();
  wireEvents();
  wireTransport();
  wireTabs();
  wireTarget();
  wireTimeline();
  wireSearch();
  initMap2d();
  wireViewToggle();
  loadTimezonePreference();
  loadStations();
  loadConstellations();

  const savedLoc = localStorage.getItem("iss.location");
  if (savedLoc) {
    try { const l = JSON.parse(savedLoc); setLocation(l.lat, l.lon, { silent: true }); } catch { /* ignore */ }
  }
  if (localStorage.getItem("iss.alerts") === "1") {
    state.alertsOn = true;
    $("alerts-toggle").checked = true;
  }

  const defaultId = await loadCatalog().catch(() => 25544);
  let saved = [];
  try { saved = JSON.parse(localStorage.getItem("iss.tracked") || "[]"); } catch { /* ignore */ }
  if (!Array.isArray(saved) || saved.length === 0) saved = [defaultId];

  for (const id of saved) await addSatellite(id);   // sequential keeps colors stable

  if (localStorage.getItem("iss.view") === "2d") setView("2d");

  requestAnimationFrame(animate);
  setInterval(tickClock, 1000);
  setInterval(refreshAllTracks, TRACK_REFRESH_MS);
  setInterval(refreshFocusInfo, INFO_REFRESH_MS);
  setInterval(refreshPasses, PASS_REFRESH_MS);
  setInterval(refreshEvents, 90_000);
  setInterval(checkAlerts, 15000);
  setInterval(refreshConstellations, 8000);
  tickClock();
  } catch (e) {
    console.error("INIT FAILED:", e && (e.stack || e.message || String(e)));
  }
}

window.addEventListener("DOMContentLoaded", init);

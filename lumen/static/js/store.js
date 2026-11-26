// Application state.  One plain object, explicit updates, subscribers re-render.
import { api } from './api.js';

// The way through a job.  Five stops and the records; the scale check is a tool that is reached from "Lines" and is
// not a stop of its own (nav: false), so that the usual way has nothing on it that can be skipped.
export const STEPS = [
  { id: 'identify', label: 'Job' },
  { id: 'markers', label: 'Tags' },
  { id: 'capture', label: 'Photos' },
  { id: 'datums', label: 'Lines' },
  { id: 'rectified', label: 'Scale check', nav: false, parent: 'datums' },
  { id: 'place', label: 'Place' },
  { id: 'verify', label: 'Records' },
];
STEPS.filter((st) => st.nav !== false).forEach((st, i) => { st.n = i + 1; });

export const state = {
  config: null,
  operations: null,
  session: null,
  step: 'identify',
  theme: 'dark',
  units: 'mm',
  frameId: null,          // frame shown on Datums / Rectified / Place
  busy: false,
};

const subs = new Set();
export function subscribe(fn) { subs.add(fn); return () => subs.delete(fn); }
export function emit() { subs.forEach((fn) => { try { fn(state); } catch (e) { console.error(e); } }); }
export function setState(patch) { Object.assign(state, patch); emit(); }

// ---------------------------------------------------------- persistence
const LS = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) { /* private mode */ } },
};
export const prefs = {
  get theme() { return LS.get('lumen.theme', null); },
  set theme(v) { LS.set('lumen.theme', v); },
  get units() { return LS.get('lumen.units', 'mm'); },
  set units(v) { LS.set('lumen.units', v); },
  get tagSizes() { return LS.get('lumen.tagSizes', { nominal_mm: 50, measured_mm: {} }); },
  set tagSizes(v) { LS.set('lumen.tagSizes', v); },
  get calibrations() { return LS.get('lumen.calibrations', []); },
  set calibrations(v) { LS.set('lumen.calibrations', v); },
  get placeSound() { return LS.get('lumen.placeSound', true); },
  set placeSound(v) { LS.set('lumen.placeSound', !!v); },
  get placeAuto() { return LS.get('lumen.placeAuto', true); },
  set placeAuto(v) { LS.set('lumen.placeAuto', !!v); },
};
export function rememberSession(id) {
  try { if (id) sessionStorage.setItem('lumen.sid', id); else sessionStorage.removeItem('lumen.sid'); } catch (_) { /* ignore */ }
}
export function rememberedSession() {
  try { return sessionStorage.getItem('lumen.sid'); } catch (_) { return null; }
}

export function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', theme === 'light' ? '#ffffff' : '#0e1a2b');
}

// The installation's brand: its colour is applied to the page; name and logo are drawn by the header.
export function brandInfo() {
  return (state.config && state.config.brand) || (state.boot && state.boot.brand) || {};
}
// Relative luminance of an sRGB colour (channels 0..255) and the contrast of two luminances, as WCAG defines them.
function luminance(rgb) {
  const [r, g, b] = rgb.map((c) => { const x = c / 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
const contrast = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
const INK_DARK = [10, 26, 51];   // the page's text colour

// The installation's colour, where it gave one.  It is used as it is where it is a surface (the rule under the header,
// the mark); what is drawn on it takes whichever of white and the dark text colour reads better.  Text and buttons on
// white need a colour that can be read there: one that is too light for that is darkened until it can.
export function applyBrand() {
  const b = brandInfo();
  const root = document.documentElement;
  ['--brand', '--brand-ink', '--brand-accent'].forEach((k) => root.style.removeProperty(k));
  if (!(b.color && /^#[0-9a-f]{6}$/i.test(b.color))) return;
  const rgb = [1, 3, 5].map((i) => parseInt(b.color.slice(i, i + 2), 16));
  const lum = luminance(rgb);
  root.style.setProperty('--brand', b.color);
  root.style.setProperty('--brand-ink', contrast(lum, 1) >= contrast(lum, luminance(INK_DARK)) ? '#ffffff' : '#0a1a33');
  let acc = rgb;
  for (let i = 0; i < 40 && contrast(luminance(acc), 1) < 4.6; i += 1) acc = acc.map((c) => c * 0.93);
  if (acc !== rgb) root.style.setProperty('--brand-accent', `rgb(${acc.map((c) => Math.round(c)).join(', ')})`);
}

// What kind of results this installation gives.  It is shown in one place on every screen.
//   demo        computer-generated photos: everything is simulated
//   evaluation  real photos, but the installation has not been validated against reference measurements
//   validated   an administrator supplied a validated range; results inside it are marked as such
export function resultMode(s = state.session) {
  if (s && s.mode === 'demo') {
    return { kind: 'demo', label: 'Demo', icon: 'play', watermark: 'Demo · simulated data', title: 'Demo data',
      text: 'This session runs on computer-generated photos of an invented panel. Everything it shows is simulated and says nothing about a real part.' };
  }
  const dom = (state.config && state.config.validated_domain) || null;
  if (dom || (!state.config && state.boot && state.boot.validated_domain)) {
    return { kind: 'validated', label: 'Validated', icon: 'check', watermark: 'Measurement evidence · not acceptance', title: 'Validated range',
      text: `Results inside the validated range${dom && dom.id ? ` (${dom.id})` : ''} are marked as validated. A result outside it is marked "check needed" and says why.${dom && dom.evidence ? ` Evidence: ${dom.evidence}.` : ''} A measurement is evidence for your quality process; it is not the acceptance decision.` };
  }
  return { kind: 'evaluation', label: 'Evaluation', icon: 'info', watermark: 'Evaluation only · not acceptance', title: 'Evaluation: not validated',
    text: 'This installation has not been checked against reference measurements, so its accuracy on real parts is not established. Use its results as measurement evidence within your evaluation. Whether a part is accepted is decided by your quality process, not by this app. Before operational use it needs an independent metrology validation, and the placard’s surface and adhesive need their own approval. The indicator changes to “Validated” when an administrator supplies the range that such a study has validated.' };
}

// ------------------------------------------------------------- session
export function setSession(s) {
  state.session = s;
  rememberSession(s ? s.id : null);
  if (s && state.frameId && !s.frames.some((f) => f.id === state.frameId)) state.frameId = null;
  emit();
}

export async function ensureSession(mode = 'real') {
  if (state.session) return state.session;
  const s = await api('POST', 'api/sessions', { mode });
  setSession(s);
  return s;
}

export async function refreshSession() {
  if (!state.session) return null;
  const s = await api('GET', `api/sessions/${state.session.id}`);
  setSession(s);
  return s;
}

export async function endSession() {
  const s = state.session;
  if (s) { try { await api('DELETE', `api/sessions/${s.id}`); } catch (_) { /* already gone */ } }
  state.frameId = null;
  setSession(null);
  setState({ step: 'identify' });
}

// ------------------------------------------------------------- derived
export function registrationFrames(s = state.session) {
  return s ? s.frames.filter((f) => f.role === 'registration') : [];
}
export function acceptedTags(frame) {
  return frame.markers.filter((m) => m.accepted);
}
export function activeFrameId() {
  const s = state.session;
  if (!s || !s.registration) return null;
  if (state.frameId && s.registration.frames_used.includes(state.frameId)) return state.frameId;
  return s.registration.reference_frame_id;
}
export function operationSummary(s = state.session) {
  return s && s.operation ? s.operation.summary : null;
}
export function activePlacard(s = state.session) {
  if (!s || !s.operation) return null;
  return (s.operation.placards_resolved || []).find((p) => p.placard_id === s.active_placard_id) || null;
}

// What each step needs before it can be opened (fail-forward navigation only).
export function stepAvailability(s = state.session) {
  const hasOp = !!(s && s.operation);
  const frames = registrationFrames(s);
  const hasTags = frames.some((f) => f.selected && acceptedTags(f).length > 0);
  const reg = !!(s && s.registration);
  const frameOk = !!(s && s.local_frame && s.local_frame.valid);
  return {
    identify: { ok: true },
    markers: { ok: hasOp, why: 'Choose the job first.' },
    capture: { ok: hasOp, why: 'Choose the job first.' },
    datums: { ok: hasTags || reg, why: 'Take a photo that shows the tags first.' },
    rectified: { ok: reg, why: 'Analyse the photos first (Photos step).' },
    place: { ok: reg, why: 'Analyse the photos first (Photos step).' },
    verify: { ok: reg && frameOk, why: 'Confirm the lines first (Lines step).' },
  };
}

export function stepDone(id, s = state.session) {
  if (!s) return false;
  switch (id) {
    case 'identify': return !!s.operation && !!s.active_placard_id;
    case 'markers': return !!s.operation && Object.keys((s.tag_sizes || {}).measured_mm || {}).length > 0;
    case 'capture': return !!s.registration;
    case 'datums': return !!(s.local_frame && s.local_frame.valid && s.datums.floor && s.datums.side && s.datums.floor.confirmed && s.datums.side.confirmed);
    case 'place': return (s.verifications || []).some((v) => v.placard_id === s.active_placard_id && v.observation === 'within_tolerance_observed');
    case 'verify': return (s.verifications || []).some((v) => v.confirmed);
    default: return false;
  }
}

export function lengthOut(mm, digits = 1) {
  if (mm === null || mm === undefined) return '—';
  if (state.units === 'in') return `${(mm / 25.4).toFixed(digits + 2)} in`;
  return `${mm.toFixed(digits)} mm`;
}

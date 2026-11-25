// Application state.  One plain object, explicit updates, subscribers re-render.
import { api } from './api.js';

export const STEPS = [
  { id: 'identify', label: 'Identify', n: 1 },
  { id: 'markers', label: 'Markers', n: 2 },
  { id: 'capture', label: 'Capture', n: 3 },
  { id: 'datums', label: 'Datums', n: 4 },
  { id: 'rectified', label: 'Rectified', n: 5 },
  { id: 'place', label: 'Place', n: 6 },
  { id: 'verify', label: 'Verify', n: 7 },
];

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
  if (meta) meta.setAttribute('content', theme === 'light' ? '#ffffff' : '#121722');
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
    markers: { ok: hasOp, why: 'Choose an operation first.' },
    capture: { ok: hasOp, why: 'Choose an operation first.' },
    datums: { ok: hasTags || reg, why: 'Capture at least one photo with an accepted tag.' },
    rectified: { ok: reg, why: 'Run the analysis on the Capture step first.' },
    place: { ok: reg, why: 'Run the analysis on the Capture step first.' },
    verify: { ok: reg && frameOk, why: 'Define and confirm the datums first.' },
  };
}

export function stepDone(id, s = state.session) {
  if (!s) return false;
  switch (id) {
    case 'identify': return !!s.operation && !!s.active_placard_id;
    case 'markers': return !!s.operation && Object.keys((s.tag_sizes || {}).measured_mm || {}).length > 0;
    case 'capture': return !!s.registration;
    case 'datums': return !!(s.local_frame && s.local_frame.valid && s.datums.floor && s.datums.side && s.datums.floor.confirmed && s.datums.side.confirmed);
    case 'verify': return (s.verifications || []).some((v) => v.confirmed);
    default: return false;
  }
}

export function lengthOut(mm, digits = 1) {
  if (mm === null || mm === undefined) return '—';
  if (state.units === 'in') return `${(mm / 25.4).toFixed(digits + 2)} in`;
  return `${mm.toFixed(digits)} mm`;
}

// Assisted placement: the placard is measured in every live frame while it is being positioned.
//
// What the operator gets
// ----------------------
// * on the picture: the placard's measured outline and an arrow to the target;
// * beside it: a gauge that magnifies the last millimetres (ring = the record's
//   tolerance; disc = the placard's centre, as large as the uncertainty of this
//   view), a level for the rotation and one line that says what to do;
// * by ear (sound.js): beeps that come faster, a steady tone in position, a
//   chime when the placement is recorded;
// * a green flash when the placard reaches its position.
//
// What the green means
// --------------------
// "In position" is the server's judgement of one frame: the measured deviation
// plus the expanded uncertainty of that frame lies inside the tolerance of the
// record.  It is guidance and nothing is recorded by it.  The record is a
// stored full-size photo, measured by the ordinary verification path.  It is
// taken automatically once the placard has been in position, with nothing
// covering it, for a few frames in a row (or when the operator asks for it),
// and it is announced as recorded only if that measurement, too, is inside
// the tolerance.
import { api, upload } from '../api.js';
import { h, clear, icon, chip, fmt } from '../dom.js';
import { state, setSession, prefs } from '../store.js';
import { Feedback } from '../sound.js';
import { flagText, STATUS_TEXT } from './common.js';

const NEED_IN_A_ROW = 3;          // frames in position, nothing covering the placard, before the record is taken ...
const NEED_HELD_MS = 1000;        // ... and for at least this long, whatever the frame rate
const REARM_MS = 2500;            // after a record photo that did not confirm it, the next one waits this long (or until the placard has moved)
const HINT_MAX_AGE_MS = 2500;     // where the placard was: only believed for this long
const SHIFT_REFRESH_MS = 5000;    // the stream's edge shift is measured afresh this often

const COLOR = { idle: '#a8b3c4', info: '#5cc8ff', guide: '#5cc8ff', warn: '#ffb020', ok: '#3ddc97', bad: '#ff5d5d' };
const PHASE = {
  starting: { cls: 'idle', title: 'Starting…' },
  lost: { cls: 'idle', title: 'No reading' },
  blocked: { cls: 'bad', title: 'No target in this view' },
  searching: { cls: 'idle', title: 'Bring the placard' },
  partial: { cls: 'idle', title: 'Placard partly seen' },
  not_flat: { cls: 'warn', title: 'Not flat on the surface' },
  unjudged: { cls: 'info', title: 'No tolerance in the record' },
  coarse: { cls: 'warn', title: 'Reading too coarse here' },
  doubt: { cls: 'warn', title: 'Outline not clear' },
  guiding: { cls: 'guide', title: 'Move it' },
  close: { cls: 'warn', title: 'Almost' },
  in_position: { cls: 'ok', title: 'In position' },
  recording: { cls: 'ok', title: 'Recording…' },
  recorded: { cls: 'ok', title: 'Recorded' },
};

// The state of one live answer.
export function phaseOf(res) {
  const t = res && res.tracking;
  if (!t || t.state !== 'tracking') return 'lost';
  const p = res.placement;
  if (!p) return 'lost';
  if (p.state === 'no_target' || p.state === 'blocked') return 'blocked';
  if (!p.found) return 'searching';
  if (!p.complete) return 'partial';
  if ((p.flags || []).includes('not_flat_on_surface')) return 'not_flat';
  if (p.tolerance_mm === null || p.tolerance_mm === undefined) return 'unjudged';
  if (p.reachable === false) return 'coarse';
  if (p.in_position) return 'in_position';
  if (p.observation === 'outside_tolerance_observed') return 'guiding';
  const within = p.position_observation === 'within' && (p.rotation_observation === 'within' || p.rotation_observation === null || p.rotation_observation === undefined);
  return within && (p.doubtful || []).length ? 'doubt' : 'close';
}

// How far the placard is from where it may be, in units of what is allowed:
// 1 is the limit of "in position" (tolerance minus the uncertainty of this view).
export function ratio(p) {
  let q = 0;
  const tol = p.tolerance_mm, U = p.expanded_mm || 0;
  if (tol && p.radial_mm !== null && p.radial_mm !== undefined) q = Math.max(q, p.radial_mm / Math.max(tol - U, 0.15 * tol));
  const tr = p.tolerance_rotation_deg, ur = p.rotation_expanded_deg || 0;
  if (tr && p.rotation_deg !== null && p.rotation_deg !== undefined) q = Math.max(q, Math.abs(p.rotation_deg) / Math.max(tr - ur, 0.15 * tr));
  return q;
}

// Gauge scale: deviation / tolerance -> fraction of the gauge radius.  Linear and
// generous up to 1.5 tolerances (the last millimetres are what the gauge is
// for), logarithmic beyond, so that a placard still far away is on the dial too.
export function gaugeMap(q) {
  const x = Math.max(q, 0);
  if (x <= 1.5) return 0.42 * x;
  return Math.min(1, 0.63 + 0.37 * Math.log(x / 1.5) / Math.log(40 / 1.5));
}

function drawGauge(cv, m) {
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const W = cv.clientWidth || 132, Hh = cv.clientHeight || 166;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(Hh * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, Hh);
  const cx = W / 2, cy = W / 2, R = W / 2 - 4;
  const col = COLOR[m.cls] || COLOR.idle;
  const ring = (r, style, width, dash) => { g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.setLineDash(dash || []); g.strokeStyle = style; g.lineWidth = width; g.stroke(); g.setLineDash([]); };
  // dial
  g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fillStyle = '#0b0f16'; g.fill();
  ring(R, 'rgba(255,255,255,0.28)', 1.5);
  ring(gaugeMap(5) * R, 'rgba(255,255,255,0.12)', 1, [2, 4]);
  ring(gaugeMap(20) * R, 'rgba(255,255,255,0.12)', 1, [2, 4]);
  g.strokeStyle = 'rgba(255,255,255,0.2)'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(cx - R, cy); g.lineTo(cx + R, cy); g.moveTo(cx, cy - R); g.lineTo(cx, cy + R); g.stroke();
  // the tolerance of the record
  if (m.tol) {
    const rt = gaugeMap(1) * R;
    if (m.cls === 'ok') { g.beginPath(); g.arc(cx, cy, rt, 0, Math.PI * 2); g.fillStyle = 'rgba(61,220,151,0.22)'; g.fill(); }
    ring(rt, m.cls === 'ok' ? COLOR.ok : '#f3f6fa', 2.4);
  }
  // the placard's centre, as large as the uncertainty of this view
  if (m.radial !== null) {
    const unit = m.tol || 1;
    const q = m.radial / unit, u = (m.U || 0) / unit;
    const rr = gaugeMap(q) * R;
    const x = cx + m.dir[0] * rr, y = cy + m.dir[1] * rr;
    const ru = Math.max(((gaugeMap(q + u) - gaugeMap(Math.max(q - u, 0))) / 2) * R, 0);
    if (ru > 5.5) { g.beginPath(); g.arc(x, y, ru, 0, Math.PI * 2); g.fillStyle = `${col}3d`; g.fill(); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke(); }
    g.beginPath(); g.arc(x, y, 5.5, 0, Math.PI * 2); g.fillStyle = col; g.fill(); g.lineWidth = 2; g.strokeStyle = '#05070a'; g.stroke();
  }
  // level for the rotation: bubble right = turned clockwise on the screen
  const ly = W + 17, x0 = 10, x1 = W - 10, L = (x1 - x0) / 2, mx = (x0 + x1) / 2;
  g.lineCap = 'round';
  g.strokeStyle = 'rgba(255,255,255,0.22)'; g.lineWidth = 12; g.beginPath(); g.moveTo(x0, ly); g.lineTo(x1, ly); g.stroke();
  g.lineCap = 'butt';
  if (m.tolRot) {
    const b = gaugeMap(1) * L;
    g.strokeStyle = m.rotOk ? COLOR.ok : '#f3f6fa'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(mx - b, ly - 10); g.lineTo(mx - b, ly + 10); g.moveTo(mx + b, ly - 10); g.lineTo(mx + b, ly + 10); g.stroke();
  } else {
    g.strokeStyle = 'rgba(255,255,255,0.4)'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(mx, ly - 9); g.lineTo(mx, ly + 9); g.stroke();
  }
  if (m.rot !== null) {
    const unit = m.tolRot || 1;
    const bx = mx + Math.sign(m.rot) * gaugeMap(Math.abs(m.rot) / unit) * L;
    g.beginPath(); g.arc(bx, ly, 6, 0, Math.PI * 2); g.fillStyle = m.rotOk ? COLOR.ok : col; g.fill(); g.lineWidth = 2; g.strokeStyle = '#05070a'; g.stroke();
  }
}

const mean = (pts) => [pts.reduce((a, c) => a + c[0], 0) / pts.length, pts.reduce((a, c) => a + c[1], 0) / pts.length];

export function createPlacement({ viewer, split, onRecorded = () => {} }) {
  const sid = state.session.id;
  const fb = new Feedback();
  let live = null;
  let phase = 'starting';
  let last = null;                 // the latest live answer
  let hint = null, hintAt = 0;
  let shift = null, shiftAt = 0;
  let streak = 0, streakSince = 0;
  let armed = true;                // automatic record allowed (disarmed after an attempt until the placard has moved, or a while has passed)
  let attemptAt = 0;
  let recording = false;
  let destroyed = false;
  let notice = '';                 // what the last record attempt said, shown until the state changes
  let sound = prefs.placeSound !== false;
  let auto = prefs.placeAuto !== false;
  const stats = { results: 0, in_position: 0, records: 0, attempts: 0, phases: [] };
  fb.setMuted(!sound);

  // ------------------------------------------------------------------ panel
  const gauge = h('canvas', { class: 'gauge', 'aria-hidden': 'true' });
  const title = h('div', { class: 'placing-title', role: 'status', 'aria-live': 'polite' }, PHASE.starting.title);
  const arrow = h('span', { class: 'placing-arrow', 'aria-hidden': 'true' }, icon('go_right'));
  const moveNum = h('b', null, '—');
  const turnIc = h('span', { class: 'placing-turn', 'aria-hidden': 'true' }, icon('refresh'));
  const turnNum = h('b', null, '—');
  const nums = h('div', { class: 'placing-nums', 'data-region': 'placement-numbers' },
    h('span', { class: 'num' }, arrow, moveNum, h('i', null, 'mm')),
    h('span', { class: 'num' }, turnIc, turnNum, h('i', null, '°')));
  const msg = h('div', { class: 'placing-msg', 'data-region': 'placement-message' }, 'Aim the phone at the tags.');
  const meta = h('div', { class: 'placing-meta', 'data-region': 'placement-meta' });
  const flags = h('ul', { class: 'checklist placing-flags' });
  const soundBtn = h('button', { class: 'btn sm grow toggle', type: 'button', 'data-action': 'place-sound', 'aria-pressed': String(sound), onclick: toggleSound });
  const autoBtn = h('button', { class: 'btn sm grow toggle', type: 'button', 'data-action': 'place-auto', 'aria-pressed': String(auto), onclick: toggleAuto });
  const recBtn = h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'place-record', disabled: true, onclick: () => record('manual') }, icon('camera'), 'Record now');
  const el = h('div', { class: 'placing', 'data-region': 'placement', 'data-phase': phase },
    h('div', { class: 'placing-main' }, gauge, h('div', { class: 'placing-say' }, title, nums, msg)),
    meta, flags,
    h('div', { class: 'row wrap' }, soundBtn, autoBtn, recBtn));
  const flash = h('div', { class: 'place-flash', 'aria-hidden': 'true' });
  split.append(flash);

  function labelButtons() {
    soundBtn.replaceChildren(icon(sound ? 'sound' : 'sound_off'), sound ? 'Sound on' : 'Sound off');
    soundBtn.setAttribute('aria-pressed', String(sound));
    autoBtn.replaceChildren(icon('check'), auto ? 'Records by itself' : 'Record by hand');
    autoBtn.setAttribute('aria-pressed', String(auto));
  }
  function toggleSound() {
    sound = !sound; prefs.placeSound = sound;
    if (sound) fb.unlock();
    fb.setMuted(!sound);
    labelButtons();
  }
  function toggleAuto() { auto = !auto; prefs.placeAuto = auto; streak = 0; armed = true; labelButtons(); }
  labelButtons();

  function doFlash(kind) {
    flash.className = 'place-flash';
    void flash.offsetWidth;                     // restart the animation
    flash.className = `place-flash go ${kind}`;
  }

  // ---------------------------------------------------------------- drawing
  function overlay(p, cls) {
    const prims = [];
    if (!p || !p.found || !p.corners_px) { viewer.setExtra([]); return; }
    const col = COLOR[cls] || COLOR.idle;
    const c = p.corners_px;
    if (c.length === 4 && c.every((x) => x.length === 2)) {
      prims.push({ kind: 'placard_live', points: c, color: col, label: cls === 'ok' ? 'in position' : '', placed: (p.sides || []).map((x) => x === 'record_size') });
      if (p.target_px && p.move) {
        const lbl = p.radial_mm === null || p.radial_mm === undefined ? '' : `${fmt(p.radial_mm, 1)} mm`;
        prims.push({ kind: 'move_arrow', points: [mean(c), mean(p.target_px)], color: col, label: cls === 'ok' ? '' : lbl });
      }
    } else {
      // part of the outline: the sides that were measured
      for (let k = 0; k < 4; k++) { const a = c[k], b = c[(k + 1) % 4]; if (a && b && a.length === 2 && b.length === 2) prims.push({ kind: 'segment', points: [a, b], color: col, width: 3 }); }
    }
    viewer.setExtra(prims);
  }

  function render(p) {
    const ph = PHASE[phase] || PHASE.lost;
    el.dataset.phase = phase;
    el.className = `placing st-${ph.cls}`;
    split.dataset.placing = ph.cls;
    if (title.textContent !== ph.title) title.textContent = ph.title;
    const found = !!(p && p.found && p.complete);
    const dir = found && p.move && p.move.px ? p.move.px : null;
    const len = dir ? Math.hypot(dir[0], dir[1]) : 0;
    // the arrow points the way the placard has to go, as seen on the screen
    arrow.style.transform = len > 1e-6 ? `rotate(${(Math.atan2(dir[1], dir[0]) * 180 / Math.PI).toFixed(1)}deg)` : '';
    // an arrow or a turn sign is an instruction: none is shown for what is already inside its tolerance
    const posOk = !!(p && p.position_observation === 'within'), rotOk = !!(p && p.rotation_observation === 'within');
    arrow.style.visibility = found && !posOk && phase !== 'recorded' && len > 1e-6 ? 'visible' : 'hidden';
    moveNum.textContent = found && p.radial_mm !== null && p.radial_mm !== undefined ? fmt(p.radial_mm, p.radial_mm < 10 ? 1 : 0) : '—';
    const turn = found && p.move ? p.move.turn_screen_deg : null;
    turnNum.textContent = turn === null || turn === undefined ? '—' : fmt(Math.abs(turn), 1);
    turnIc.classList.toggle('ccw', (turn || 0) < 0);
    turnIc.style.visibility = turn !== null && turn !== undefined && !rotOk && phase !== 'recorded' && Math.abs(turn) >= 0.05 ? 'visible' : 'hidden';
    nums.style.visibility = found ? 'visible' : 'hidden';
    const text = notice || (p && p.message) || (last && last.tracking && last.tracking.message) || '';
    if (msg.textContent !== text) msg.textContent = text;
    // what this view is worth
    clear(meta);
    if (p && p.status) meta.append(chip((STATUS_TEXT[p.status] || STATUS_TEXT.blocked).title, p.status === 'prototype_guidance' || p.status === 'validated' ? 'ok' : p.status === 'blocked' ? 'danger' : 'warn'));
    // of the placard's position when one is measured, else of the target in this view
    const U = p ? (p.expanded_mm !== null && p.expanded_mm !== undefined ? p.expanded_mm : p.target_expanded_mm) : null;
    if (U !== null && U !== undefined) meta.append(chip(`uncertainty ±${fmt(U, 2)} mm`, ''));
    if (p && p.tolerance_mm !== null && p.tolerance_mm !== undefined) meta.append(chip(`tolerance ±${fmt(p.tolerance_mm, 1)} mm`, ''));
    if (last && last.simulated) meta.append(chip('simulated demo data', 'info'));
    clear(flags);
    ((p && p.flags) || []).filter((f) => f !== 'registration_preview_only').slice(0, 3).forEach((f) => flags.append(h('li', { class: 'todo', 'data-flag': f }, icon('alert'), h('span', null, flagText(f)))));
    // the uncertainty of this view leaves little of the tolerance: say so, and what limits it
    if (p && p.tight && phase !== 'in_position' && phase !== 'recorded') {
      flags.append(h('li', { class: 'todo', 'data-flag': 'tight' }, icon('info'), h('span', null,
        `It counts as in position only within ±${fmt(p.room_mm, 1)} mm of the target: the uncertainty of this view takes the rest of the ±${fmt(p.tolerance_mm, 1)} mm tolerance. ${p.limit_hint || ''}`)));
    }
    recBtn.disabled = !(p && p.found) || recording;
    drawGauge(gauge, {
      cls: ph.cls, tol: p ? p.tolerance_mm : null, U: p ? p.expanded_mm : null,
      radial: found && p.radial_mm !== null && p.radial_mm !== undefined ? p.radial_mm : null,
      // on the dial the disc sits where the placard is, relative to the target, as on the screen
      dir: len > 1e-6 ? [-dir[0] / len, -dir[1] / len] : [0, 0],
      rot: turn === null || turn === undefined ? null : -turn, tolRot: p ? p.tolerance_rotation_deg : null,
      rotOk: !!(p && p.rotation_observation === 'within'),
    });
    overlay(p, ph.cls);
  }

  function setPhase(next, p) {
    const prev = phase;
    phase = next;
    if (next !== prev) {
      stats.phases.push(next);
      if (stats.phases.length > 60) stats.phases.shift();
      if (next === 'in_position') { fb.chime('in'); doFlash('in'); }
      else if (prev === 'in_position' && (next === 'guiding' || next === 'close')) fb.chime('out');
    }
    if (next === 'in_position') fb.hold();
    else if ((next === 'guiding' || next === 'close') && p) fb.guide(ratio(p));
    else if (next !== 'recording' && next !== 'recorded') fb.silence();
    render(p);
  }

  // ------------------------------------------------------------- live loop
  function onResult(res) {
    if (destroyed || recording) return;
    last = res;
    stats.results += 1;
    const now = performance.now();
    const t = res.tracking || {};
    const tracking = t.state === 'tracking';
    const p = tracking ? res.placement : null;
    const es = tracking ? t.edge_shift : null;
    if (es && es.source === 'this_frame') { shift = { delta_px: es.delta_px, sigma_px: es.sigma_px, n: es.n, for: es.for }; shiftAt = now; }
    else if (es && es.source !== 'stream') shift = null;      // these tags are too small to measure it: nothing of the stream's to hand back
    hint = p && p.found && p.complete ? { corners_plane: p.corners_plane, polarity: p.polarity } : null;
    hintAt = now;
    const next = phaseOf(res);
    if (next !== phase) notice = '';
    if (next === 'in_position') stats.in_position += 1;
    // the record: in position with nothing covering the placard, several frames in a row
    if (next === 'in_position' && p.clean) { if (!streak) streakSince = now; streak += 1; } else streak = 0;
    if (next !== 'in_position' || now - attemptAt > REARM_MS) armed = true;
    setPhase(next, p);
    if (auto && armed && streak >= NEED_IN_A_ROW && now - streakSince >= NEED_HELD_MS) record('auto');
  }

  // No current reading (tracking lost, no answer, frozen picture): nothing may keep sounding or showing "in position".
  function onGap(text) {
    if (destroyed || recording) return;
    streak = 0;
    last = { tracking: { state: 'lost', message: text || 'No current reading.' } };
    notice = '';
    setPhase('lost', null);
  }

  // What the next frame's request carries: where the placard was, and the stream's edge shift.
  function nextMeta() {
    const now = performance.now();
    const m = {};
    if (hint && now - hintAt < HINT_MAX_AGE_MS) m.placement_hint = hint;
    if (shift && now - shiftAt < SHIFT_REFRESH_MS) m.edge_shift = shift;
    return m;
  }

  // ----------------------------------------------------------------- record
  async function record(how) {
    if (recording || destroyed || !live) return;
    const pid = state.session.active_placard_id;
    recording = true; streak = 0; armed = false;
    let kept = false;                // a record was stored: the loop stays held until the Place step takes over
    stats.attempts += 1;
    live.hold(true);
    notice = '';
    setPhase('recording', last && last.placement);
    msg.textContent = 'Taking the record photo: keep everything still.';
    let fid = null;
    try {
      const shot = await live.camera.grab({ quality: 0.95 });
      const up = await upload(`api/sessions/${sid}/frames?role=verification&source=camera_still`, shot.blob, { meta: { ...shot.meta, capture_mode: 'live_placement' }, timeout: 60000 });
      fid = up.frame.id;
      const out = await api('POST', `api/sessions/${sid}/verify`, { frame_id: fid, placard_id: pid, snap: true }, { timeout: 120000 });
      const v = (out.session.verifications || []).find((x) => x.frame_id === fid && x.placard_id === pid) || null;
      if (destroyed) { setSession(out.session); return; }     // the view was closed meanwhile: the photo and its result are stored all the same
      if (v && v.observation === 'within_tolerance_observed') {
        setSession(out.session);
        stats.records += 1;
        fb.silence(); fb.chime('recorded'); doFlash('recorded');
        phase = 'recorded'; kept = true;
        stats.phases.push('recorded');
        notice = `Recorded: ${fmt(v.radial_mm, 2)} mm from the target, ${fmt(Math.abs(v.rotation_rad) * 180 / Math.PI, 2)}° — observed within the tolerance.`;
        render(last && last.placement);
        onRecorded(v, how);
        return;
      }
      if (how === 'auto') {
        // the full-size photo did not confirm it: that photo is not kept
        setSession(await api('DELETE', `api/sessions/${sid}/frames/${fid}`));
        fb.chime('fail');
        notice = `The record photo did not confirm it${v ? ` (${v.observation.replace(/_/g, ' ')})` : ''}. Keep adjusting.`;
      } else {
        setSession(out.session);
        fb.chime(v ? 'out' : 'fail');
        notice = v ? `Recorded as it is: ${v.observation.replace(/_/g, ' ')}. See Verify for the numbers.` : (out.message || 'The photo could not be measured.');
        if (v) { kept = true; onRecorded(v, how); }
      }
    } catch (err) {
      fb.chime('fail');
      notice = (err && err.lumen && err.lumen.message) || 'The record could not be taken.';
      if (fid && how === 'auto') { try { setSession(await api('DELETE', `api/sessions/${sid}/frames/${fid}`)); } catch (_) { /* it stays; the Verify step shows it */ } }
    } finally {
      recording = false;
      attemptAt = performance.now();
      if (!destroyed && !kept) {
        if (live) live.hold(false);
        // until the next frame is measured there is no current reading; what the attempt said stays on screen
        const said = notice;
        setPhase('lost', null);
        notice = said; msg.textContent = said;
      }
    }
  }

  // Another placard was chosen: where the last one was says nothing about this one.
  function reset() {
    hint = null; streak = 0; armed = true; notice = '';
  }

  function destroy() {
    destroyed = true;
    fb.close();
    viewer.setExtra([]);
    flash.remove();
    delete split.dataset.placing;
    if (window.__lumenPlacement && window.__lumenPlacement.el === el) window.__lumenPlacement = null;
    window.__lumenPlacementLast = { stats, log: fb.log.map((e) => e.kind) };     // for the browser tests
  }

  render(null);
  const self = {
    el,
    unlock: () => { if (sound) fb.unlock(); },
    attach: (lv) => { live = lv; },
    meta: nextMeta, onResult, onGap, record, reset, destroy,
    get phase() { return phase; },
    get recording() { return recording; },
    // for the browser tests
    stats, sound: fb, get last() { return last; }, get auto() { return auto; },
  };
  window.__lumenPlacement = self;
  return self;
}

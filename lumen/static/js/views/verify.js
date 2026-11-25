// Step 7 — verification after installation: where is the placard, compared with the target?
//
// The result is measurement evidence with an uncertainty.  It is never an
// acceptance decision, and the screen says so wherever a number is shown.
import { api, upload } from '../api.js';
import { h, clear, icon, banner, chip, toast, showError, busy, fmt } from '../dom.js';
import { state, setSession, refreshSession, activePlacard, lengthOut } from '../store.js';
import { Viewer } from '../viewer.js';
import { showFrame, viewerShell, zoomButtons, maximizeButton, cell, budgetTable } from './common.js';
import { openCamera } from './camera_sheet.js';

const OBS = {
  within_tolerance_observed: { title: 'Observed within tolerance', cls: 'prototype_guidance', ic: 'check', sub: 'The measured deviation plus its uncertainty is inside the tolerance of the record.' },
  outside_tolerance_observed: { title: 'Observed outside tolerance', cls: 'blocked', ic: 'block', sub: 'The measured deviation exceeds the tolerance by more than its uncertainty.' },
  indeterminate: { title: 'Indeterminate', cls: 'research_preview', ic: 'alert', sub: 'The measurement cannot say whether the placard is inside the tolerance.' },
  partial: { title: 'Partial outline only', cls: 'research_preview', ic: 'alert', sub: 'Only part of the placard outline could be measured; the centre is not reported.' },
  no_evidence: { title: 'No measurement', cls: 'blocked', ic: 'block', sub: 'This photo gives no evidence of the placard position.' },
};

const SIDE = { top: 'top edge', right: 'right edge', bottom: 'bottom edge', left: 'left edge' };
const CORNER = { TL: 'top-left', TR: 'top-right', BR: 'bottom-right', BL: 'bottom-left' };
export function flagText(flag) {
  const [code, arg] = flag.split(':');
  const where = SIDE[arg] || (CORNER[arg] ? `${CORNER[arg]} corner` : '');
  const T = {
    marker_moved: 'A registered tag has moved since capture: the photo cannot locate the placard.',
    registration_blocked: 'The target could not be established in this photo.',
    registration_preview_only: 'The registration does not meet every gate: treat the numbers as a preview.',
    placard_not_found: 'No placard outline was found near the target.',
    ambiguous_proposal: 'Two different outlines fit about equally well: check that the green outline is the placard.',
    at_search_limit: 'The outline was found at the edge of the search area: check it.',
    size_differs_from_record: 'The measured size differs from the record. Is this the right part?',
    corner_lifted_or_hidden: `The ${where} is folded, lifted or covered.`,
    corner_not_directly_seen: `The ${where} is not directly visible; it was located from the two sides.`,
    corner_outside_image: `The ${where} is outside the photo.`,
    edge_partly_hidden: `Part of the ${where} is hidden or unclear.`,
    edge_not_straight: `The ${where} is not straight (curled sticker, or lens distortion).`,
    glare_on_edge: `Glare on the ${where}.`,
    edge_fit_fallback: `The ${where} was located less precisely.`,
    side_not_measured: `The ${where} could not be measured.`,
    partial_perimeter: 'Only three sides were measured.',
    manual_side: `The ${where} is as marked by hand.`,
  };
  return T[code] || flag.replace(/_/g, ' ');
}

export function mount(root, ctx) {
  const s = () => state.session;
  const shell = viewerShell({ label: 'Verification' });
  const panel = h('div', { class: 'panel' });
  const split = h('div', { class: 'split' }, shell.wrap, panel);
  root.append(split);
  const viewer = new Viewer(shell.wrap, { label: 'Verification photo with the target outline and the measured placard outline', onTap: onTap });
  window.__lumenViewer = viewer;
  shell.tools.append(maximizeButton(viewer, split), ...zoomButtons(viewer));
  const pill = h('span', { class: 'pill frozen', 'data-mode': 'frozen' }, 'Verification photo');
  shell.hud.append(pill);
  const pick = h('input', { type: 'file', accept: 'image/*', hidden: true, 'data-input': 'verify-photo', onchange: (ev) => { addPhoto(ev.target.files[0]); ev.target.value = ''; } });
  const pickNative = h('input', { type: 'file', accept: 'image/*', capture: 'environment', hidden: true, 'data-input': 'verify-native', onchange: (ev) => { addPhoto(ev.target.files[0]); ev.target.value = ''; } });
  root.append(pick, pickNative);
  let destroyed = false;
  let frameId = null;
  let marking = false;
  let taps = [];
  let detail = null;      // detail of the latest measurement in this view (not stored client-side otherwise)
  let message = '';

  const vframes = () => s().frames.filter((f) => f.role === 'verification');
  const current = () => (s().verifications || []).find((v) => v.frame_id === frameId && v.placard_id === s().active_placard_id) || null;

  async function addPhoto(file) {
    if (!file) return;
    const b = busy('Uploading the verification photo…');
    try {
      let blob = file;
      if (!/^image\/(jpeg|png|webp|bmp)$/.test(file.type)) {
        const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
        const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
        c.getContext('2d').drawImage(bmp, 0, 0);
        blob = await new Promise((ok, bad) => c.toBlob((x) => (x ? ok(x) : bad(new Error('encode failed'))), 'image/jpeg', 0.95));
      }
      const out = await upload(`api/sessions/${s().id}/frames?role=verification&source=photo_upload`, blob, { meta: { captured_at: new Date().toISOString() }, onProgress: (p) => b.update(`Uploading… ${Math.round(p * 100)} %`) });
      await refreshSession();
      b.done();
      await measure(out.frame.id);
    } catch (err) { showError(err, 'The photo could not be used'); } finally { b.done(); }
  }

  async function measure(fid, corners = null, snap = true) {
    frameId = fid;
    const b = busy(corners ? 'Measuring from the marked corners…' : 'Locating the placard…');
    try {
      const body = { frame_id: fid, placard_id: s().active_placard_id, snap };
      if (corners) body.corners_px = corners;
      const out = await api('POST', `api/sessions/${s().id}/verify`, body, { timeout: 120000 });
      setSession(out.session);
      detail = out.detail; message = out.message || '';
    } catch (err) {
      detail = null; message = '';
      showError(err, 'Verification failed');
    } finally { b.done(); }
    marking = false; taps = [];
    viewer.setExtra([]);
    await show();
  }

  async function show() {
    if (destroyed) return;
    const frames = vframes();
    if (!frames.length) { frameId = null; renderEmpty(); return; }
    if (!frameId || !frames.some((f) => f.id === frameId)) frameId = frames[frames.length - 1].id;
    // the picture area must be laid out before anything is centred in it
    split.classList.remove('max-panel');
    renderPanel();
    try {
      viewer.userMoved = false;
      const out = await showFrame(viewer, frameId, 'image', { uncertainty: false, placards: [s().active_placard_id] });
      if (destroyed) return;
      requestAnimationFrame(() => { viewer._resize(); focusOn(out.overlay); });
    } catch (err) { showError(err, 'Could not load the verification photo'); }
  }

  function focusOn(overlay) {
    const t = (overlay.primitives || []).find((p) => p.kind === 'measured' && p.corners) || (overlay.primitives || []).find((p) => p.kind.startsWith('target') && p.corners);
    if (!t) return;
    const cx = t.corners.reduce((a, c) => a + c[0], 0) / 4, cy = t.corners.reduce((a, c) => a + c[1], 0) / 4;
    viewer.fit(); viewer.centerOn(cx, cy, 1.0);
    const r = viewer._box();
    const wpx = Math.hypot(t.corners[1][0] - t.corners[0][0], t.corners[1][1] - t.corners[0][1]) * viewer.t.s;
    if (wpx < r.width * 0.4) viewer.zoomAt(r.width / 2, r.height / 2, (r.width * 0.5) / Math.max(wpx, 1));
  }

  async function useDemoPhoto() {
    const b = busy('Loading the simulated verification photo…');
    try {
      const out = await api('POST', `api/sessions/${s().id}/demo/verification`, undefined, { timeout: 120000 });
      setSession(out.session);
      b.done();
      await measure(out.frame.id);
    } catch (err) { showError(err, 'No demo photo'); } finally { b.done(); }
  }

  function captureButtons() {
    return h('div', { class: 'stack' },
      s().mode === 'demo' ? h('button', { class: 'btn primary lg block', type: 'button', 'data-action': 'verify-demo', onclick: useDemoPhoto }, icon('play'), 'Use the simulated photo of the installed placard') : null,
      h('button', { class: `btn ${s().mode === 'demo' ? '' : 'primary'} lg block`, type: 'button', 'data-action': 'verify-camera', onclick: () => openCamera({ role: 'verification', single: true, onFallback: () => pickNative.click(), onCaptured: (f) => measure(f.id) }) }, icon('camera'), 'Open camera'),
      h('div', { class: 'row' },
        h('button', { class: 'btn grow', type: 'button', 'data-action': 'verify-native', onclick: () => pickNative.click() }, icon('camera'), 'Full-res photo'),
        h('button', { class: 'btn grow', type: 'button', 'data-action': 'verify-choose', onclick: () => pick.click() }, icon('upload'), 'Choose photo')));
  }

  function renderEmpty() {
    clear(panel);
    split.classList.add('max-panel');
    pill.textContent = 'No verification photo yet';
    const p = activePlacard(s());
    panel.append(
      h('h1', { class: 'view-title' }, 'Verify the placement'),
      h('p', { class: 'view-lead' }, `After installing ${p ? p.placard_id : 'the placard'}, take one new photo of it. The tags must still be exactly where they were during capture.`),
      h('ul', { class: 'checklist' },
        h('li', null, icon('eye'), h('span', null, 'The whole placard and the tags are in the picture.')),
        h('li', null, icon('eye'), h('span', null, 'No finger, tape or glare on the placard’s edges.')),
        h('li', null, icon('eye'), h('span', null, 'Roughly face the panel; fill the frame as far as the tags allow.'))),
      captureButtons(),
      banner('info', 'What you will get', 'Signed deviations of the installed placard from the target, with their uncertainty. This is measurement evidence for research — not an acceptance decision.'));
    ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('place') }]);
  }

  function dev(label, v, unit = 'mm', digits = 2) {
    return cell(label, v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${fmt(v, digits)}`, unit);
  }

  function renderPanel() {
    clear(panel);
    split.classList.remove('max-panel');
    const sess = s();
    const v = current();
    const frames = vframes();
    pill.textContent = marking ? `Tap the corners (${taps.length}/4)` : `Verification photo${frames.length > 1 ? ` ${frames.findIndex((f) => f.id === frameId) + 1}/${frames.length}` : ''}`;
    pill.className = `pill ${marking ? 'warn' : 'frozen'}`;
    if (marking) {
      panel.append(
        banner('info', 'Mark the four corners of the placard', 'Tap each corner once, in any order. Pinch to zoom in first for accuracy.'),
        h('div', { class: 'row wrap' },
          h('button', { class: 'btn primary grow', type: 'button', 'data-action': 'manual-snap', disabled: taps.length === 4 ? null : true, onclick: () => measure(frameId, taps, true) }, icon('target'), 'Measure (snap to edges)'),
          h('button', { class: 'btn grow', type: 'button', 'data-action': 'manual-exact', disabled: taps.length === 4 ? null : true, onclick: () => measure(frameId, taps, false) }, 'Use exactly as marked')),
        h('div', { class: 'row' },
          h('button', { class: 'btn grow', type: 'button', 'data-action': 'manual-clear', onclick: () => { taps = []; drawTaps(); renderPanel(); } }, 'Clear'),
          h('button', { class: 'btn grow', type: 'button', onclick: () => { marking = false; taps = []; drawTaps(); renderPanel(); } }, 'Cancel')));
      ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('place') }]);
      return;
    }
    if (!v) {
      panel.append(banner('warn', 'This photo has not been measured yet.', null),
        h('button', { class: 'btn primary block', type: 'button', 'data-action': 'verify-measure', onclick: () => measure(frameId) }, icon('target'), 'Locate the placard'));
    } else {
      const o = OBS[v.observation] || OBS.indeterminate;
      panel.append(h('div', { class: `status-hero ${o.cls}`, role: 'status', 'data-observation': v.observation },
        icon(o.ic), h('div', { class: 'grow' }, h('div', { class: 'big' }, o.title), h('div', { class: 'small' }, `${o.sub} Measurement evidence only — not an acceptance decision.${sess.mode === 'demo' ? ' SIMULATED demo data.' : ''}`))));
      if (message) panel.append(banner('warn', message));
      const U = v.uncertainty ? v.uncertainty.expanded_mm : null;
      const tol = (activePlacard(sess) || {}).tol_position_mm;
      if (v.dx_mm !== null && v.dx_mm !== undefined) {
        panel.append(
          h('div', { class: 'eyebrow' }, 'Measured minus target (local frame)'),
          h('div', { class: 'readout', 'data-region': 'deviations' },
            dev('Δx', state.units === 'in' ? v.dx_mm / 25.4 : v.dx_mm, state.units, state.units === 'in' ? 3 : 2),
            dev('Δz', state.units === 'in' ? v.dz_mm / 25.4 : v.dz_mm, state.units, state.units === 'in' ? 3 : 2),
            cell('Distance', lengthOut(v.radial_mm, 2).split(' ')[0], state.units),
            dev('Rotation', (v.rotation_rad * 180) / Math.PI, '°', 2),
            dev('Width', v.width_error_mm, 'mm'), dev('Height', v.height_error_mm, 'mm')),
          h('p', { class: 'hint' }, `Expanded uncertainty of the position: ${U === null ? 'not available' : `±${fmt(U, 2)} mm (k=${v.uncertainty.k})`}. ${tol === null || tol === undefined ? 'The record gives no position tolerance.' : `Record tolerance: ±${fmt(tol, 2)} mm.`} Positive Δx is toward +x, positive Δz toward +z.`));
      }
      const e = v.edge_deviation_mm || {};
      if (Object.keys(e).length) {
        panel.append(h('details', { class: 'fold', open: v.observation === 'partial' ? true : null }, h('summary', null, 'Offsets of each edge'),
          h('div', { class: 'body' }, h('div', { class: 'readout' }, ['left', 'right', 'bottom', 'top'].filter((k) => k in e).map((k) => dev(k[0].toUpperCase() + k.slice(1), e[k]))),
            h('p', { class: 'hint' }, 'Left/right: positive toward the placard’s right. Bottom/top: positive toward its top.'))));
      }
      if (v.flags.length) {
        panel.append(h('div', { class: 'card tight', 'data-region': 'flags' }, h('h3', null, 'Check before relying on this'), h('ul', { class: 'checklist' }, v.flags.map((f) => h('li', { class: 'todo', 'data-flag': f }, icon('alert'), h('span', null, flagText(f)))))));
      }
      if (v.uncertainty) panel.append(h('details', { class: 'fold' }, h('summary', null, 'Uncertainty budget'), h('div', { class: 'body' }, budgetTable(v.uncertainty))));
      panel.append(h('div', { class: 'legend' },
        h('span', null, h('i', { style: { background: 'var(--ov-target)' } }), 'target'),
        h('span', null, h('i', { style: { background: '#3ddc97' } }), 'measured outline')));
      if (v.confirmed) panel.append(h('div', null, chip('outline confirmed by the operator', 'ok')));
    }
    panel.append(
      h('div', { class: 'eyebrow' }, 'If the green outline is not the placard'),
      h('div', { class: 'row wrap' },
        h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'verify-manual', onclick: () => { marking = true; taps = []; viewer.setExtra([]); renderPanel(); } }, icon('line'), 'Mark corners by hand'),
        h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'verify-retake', onclick: () => openCamera({ role: 'verification', single: true, onFallback: () => pickNative.click(), onCaptured: (f) => measure(f.id) }) }, icon('camera'), 'New photo'),
        h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'verify-upload', onclick: () => pick.click() }, icon('upload'), 'Upload')));
    if (frames.length > 1) {
      const idx = frames.findIndex((f) => f.id === frameId);
      panel.append(h('div', { class: 'row' },
        h('button', { class: 'btn sm', type: 'button', 'aria-label': 'Previous verification photo', onclick: () => { frameId = frames[(idx - 1 + frames.length) % frames.length].id; detail = null; message = ''; show(); } }, '‹'),
        h('span', { class: 'grow hint', style: { textAlign: 'center' } }, `Photo ${idx + 1} of ${frames.length}`),
        h('button', { class: 'btn sm', type: 'button', 'aria-label': 'Next verification photo', onclick: () => { frameId = frames[(idx + 1) % frames.length].id; detail = null; message = ''; show(); } }, '›')));
    }
    panel.append(
      h('div', { class: 'row wrap' },
        h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'verify-export', onclick: exportSummary }, icon('download'), 'Export summary'),
        h('button', { class: 'btn sm grow danger', type: 'button', 'data-action': 'verify-delete-photo', onclick: removePhoto }, icon('trash'), 'Delete this photo')),
      h('p', { class: 'hint' }, 'Photos stay on the server only until the session ends or you delete it (menu → End session). The exported summary contains numbers and annotations, never images.'));
    const canConfirm = v && !['no_evidence'].includes(v.observation);
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('place') },
      v && v.confirmed
        ? { id: 'unconfirm', label: 'Confirmed — undo', kind: 'lg', icon: 'check', onClick: () => confirm(false) }
        : { id: 'confirm', label: 'Confirm measured outline', kind: 'primary lg', icon: 'check', disabled: !canConfirm, onClick: () => confirm(true) },
    ]);
  }

  function drawTaps() {
    viewer.setExtra(taps.map((p, i) => ({ kind: 'handle', points: [p], label: String(i + 1), color: '#3ddc97' })));
  }
  function onTap(p) {
    if (!marking || taps.length >= 4) return;
    taps.push([p[0], p[1]]);
    drawTaps(); renderPanel();
  }

  async function confirm(on) {
    const v = current();
    if (!v) return;
    try { setSession(await api('POST', `api/sessions/${s().id}/verifications/${v.id}/confirm`, { confirmed: on })); renderPanel(); if (on) toast({ title: 'Outline confirmed', detail: 'Recorded as measurement evidence, not as acceptance.', kind: 'ok' }); } catch (err) { showError(err); }
  }
  async function removePhoto() {
    try { setSession(await api('DELETE', `api/sessions/${s().id}/frames/${frameId}`)); frameId = null; detail = null; message = ''; await show(); } catch (err) { showError(err); }
  }
  async function exportSummary() {
    try {
      const data = await api('GET', `api/sessions/${s().id}/export`);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const a = h('a', { href: URL.createObjectURL(blob), download: `lumen-session-${s().id.slice(0, 8)}.json` });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 20000);
      toast({ title: 'Summary exported', detail: 'Numbers and annotations only; photos are not included.', kind: 'ok' });
    } catch (err) { showError(err); }
  }

  ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('place') }]);
  show();
  return {
    flush: true,
    redraw: () => viewer.draw(),
    destroy() { destroyed = true; viewer.destroy(); window.__lumenViewer = null; },
  };
}

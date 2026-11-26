// Step 6 — placing the placard.
//
// The target is first shown on a photo.  The work itself is done in live
// placement: the camera watches while the placard is positioned, the screen and
// the sound say which way to move it, and when it is in position a full-size
// photo is stored and measured as the record (views/placement.js).
import { api } from '../api.js';
import { h, clear, icon, banner, showError, chip, fmt } from '../dom.js';
import { state, setSession, activeFrameId, activePlacard, setState } from '../store.js';
import { Viewer } from '../viewer.js';
import { showFrame, viewerShell, zoomButtons, maximizeButton, statusHero, gateList, budgetTable, targetReadout } from './common.js';
import { startLive } from './live.js';
import { createPlacement } from './placement.js';

const OBSERVED = {
  within_tolerance_observed: ['ok', 'observed within the tolerance'],
  outside_tolerance_observed: ['danger', 'observed outside the tolerance'],
  indeterminate: ['warn', 'cannot be told from the tolerance limit'],
  partial: ['warn', 'only part of the outline was measured'],
  no_evidence: ['danger', 'no measurement'],
};

// "Checks (1 open, 2 notes)": open = not met; notes = met, but saying what was not checked
function checksTitle(rep) {
  const open = rep.gates.filter((g) => !g.passed).length;
  const notes = rep.gates.filter((g) => g.passed && g.caveat).length;
  return `Checks (${open} open${notes ? `, ${notes} note${notes === 1 ? '' : 's'}` : ''})`;
}

export function mount(root, ctx) {
  const s = () => state.session;
  const shell = viewerShell({ label: 'Placement' });
  const panel = h('div', { class: 'panel' });
  const split = h('div', { class: 'split' }, shell.wrap, panel);
  root.append(split);
  const viewer = new Viewer(shell.wrap, { label: 'Photo with the placard target outline, metric grid and dimensions' });
  window.__lumenViewer = viewer;   // hook for automated browser tests
  shell.tools.append(maximizeButton(viewer, split), ...zoomButtons(viewer));
  const modePill = h('span', { class: 'pill frozen', 'data-mode': 'frozen' }, 'Frozen photo — not live');
  const frameSel = h('div', { class: 'row' });
  shell.hud.append(modePill, frameSel);
  let destroyed = false;
  let live = null;
  let placement = null;
  let starting = false;
  let last = null;
  let liveRes = null;
  let liveKey = '';
  let liveCentered = false;
  let stopTimer = 0;
  const liveBox = h('div', { class: 'stack', 'data-region': 'live-status' });
  const liveNote = h('div', { class: 'stack' });

  // What goes with the placement view while live: why nothing is drawn, when nothing is (rewritten only when it changes).
  function renderLive() {
    const p = activePlacard(s());
    const t = liveRes && liveRes.tracking;
    const rep = liveRes && p ? liveRes.reports[p.placard_id] : null;
    const key = !t ? 'none' : t.state !== 'tracking' ? `lost:${t.code}` : `${rep ? rep.status : 'none'}:${t.code}`;
    if (key === liveKey) return;
    liveKey = key;
    clear(liveNote);
    if (!t) { liveNote.append(h('p', { class: 'hint' }, 'Starting the camera… Aim the phone at the tags.')); return; }
    if (t.state !== 'tracking') { liveNote.append(banner('warn', 'Tracking lost — nothing is drawn', t.message, t.suggestions)); return; }
    if (rep && rep.status === 'blocked') liveNote.append(banner('danger', 'No target is drawn for this view.', t.code !== 'tracking' ? t.message : null, rep.suggestions));
  }

  // The stored record of the active placard, if there is one (the latest).
  function recordOf(pid) {
    const vs = (s().verifications || []).filter((v) => v.placard_id === pid);
    return vs.length ? vs[vs.length - 1] : null;
  }
  function nextPlacard() {
    const ops = s().operation ? s().operation.placards_resolved : [];
    const open = ops.filter((x) => x.placard_id !== s().active_placard_id && !(s().verifications || []).some((v) => v.placard_id === x.placard_id && v.observation === 'within_tolerance_observed'));
    return open.length ? open[0].placard_id : null;
  }
  function recordCard(p) {
    const v = p ? recordOf(p.placard_id) : null;
    if (!v) return null;
    const [kind, words] = OBSERVED[v.observation] || OBSERVED.indeterminate;
    const U = v.uncertainty ? v.uncertainty.expanded_mm : null;
    const nums = v.radial_mm === null || v.radial_mm === undefined ? '' : `${fmt(v.radial_mm, 2)} mm from the target, turned ${fmt(Math.abs(v.rotation_rad) * 180 / Math.PI, 2)}°${U === null ? '' : ` (uncertainty ±${fmt(U, 2)} mm)`}. `;
    const nxt = nextPlacard();
    return h('div', { class: `card tight record-card ${kind}`, 'data-region': 'placement-record', 'data-observation': v.observation },
      h('h3', null, icon(kind === 'ok' ? 'check' : 'alert'), `${p.placard_id}: recorded, ${words}`),
      h('p', { class: 'hint' }, `${nums}A stored photo measured as measurement evidence — not an acceptance decision.`),
      h('div', { class: 'row wrap' },
        nxt ? h('button', { class: 'btn sm grow primary', type: 'button', 'data-action': 'next-placard', onclick: () => choosePlacard(nxt) }, icon('arrow'), `Next: ${nxt}`) : null,
        h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'open-record', onclick: () => ctx.go('verify') }, icon('eye'), 'Open the record')));
  }

  async function load() {
    const fid = activeFrameId();
    try {
      const p = activePlacard(s());
      const out = await showFrame(viewer, fid, 'image', { uncertainty: true, placards: p ? allPlacardIds() : null });
      if (destroyed) return;
      const first = last === null;
      last = out;
      renderPanel();
      if (first) centerOnTarget();      // open on the target, with some context around it
    } catch (err) { showError(err, 'Could not load the placement view'); }
  }

  function allPlacardIds() {
    const op = s().operation;
    return op ? op.placards_resolved.map((p) => p.placard_id) : [];
  }

  async function choosePlacard(pid) {
    try {
      setSession(await api('PUT', `api/sessions/${s().id}/placard`, { placard_id: pid }));
      if (live) { liveCentered = false; if (placement) placement.reset(); renderPanel(); return; }   // the next live frame is measured against the new target
      await load();
    } catch (err) { showError(err); }
  }

  function centerOnTarget() {
    const src = liveRes || (last && last.overlay);
    if (!src) return;
    const t = (src.primitives || []).find((p) => (p.kind === 'target' || p.kind === 'target_preview') && p.active !== false && p.corners);
    if (!t) return;
    const cx = t.corners.reduce((a, c) => a + c[0], 0) / 4, cy = t.corners.reduce((a, c) => a + c[1], 0) / 4;
    viewer.fit();
    viewer.centerOn(cx, cy, 1.0);
    const r = viewer._box();
    const wpx = Math.hypot(t.corners[1][0] - t.corners[0][0], t.corners[1][1] - t.corners[0][1]) * viewer.t.s;
    if (wpx < r.width * 0.28) viewer.zoomAt(r.width / 2, r.height / 2, (r.width * 0.34) / Math.max(wpx, 1));
  }

  function renderPanel() {
    const sess = s();
    clear(panel);
    clear(frameSel);
    const reg = sess.registration;
    if (reg && reg.frames_used.length > 1 && !live) {
      const cur = activeFrameId();
      const idx = reg.frames_used.indexOf(cur);
      const stepTo = (d) => { setState({ frameId: reg.frames_used[(idx + d + reg.frames_used.length) % reg.frames_used.length] }); viewer.userMoved = false; load(); };
      frameSel.append(
        h('button', { class: 'pill', type: 'button', 'aria-label': 'Previous photo', onclick: () => stepTo(-1) }, '‹'),
        h('span', { class: 'pill' }, `Photo ${idx + 1}/${reg.frames_used.length}`),
        h('button', { class: 'pill', type: 'button', 'aria-label': 'Next photo', onclick: () => stepTo(1) }, '›'));
    }
    const p = activePlacard(sess);
    const rep = last && p ? last.overlay.reports[p.placard_id] : null;
    const blocked = !!(rep && rep.status === 'blocked');
    if (live && placement) { liveKey = ''; clear(liveBox); liveBox.append(placement.el, liveNote); panel.append(liveBox); renderLive(); }
    else if (rep) panel.append(statusHero(rep));
    if (live) {
      /* the checks below describe the frozen photo; the live frame's status is in the box above */
    } else if (rep && rep.status === 'blocked') {
      panel.append(banner('danger', 'No target is drawn.', 'LUMEN does not show a placement outline while required geometry is missing or ambiguous.', rep.suggestions));
    } else if (rep && rep.suggestions.length) {
      panel.append(banner('warn', 'To reach guidance status', null, rep.suggestions));
    }
    const ops = sess.operation ? sess.operation.placards_resolved : [];
    if (ops.length > 1) {
      panel.append(h('div', { class: 'row wrap', role: 'radiogroup', 'aria-label': 'Placard' }, ops.map((x) =>
        h('button', { class: 'btn sm toggle', type: 'button', role: 'radio', 'aria-checked': x.placard_id === sess.active_placard_id ? 'true' : 'false', 'aria-pressed': x.placard_id === sess.active_placard_id ? 'true' : 'false', 'data-placard': x.placard_id, onclick: () => choosePlacard(x.placard_id) }, x.placard_id))));
    }
    const rec = p ? recordOf(p.placard_id) : null;
    const canVerify = !!(sess.local_frame && sess.local_frame.valid);
    if (!live) panel.append(recordCard(p));
    panel.append(
      live
        ? h('button', { class: 'btn sm block', type: 'button', 'data-action': 'center-target', onclick: centerOnTarget }, icon('target'), 'Find target')
        : [
          rec ? null : h('p', { class: 'hint', 'data-region': 'live-intro' }, 'Live placement: the camera watches while you position the placard. The screen and the sound say which way to move it, it turns green when it is in position, and the placement is recorded with a chime. Stand the phone where it sees the tags and the target, so that both hands are free.'),
          h('div', { class: 'row wrap' },
            h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'center-target', disabled: blocked ? true : null, onclick: centerOnTarget }, icon('target'), 'Find target'),
            rec
              ? h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'live', 'aria-pressed': 'false', disabled: blocked || starting ? true : null, onclick: toggleLive }, icon('video'), 'Place again, live')
              : h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'continue', disabled: canVerify ? null : true, onclick: () => ctx.go('verify') }, icon('camera'), 'Check from a photo')),
        ],
      live ? null : h('div', { class: 'eyebrow' }, 'Target'), live ? null : targetReadout(p));
    if (rep && !live) {
      panel.append(
        h('details', { class: 'fold', open: rep.status !== 'prototype_guidance' ? true : null }, h('summary', null, checksTitle(rep)), h('div', { class: 'body' }, gateList(rep))),
        h('details', { class: 'fold' }, h('summary', null, 'Uncertainty budget'), h('div', { class: 'body' }, budgetTable(rep.uncertainty))));
    }
    panel.append(
      h('div', { class: 'legend' },
        h('span', null, h('i', { style: { background: 'var(--ov-target)' } }), 'target outline'),
        h('span', null, h('i', { style: { background: 'var(--ov-target-preview)' } }), 'preview only'),
        live ? h('span', null, h('i', { style: { background: '#5cc8ff' } }), 'placard as measured') : h('span', null, h('i', { style: { background: '#fff' } }), 'offsets from datums')),
      h('p', { class: 'hint' }, live
        ? 'Each picture is shown together with what was measured in that same picture, a moment after it was taken. On the dial, the ring is the tolerance of the record and the disc is the placard, as large as the uncertainty of this view: it is in position when the disc is inside the ring. The record is a stored photo, taken when nothing covers the placard.'
        : 'The outline here is drawn on a photo that does not update. A placard that is already installed can also be checked from one new photo in the Verify step.'),
      sess.mode === 'demo' ? h('div', null, chip('simulated demo data', 'info')) : null);
    // What comes next: placing it (live), or, once a record exists, the record.
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('rectified') },
      live
        ? { id: 'live', label: 'Stop live placement', kind: 'lg', icon: 'pause', onClick: toggleLive }
        : rec
          ? { id: 'continue', label: 'Verify placement', kind: 'primary lg', icon: 'arrow', disabled: !canVerify, onClick: () => ctx.go('verify') }
          : { id: 'live', label: 'Start live placement', kind: 'primary lg', icon: 'video', disabled: blocked || starting, onClick: toggleLive },
    ]);
  }

  function backToPhoto() {
    clearTimeout(stopTimer);
    live = null; liveRes = null; liveCentered = false;
    if (placement) { placement.destroy(); placement = null; }
    viewer.setLayer('grid', true); viewer.setLayer('dims', true);
    split.classList.remove('live', 'live-place');
    modePill.textContent = 'Frozen photo — not live'; modePill.className = 'pill frozen'; modePill.dataset.mode = 'frozen';
    viewer.userMoved = false;
    if (destroyed) return;
    // at once: nothing of the live view stays on screen while the photo is fetched
    viewer.setOverlay({ primitives: [] });
    renderPanel();
    load();
  }

  // The record was stored: the camera stops, and the Place step shows what was recorded.
  function onRecorded(v) {
    const ok = v.observation === 'within_tolerance_observed';
    clearTimeout(stopTimer);
    // leave the green and the chime a moment before the camera is released
    stopTimer = setTimeout(() => { if (live) live.stop(); }, ok ? 1300 : 200);
  }

  async function toggleLive() {
    if (live) { live.stop(); return; }
    if (starting) return;
    starting = true;
    clear(frameSel);
    viewer.userMoved = false;
    const pl = createPlacement({ viewer, split, onRecorded });
    pl.unlock();                         // sound may only be started from this tap
    placement = pl;
    // the picture is busy enough with the placard and its target: the grid and the dimension lines wait on the photo
    viewer.setLayer('grid', false); viewer.setLayer('dims', false);
    const lv = await startLive({ viewer, shell, modePill, placement: pl, onStopped: backToPhoto, onReport: (res) => {
      liveRes = res; renderLive();
      // open on the target the first time it is found; afterwards the operator's own zoom is kept
      if (!liveCentered && res.tracking && res.tracking.state === 'tracking' && (res.primitives || []).some((p) => p.kind.startsWith('target') && p.active !== false && p.corners)) {
        liveCentered = true;
        requestAnimationFrame(centerOnTarget);
      }
    } });
    starting = false;
    if (destroyed) { if (lv) lv.stop(); return; }
    if (lv) { live = lv; pl.attach(lv); split.classList.add('live', 'live-place'); }
    renderPanel();
  }

  ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('rectified') }]);
  load();
  return {
    flush: true,
    redraw: () => viewer.draw(),
    pause() { if (live) live.stop(); },
    destroy() { destroyed = true; clearTimeout(stopTimer); if (live) live.stop(); else if (placement) { placement.destroy(); placement = null; } viewer.destroy(); window.__lumenViewer = null; },
  };
}

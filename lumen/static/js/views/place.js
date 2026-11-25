// Step 6 — placement guidance on the photo (and, when started, live video).
import { api } from '../api.js';
import { h, clear, icon, banner, showError, chip } from '../dom.js';
import { state, setSession, activeFrameId, activePlacard, setState } from '../store.js';
import { Viewer } from '../viewer.js';
import { showFrame, viewerShell, zoomButtons, maximizeButton, statusHero, gateList, budgetTable, targetReadout } from './common.js';
import { startLive } from './live.js';

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
  let last = null;
  let liveRes = null;
  let liveKey = '';
  let liveCentered = false;
  const liveBox = h('div', { class: 'stack', 'data-region': 'live-status' });

  // Status of the frame currently shown by live guidance (rewritten only when it changes).
  function renderLive() {
    const p = activePlacard(s());
    const t = liveRes && liveRes.tracking;
    const rep = liveRes && p ? liveRes.reports[p.placard_id] : null;
    const key = !t ? 'none' : t.state !== 'tracking' ? `lost:${t.code}` : `${rep ? rep.status : 'none'}:${t.code}:${rep && rep.uncertainty ? rep.uncertainty.expanded_mm.toFixed(2) : ''}`;
    if (key === liveKey) return;
    liveKey = key;
    clear(liveBox);
    if (!t) { liveBox.append(banner('info', 'Starting live guidance…', 'Aim the phone at the tags.')); return; }
    if (t.state !== 'tracking') { liveBox.append(banner('warn', 'Tracking lost — nothing is drawn', t.message, t.suggestions)); return; }
    if (rep) {
      liveBox.append(statusHero(rep));
      if (rep.status === 'blocked') liveBox.append(banner('danger', 'No target is drawn for this view.', t.code !== 'tracking' ? t.message : null, rep.suggestions));
      else if (rep.uncertainty) liveBox.append(h('p', { class: 'hint' }, `Expanded uncertainty at the target in this view: ${rep.uncertainty.expanded_mm.toFixed(2)} mm (k=${rep.uncertainty.k}).`));
    }
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
    if (live) { liveKey = ''; panel.append(liveBox); renderLive(); }
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
    panel.append(
      h('div', { class: 'row' },
        h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'center-target', disabled: rep && rep.status === 'blocked' ? true : null, onclick: centerOnTarget }, icon('target'), 'Find target'),
        h('button', { class: 'btn sm grow toggle', type: 'button', 'data-action': 'live', 'aria-pressed': live ? 'true' : 'false', disabled: rep && rep.status === 'blocked' ? true : null, onclick: toggleLive }, icon(live ? 'pause' : 'video'), live ? 'Stop live view' : 'Live guidance')),
      h('div', { class: 'eyebrow' }, 'Target'), targetReadout(p));
    if (rep) {
      panel.append(
        h('details', { class: 'fold', open: rep.status !== 'prototype_guidance' ? true : null }, h('summary', null, checksTitle(rep)), h('div', { class: 'body' }, gateList(rep))),
        h('details', { class: 'fold' }, h('summary', null, 'Uncertainty budget'), h('div', { class: 'body' }, budgetTable(rep.uncertainty))));
    }
    panel.append(
      h('div', { class: 'legend' },
        h('span', null, h('i', { style: { background: 'var(--ov-target)' } }), 'target outline'),
        h('span', null, h('i', { style: { background: 'var(--ov-target-preview)' } }), 'preview only'),
        h('span', null, h('i', { style: { background: '#fff' } }), 'offsets from datums')),
      h('p', { class: 'hint' }, live
        ? 'Live guidance shows each camera frame together with the outline computed for that same frame, a moment after it was taken. Hold the phone still (a stand or a second person helps) and wait for the picture to update before trusting a position.'
        : 'A phone stand or a second person holding the phone makes alignment much easier; neither is required. The outline is drawn on a photo that does not update — use Live guidance to see the panel as it is now.'),
      sess.mode === 'demo' ? h('div', null, chip('simulated demo data', 'info')) : null);
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('rectified') },
      { id: 'continue', label: 'Verify placement', kind: 'primary lg', icon: 'arrow', disabled: !(sess.local_frame && sess.local_frame.valid), onClick: () => ctx.go('verify') },
    ]);
  }

  function backToPhoto() {
    live = null; liveRes = null; liveCentered = false;
    split.classList.remove('live');
    modePill.textContent = 'Frozen photo — not live'; modePill.className = 'pill frozen'; modePill.dataset.mode = 'frozen';
    viewer.userMoved = false;
    if (!destroyed) load();
  }

  async function toggleLive() {
    if (live) { live.stop(); return; }
    clear(frameSel);
    viewer.userMoved = false;
    live = await startLive({ viewer, shell, modePill, onStopped: backToPhoto, onReport: (res) => {
      liveRes = res; renderLive();
      // open on the target the first time it is found; afterwards the operator's own zoom is kept
      if (!liveCentered && res.tracking && res.tracking.state === 'tracking' && (res.primitives || []).some((p) => p.kind.startsWith('target') && p.active !== false && p.corners)) {
        liveCentered = true;
        requestAnimationFrame(centerOnTarget);
      }
    } });
    if (live) split.classList.add('live');
    renderPanel();
  }

  ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('rectified') }]);
  load();
  return {
    flush: true,
    redraw: () => viewer.draw(),
    pause() { if (live) live.stop(); },
    destroy() { destroyed = true; if (live) live.stop(); viewer.destroy(); window.__lumenViewer = null; },
  };
}

// Step 6 — placement guidance on the photo (and, when started, live video).
import { api } from '../api.js';
import { h, clear, icon, banner, showError, chip } from '../dom.js';
import { state, setSession, activeFrameId, activePlacard, setState } from '../store.js';
import { Viewer } from '../viewer.js';
import { showFrame, viewerShell, zoomButtons, statusHero, gateList, budgetTable, targetReadout } from './common.js';
import { startLive } from './live.js';

export function mount(root, ctx) {
  const s = () => state.session;
  const shell = viewerShell({ label: 'Placement' });
  const panel = h('div', { class: 'panel' });
  root.append(h('div', { class: 'split' }, shell.wrap, panel));
  const viewer = new Viewer(shell.wrap, { label: 'Photo with the placard target outline, metric grid and dimensions' });
  window.__lumenViewer = viewer;   // hook for automated browser tests
  shell.tools.append(...zoomButtons(viewer));
  const modePill = h('span', { class: 'pill frozen', 'data-mode': 'frozen' }, 'Frozen photo — not live');
  const frameSel = h('div', { class: 'row' });
  shell.hud.append(modePill, frameSel);
  let destroyed = false;
  let live = null;
  let last = null;

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
    if (!last) return;
    const t = (last.overlay.primitives || []).find((p) => (p.kind === 'target' || p.kind === 'target_preview') && p.active !== false && p.corners);
    if (!t) return;
    const cx = t.corners.reduce((a, c) => a + c[0], 0) / 4, cy = t.corners.reduce((a, c) => a + c[1], 0) / 4;
    viewer.fit();
    viewer.centerOn(cx, cy, 1.0);
    const r = viewer.host.getBoundingClientRect();
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
    if (rep) panel.append(statusHero(rep));
    if (rep && rep.status === 'blocked') {
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
        h('details', { class: 'fold', open: rep.status !== 'prototype_guidance' ? true : null }, h('summary', null, `Checks (${rep.gates.filter((g) => !g.passed).length} open)`), h('div', { class: 'body' }, gateList(rep))),
        h('details', { class: 'fold' }, h('summary', null, 'Uncertainty budget'), h('div', { class: 'body' }, budgetTable(rep.uncertainty))));
    }
    panel.append(
      h('div', { class: 'legend' },
        h('span', null, h('i', { style: { background: 'var(--ov-target)' } }), 'target outline'),
        h('span', null, h('i', { style: { background: 'var(--ov-target-preview)' } }), 'preview only'),
        h('span', null, h('i', { style: { background: '#fff' } }), 'offsets from datums')),
      h('p', { class: 'hint' }, 'A phone stand or a second person holding the phone makes alignment much easier; neither is required. The outline is drawn on a photo that does not update — use Live guidance to see the panel move.'),
      sess.mode === 'demo' ? h('div', null, chip('simulated demo data', 'info')) : null);
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('rectified') },
      { id: 'continue', label: 'Verify placement', kind: 'primary lg', icon: 'arrow', disabled: !(sess.local_frame && sess.local_frame.valid), onClick: () => ctx.go('verify') },
    ]);
  }

  async function toggleLive() {
    if (live) { live.stop(); live = null; modePill.textContent = 'Frozen photo — not live'; modePill.className = 'pill frozen'; modePill.dataset.mode = 'frozen'; viewer.userMoved = false; await load(); return; }
    live = await startLive({ viewer, shell, modePill, onStopped: () => { live = null; modePill.textContent = 'Frozen photo — not live'; modePill.className = 'pill frozen'; modePill.dataset.mode = 'frozen'; viewer.userMoved = false; load(); } });
    renderPanel();
  }

  ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('rectified') }]);
  load();
  return {
    flush: true,
    redraw: () => viewer.draw(),
    pause() { if (live) { live.stop(); live = null; } },
    destroy() { destroyed = true; if (live) live.stop(); viewer.destroy(); },
  };
}

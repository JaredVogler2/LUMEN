// Step 4 — review the registration and define the two datum lines.
import { api } from '../api.js';
import { h, clear, icon, chip, banner, toast, showError, busy, fmt } from '../dom.js';
import { state, setSession, activeFrameId, setState } from '../store.js';
import { Viewer } from '../viewer.js';
import { showFrame, viewerShell, zoomButtons, provenanceChip } from './common.js';

const FRAME_REASON = {
  datums_nearly_parallel: 'the two datum lines are nearly parallel',
  axis_disagreement_exceeds_limit: 'the angle between the datums is outside the configured limit',
  axis_limit_not_configured: 'the record defines no limit for the axis disagreement',
  side_datum_on_unexpected_side: 'the side datum is on the opposite side from the record (wrong edge or wrong panel?)',
  interior_point_on_datum: 'a datum line runs through the tags',
  x_sign_not_configured: 'the sign of x is not defined in the record',
  z_sign_not_configured: 'the sign of z is not defined in the record',
};
const CHECK_REASON = {
  size_mismatch: 'its size does not match the other tags (wrong size entered, or it is not on the same plane)',
  not_square_in_plane: 'it is not square when mapped onto the plane (tilted, bent or on another surface)',
  high_residual: 'its corners do not fit the common plane',
  not_connected_to_gauge: 'it never appears together with the other tags',
};

export function mount(root, ctx) {
  const s = () => state.session;
  const shell = viewerShell({ label: 'Datum review' });
  const panel = h('div', { class: 'panel' });
  root.append(h('div', { class: 'split' }, shell.wrap, panel));
  const guide = h('div', { class: 'guide', hidden: true });
  shell.wrap.append(guide);

  let view = 'image';
  let tool = null;          // null | 'floor' | 'side'
  let snap = true;
  let rect = null;
  let destroyed = false;
  let loading = 0;

  const viewer = new Viewer(shell.wrap, { label: 'Photo with tags and datum lines', onLine: onLine });
  window.__lumenViewer = viewer;   // hook for automated browser tests
  shell.tools.append(...zoomButtons(viewer));

  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'View' },
    h('button', { type: 'button', 'aria-pressed': 'true', 'data-view': 'image', onclick: () => setView('image') }, 'Photo'),
    h('button', { type: 'button', 'aria-pressed': 'false', 'data-view': 'rect', onclick: () => setView('rect') }, 'Rectified'));
  const frameSel = h('div', { class: 'row' });
  const frozen = h('span', { class: 'pill frozen' }, 'Still photo');
  shell.hud.append(seg, frameSel, frozen);

  function setView(v) {
    view = v;
    seg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.view === v ? 'true' : 'false'));
    viewer.userMoved = false;
    load();
  }

  async function load() {
    const fid = activeFrameId();
    if (!fid) return;
    const my = ++loading;
    try {
      const out = await showFrame(viewer, fid, view, { uncertainty: false });
      if (destroyed || my !== loading) return;
      rect = out.rect;
    } catch (err) { showError(err, 'Could not load the view'); }
  }

  function setTool(kind) {
    tool = tool === kind ? null : kind;
    viewer.setTool(tool ? 'line' : 'pan');
    guide.hidden = !tool;
    if (tool) {
      const d = datumDef(tool);
      clear(guide).append(icon('line'), h('span', null, `Drag along ${d ? d.name.toLowerCase() : `the ${tool} datum`}. Pinch to zoom first for precision.`));
    }
    renderPanel();
  }

  function datumDef(kind) {
    const op = s().operation;
    return op ? op.operation.datums[kind] : null;
  }

  async function onLine(p0, p1) {
    if (!tool) return;
    const kind = tool;
    const b = busy(snap ? 'Fitting the edge…' : 'Saving the line…');
    try {
      const body = { frame_id: activeFrameId(), p0, p1, space: view === 'rect' ? 'rect' : 'image', rect: view === 'rect' ? rect : null, snap };
      const out = await api('POST', `api/sessions/${s().id}/datums/${kind}`, body);
      setSession(out.session);
      const d = out.datum;
      if (d.provenance === 'manual' && snap) toast({ title: 'No clear edge found there', detail: 'The line was kept as drawn (manual). Zoom in and draw closer to the edge to measure it.', kind: 'error', timeout: 8000 });
      else if (d.distinctness !== null && d.distinctness < 1.5) toast({ title: 'Two similar edges are close together', detail: 'Check that the line sits on the intended one.', kind: 'error', timeout: 8000 });
      tool = null; viewer.setTool('pan'); guide.hidden = true;
      await load();
      renderPanel();
    } catch (err) { showError(err, 'Could not set the datum'); } finally { b.done(); }
  }

  async function clearDatum(kind) {
    try { setSession(await api('DELETE', `api/sessions/${s().id}/datums/${kind}`)); await load(); renderPanel(); } catch (err) { showError(err); }
  }

  async function confirm() {
    const b = busy('Confirming datums…');
    try {
      setSession(await api('POST', `api/sessions/${s().id}/datums/confirm?confirmed=true`));
      ctx.go('rectified');
    } catch (err) { showError(err); } finally { b.done(); }
  }

  function datumCard(kind) {
    const def = datumDef(kind);
    const d = s().datums[kind];
    const color = kind === 'floor' ? 'var(--ov-floor)' : 'var(--ov-side)';
    const head = h('div', { class: 'row wrap' },
      h('span', { style: { width: '14px', height: '14px', borderRadius: '4px', background: color, flex: 'none' } }),
      h('span', { style: { fontWeight: 750 } }, def ? `${def.id} · ${def.name}` : `${kind} datum`),
      d ? provenanceChip(d.provenance) : chip('not set', 'warn'),
      d && d.confirmed ? chip('confirmed', 'ok') : null);
    const facts = d ? h('div', { class: 'hint' },
      d.feature_kind === 'step' ? 'Fitted to an edge' : d.feature_kind === 'bar' ? 'Fitted to a thin line' : 'Hand-drawn, not fitted',
      d.sigma_perp_mm !== null && d.provenance === 'measured' ? ` · ±${fmt(d.sigma_perp_mm, 3)} mm (1σ)` : '',
      d.distinctness !== null && d.distinctness !== undefined ? ` · ${d.distinctness >= 10 ? 'clearly' : `×${fmt(d.distinctness, 1)}`} stronger than the next edge` : '',
      d.offset_applied_mm ? ` · configured offset ${d.offset_applied_mm} mm applied` : '',
      d.on_plane_evidence === 'configured' ? ' · declared on the target plane (not verified by a second view)' : '') : null;
    const needs3d = def && (def.feature === 'curve_tangent' || def.feature === 'curve_outer_extreme');
    return h('div', { class: 'card tight', 'data-datum': kind },
      head, facts,
      def && def.notes ? h('div', { class: 'hint' }, def.notes) : null,
      needs3d ? banner('warn', 'Rounded-edge datum', 'A single flat view cannot locate this line. A line drawn here is only a preview; capture a sweep so the edge can be modelled in 3-D.') : null,
      h('div', { class: 'row' },
        h('button', { class: 'btn sm toggle grow', type: 'button', 'aria-pressed': tool === kind ? 'true' : 'false', 'data-action': `draw-${kind}`, onclick: () => setTool(kind) }, icon('line'), tool === kind ? 'Drawing… tap to cancel' : (d ? 'Redraw' : 'Draw line')),
        d ? h('button', { class: 'btn sm ghost', type: 'button', 'aria-label': `Clear ${kind} datum`, onclick: () => clearDatum(kind) }, icon('trash')) : null));
  }

  function renderPanel() {
    const sess = s();
    const reg = sess.registration;
    clear(panel);
    clear(frameSel);
    if (!reg) { panel.append(banner('warn', 'Not analysed yet.', 'Go back to Capture and run the analysis.')); return; }
    if (reg.frames_used.length > 1) {
      const cur = activeFrameId();
      const idx = reg.frames_used.indexOf(cur);
      const stepTo = (d) => { const n = reg.frames_used[(idx + d + reg.frames_used.length) % reg.frames_used.length]; setState({ frameId: n }); viewer.userMoved = false; load(); renderPanel(); };
      frameSel.append(
        h('button', { class: 'pill', type: 'button', 'aria-label': 'Previous photo', onclick: () => stepTo(-1) }, '‹'),
        h('span', { class: 'pill' }, `Photo ${idx + 1}/${reg.frames_used.length}`),
        h('button', { class: 'pill', type: 'button', 'aria-label': 'Next photo', onclick: () => stepTo(1) }, '›'));
    }

    // --- registration summary
    const checks = Object.values(reg.checks);
    const bad = checks.filter((c) => !c.consistent);
    const excluded = Object.entries(reg.excluded_tags || {});
    panel.append(h('div', { class: 'card tight' },
      h('div', { class: 'row wrap' }, h('span', { class: 'eyebrow' }, 'Registration'),
        chip(`${reg.n_tags} tag${reg.n_tags === 1 ? '' : 's'}`, reg.n_tags >= 2 ? 'ok' : 'warn'),
        chip(`${reg.frames_used.length} photo${reg.frames_used.length === 1 ? '' : 's'}`, 'solid'),
        chip(`fit ${fmt(reg.rms_px, 2)} px`, 'solid'),
        reg.independently_checked ? chip('cross-checked', 'ok') : chip('no independent check', 'warn')),
      checks.filter((c) => c.checked).length ? h('div', { class: 'hint' }, checks.filter((c) => c.checked).map((c) => `ID ${c.tag_id}: size ${fmt((c.size_ratio - 1) * 100, 2)} % ± ${fmt(c.size_ratio_sigma * 100, 2)} %`).join(' · ')) : null));
    if (bad.length) panel.append(banner('danger', 'Tags contradict each other.', 'No guidance until this is resolved.', bad.map((c) => `ID ${c.tag_id}: ${c.reasons.map((r) => CHECK_REASON[r] || r).join('; ')}`)));
    if (excluded.length) panel.append(banner('warn', 'A tag was excluded.', null, excluded.map(([id, rs]) => `ID ${id}: ${rs.map((r) => CHECK_REASON[r] || r).join('; ')}`)));
    if (reg.n_tags < 2) panel.append(banner('warn', 'Only one tag.', 'Nothing checks its size or that the panel is flat here. Add a second measured tag for guidance status.'));
    if ((reg.flags || []).includes('high_chi2')) panel.append(banner('warn', 'The tags do not fit one plane as well as their sharpness suggests.', 'Typical causes: lens distortion without a calibration, a bent tag, a panel that is not flat.'));

    // --- datums
    panel.append(h('div', { class: 'eyebrow' }, 'Datums'), datumCard('floor'), datumCard('side'),
      h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: snap ? true : null, onchange: (ev) => { snap = ev.target.checked; } }), h('span', null, 'Snap the line to the nearest edge and measure it')));

    // --- local frame
    const lf = sess.local_frame;
    const hasBoth = sess.datums.floor && sess.datums.side;
    if (lf) {
      panel.append(h('div', { class: 'card tight' },
        h('div', { class: 'row wrap' }, h('span', { class: 'eyebrow' }, 'Local frame'), lf.valid ? chip('valid', 'ok') : chip('not usable', 'danger'), chip(lf.policy.replace(/_/g, ' '), 'solid'), provenanceChip(lf.provenance)),
        h('div', { class: 'hint' }, `Angle between datums ${fmt(lf.raw_angle_deg, 2)}° (nominal ${fmt(lf.nominal_angle_deg, 0)}°, off by ${fmt(lf.axis_disagreement_deg, 2)}°) · side datum on the ${lf.side_of_panel}`),
        lf.valid ? null : h('div', { class: 'hint', style: { color: 'var(--danger)' } }, lf.reasons.map((r) => FRAME_REASON[r] || r).join('; '))));
    } else if (hasBoth) {
      panel.append(banner('danger', 'The local frame cannot be built.', 'The operation record does not define the axis policy or the datum signs.'));
    }
    panel.append(h('div', { class: 'legend' },
      h('span', null, h('i', { style: { background: 'var(--ov-tag)' } }), 'accepted tag'),
      h('span', null, h('i', { style: { background: 'var(--ov-floor)' } }), 'floor datum'),
      h('span', null, h('i', { style: { background: 'var(--ov-side)' } }), 'side datum'),
      h('span', null, h('i', { style: { background: 'var(--ov-grid)' } }), 'metric grid')),
      h('p', { class: 'hint' }, view === 'rect' ? 'Rectified view: only things on the panel face are in their true position. The floor, a curved edge or a raised trim appear displaced.' : 'Draw each line along the feature the record names. You can switch to the rectified view at any time.'));

    const ok = !!(lf && lf.valid && hasBoth);
    const done = ok && sess.datums.floor.confirmed && sess.datums.side.confirmed;
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('capture') },
      done
        ? { id: 'continue', label: 'Rectified view', kind: 'primary lg', icon: 'arrow', onClick: () => ctx.go('rectified') }
        : { id: 'confirm-datums', label: 'Confirm datums', kind: 'primary lg', icon: 'check', disabled: !ok, onClick: confirm, title: ok ? '' : 'Define both datum lines first' },
    ]);
  }

  renderPanel();
  load();
  return {
    flush: true,
    redraw: () => viewer.draw(),
    destroy() { destroyed = true; viewer.destroy(); },
  };
}

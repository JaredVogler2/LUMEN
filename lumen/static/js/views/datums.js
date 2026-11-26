// Step 4 — review the registration and the two datum lines.
//
// Datums are proposed automatically (straight features found in the photos,
// re-found in every other photo and classified in 3-D).  The operator reviews
// the evidence, can pick another detected feature by tapping it, or draw a
// line by hand, and finally confirms.  Nothing is confirmed automatically.
import { api } from '../api.js';
import { h, clear, icon, chip, banner, toast, showError, busy, fmt } from '../dom.js';
import { state, setSession, activeFrameId, setState } from '../store.js';
import { Viewer } from '../viewer.js';
import { showFrame, viewerShell, zoomButtons, maximizeButton, provenanceChip } from './common.js';

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
  moved_or_misread: 'it moved between photos (or was misread in one)',
  declared_size_inconsistent: 'its entered size contradicts the other tags',
  off_the_common_plane: 'it is not on the same plane as the other tags',
};
const EVIDENCE = {
  verified_multiview: ['verified on the face in 3-D', 'ok'],
  cylinder_model: ['from the 3-D edge model', 'ok'],
  configured: ['declared on the face (not verified)', 'info'],
  off_plane: ['NOT on the face', 'danger'],
  unverified: ['not verified', 'warn'],
};

export function mount(root, ctx) {
  const s = () => state.session;
  const shell = viewerShell({ label: 'Datum review' });
  const panel = h('div', { class: 'panel' });
  const split = h('div', { class: 'split' }, shell.wrap, panel);
  root.append(split);
  const guide = h('div', { class: 'guide', hidden: true });
  shell.wrap.append(guide);

  let view = 'image';
  let tool = null;          // null | 'floor' | 'side'      (drawing a line)
  let picking = null;       // null | 'floor' | 'side'      (tapping a detected feature)
  let snap = true;
  let detailsOpen = false;        // the fold with how the lines were found; kept as the operator left it
  let rect = null;
  let destroyed = false;
  let loading = 0;

  const viewer = new Viewer(shell.wrap, { label: 'Photo with tags and datum lines', onLine, onTap });
  window.__lumenViewer = viewer;   // hook for automated browser tests
  shell.tools.append(maximizeButton(viewer, split), ...zoomButtons(viewer));

  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'View' },
    h('button', { type: 'button', 'aria-pressed': 'true', 'data-view': 'image', onclick: () => setView('image') }, 'Photo'),
    h('button', { type: 'button', 'aria-pressed': 'false', 'data-view': 'rect', onclick: () => setView('rect') }, 'Rectified'));
  const frameSel = h('div', { class: 'row' });
  shell.hud.append(seg, frameSel, h('span', { class: 'pill frozen' }, 'Still photo'));

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
      const out = await showFrame(viewer, fid, view, { uncertainty: false, candidates: true });
      if (destroyed || my !== loading) return;
      rect = out.rect;
    } catch (err) { showError(err, 'Could not load the view'); }
  }

  function datumDef(kind) {
    const op = s().operation;
    return op ? op.operation.datums[kind] : null;
  }

  function setMode(nextTool, nextPick) {
    tool = nextTool; picking = nextPick;
    viewer.setTool(tool ? 'line' : 'pan');
    viewer.setLayer('candidates', !!picking);
    guide.hidden = !(tool || picking);
    if (tool) {
      const d = datumDef(tool);
      clear(guide).append(icon('line'), h('span', null, `Drag along ${d ? d.name.toLowerCase() : `the ${tool} datum`}. Pinch to zoom first for precision.`));
    } else if (picking) {
      clear(guide).append(icon('target'), h('span', null, `Tap the dashed line that is the ${picking} datum. Green lines are verified on the panel face.`));
    }
    renderPanel();
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
      else if (d.on_plane_evidence === 'off_plane') toast({ title: 'That feature is not on the panel face', detail: 'It moves against the tags between photos. It cannot serve as a datum on the face.', kind: 'error', timeout: 9000 });
      else if (d.distinctness !== null && d.distinctness < 1.5) toast({ title: 'Two similar edges are close together', detail: 'Check that the line sits on the intended one.', kind: 'error', timeout: 8000 });
      tool = null; viewer.setTool('pan'); guide.hidden = true;
      await load();
      renderPanel();
    } catch (err) { showError(err, 'Could not set the datum'); } finally { b.done(); }
  }

  async function onTap(pt) {
    if (!picking) return;
    const hit = viewer.nearest('candidate', pt[0], pt[1], 30);
    if (!hit) { toast({ title: 'No detected line there', detail: 'Tap directly on a dashed line, or draw the line by hand.', kind: 'info', timeout: 3500 }); return; }
    const kind = picking;
    const b = busy('Using the selected feature…');
    try {
      const out = await api('POST', `api/sessions/${s().id}/datums/select`, { kind, candidate_id: hit.id });
      setSession(out.session);
      picking = null; viewer.setLayer('candidates', false); guide.hidden = true;
      await load();
      renderPanel();
    } catch (err) { showError(err, 'Could not use that feature'); } finally { b.done(); }
  }

  async function findAutomatically(quiet = false) {
    const b = busy('Looking for the floor seam and the side edge…');
    try {
      const out = await api('POST', `api/sessions/${s().id}/datums/propose`, undefined, { timeout: 180000 });
      setSession(out.session);
      const p = out.proposals;
      const found = Object.keys(p.applied);
      if (!quiet || found.length < 2) {
        if (found.length === 2) toast({ title: 'Both datums found', detail: 'Review them on the picture and confirm.', kind: 'ok', timeout: 4000 });
        else toast({ title: found.length ? `Only the ${found[0]} datum was found` : 'No datum was found', detail: (p.hints || []).join(' '), kind: 'error', timeout: 10000 });
      }
      await load();
      renderPanel();
    } catch (err) { showError(err, 'Automatic search failed'); } finally { b.done(); }
  }

  async function clearDatum(kind) {
    try { setSession(await api('DELETE', `api/sessions/${s().id}/datums/${kind}`)); await load(); renderPanel(); } catch (err) { showError(err); }
  }

  async function confirm() {
    const b = busy('Confirming datums…');
    try {
      setSession(await api('POST', `api/sessions/${s().id}/datums/confirm?confirmed=true`));
      ctx.go('place');
    } catch (err) { showError(err); } finally { b.done(); }
  }

  function datumCard(kind) {
    const def = datumDef(kind);
    const d = s().datums[kind];
    const props = (s().research || {}).datum_proposals || null;
    const nAlt = props ? (props[kind] || []).length : 0;
    const ambiguous = props ? props[`ambiguous_${kind}`] : false;
    const color = kind === 'floor' ? 'var(--ov-floor)' : 'var(--ov-side)';
    const ev = d ? EVIDENCE[d.on_plane_evidence] || EVIDENCE.unverified : null;
    const head = h('div', { class: 'row wrap' },
      h('span', { style: { width: '14px', height: '14px', borderRadius: '4px', background: color, flex: 'none' } }),
      h('span', { style: { fontWeight: 750 } }, def ? `${def.id} · ${def.name}` : `${kind} datum`),
      d ? provenanceChip(d.provenance) : chip('not set', 'warn'),
      d ? chip(ev[0], ev[1]) : null,
      d && d.confirmed ? chip('confirmed', 'ok') : null);
    const what = d ? (d.feature_kind === 'step' ? 'Fitted to an edge' : d.feature_kind === 'bar' ? 'Fitted to a thin line' : d.feature_kind === 'cylinder_model' ? 'Tangent line of the fitted rounded edge' : 'Hand-drawn, not fitted') : '';
    const facts = d ? h('div', { class: 'hint' },
      what,
      d.sigma_perp_mm !== null && d.provenance !== 'manual' ? ` · ±${fmt(d.sigma_perp_mm, 3)} mm (1σ)` : '',
      d.image_fit && d.image_fit.n_views > 1 ? ` · seen in ${d.image_fit.n_views} photos` : '',
      d.offset_applied_mm ? ` · configured offset ${d.offset_applied_mm} mm applied` : '') : null;
    const needs3d = def && (def.feature === 'curve_tangent' || def.feature === 'curve_outer_extreme');
    return h('div', { class: 'card tight', 'data-datum': kind },
      head, facts,
      d && d.notes && d.notes.length ? h('div', { class: 'hint' }, d.notes.join(' · ')) : null,
      !d && def && def.notes ? h('div', { class: 'hint' }, def.notes) : null,
      d && ambiguous ? banner('warn', 'More than one similar feature was found.', 'Tap “Choose line” and select the intended one.') : null,
      d && d.depth_neighbour ? h('div', { 'data-region': `step-beside-${kind}` }, banner('warn', 'Is this the right line?',
        `A fixed edge about ${fmt(Math.abs(d.depth_neighbour.height_mm), 0)} mm ${d.depth_neighbour.height_mm > 0 ? 'in front of' : 'behind'} the panel face runs `
        + `${d.depth_neighbour.relation === 'beyond' ? `${fmt(Math.abs(d.depth_neighbour.gap_mm), 0)} mm beyond` : 'right beside'} it. `
        + 'The line may be the edge of a trim or moulding, not the feature the record means. Tap “Choose line” and select the intended one, or draw it by hand.')) : null,
      needs3d && d && d.feature_kind !== 'cylinder_model' ? banner('warn', 'Rounded-edge datum', 'A line on one flat view cannot locate this edge. Capture 5–9 photos while moving sideways around it, then search again.') : null,
      h('div', { class: 'row wrap' },
        nAlt > 0 && !needs3d ? h('button', { class: 'btn sm toggle grow nowrap', type: 'button', 'aria-pressed': picking === kind ? 'true' : 'false', 'data-action': `pick-${kind}`, onclick: () => setMode(null, picking === kind ? null : kind) }, picking === kind ? 'Tap a line (cancel)' : `Choose line (${nAlt})`) : null,
        h('button', { class: 'btn sm toggle grow nowrap', type: 'button', 'aria-pressed': tool === kind ? 'true' : 'false', 'data-action': `draw-${kind}`, onclick: () => setMode(tool === kind ? null : kind, null) }, tool === kind ? 'Drawing (cancel)' : 'Draw by hand'),
        d ? h('button', { class: 'btn sm ghost', type: 'button', 'aria-label': `Clear ${kind} datum`, onclick: () => clearDatum(kind) }, icon('trash')) : null));
  }

  // Flashlight experiment: research evidence shown next to the outline method.  It never moves a datum.
  function photometricCard() {
    const sess = s();
    if (!(state.config && state.config.features && state.config.features.photometric_research)) return null;
    const def = datumDef('side');
    const rec = (sess.research || {}).photometric || null;
    const curved = def && (def.feature === 'curve_tangent' || def.feature === 'curve_outer_extreme');
    if (!curved && !rec) return null;
    const lit = sess.frames.filter((f) => f.torch === 'on' && f.selected !== false).length;
    const body = [];
    if (rec) {
      const r = rec.result;
      const deg = (v) => (v === null || v === undefined ? '—' : fmt(v, 3));
      if (r.status === 'supported') {
        const off = rec.offset_from_side_datum_mm;
        const cmp = rec.comparison_with_outline_method;
        body.push(h('div', { class: 'row wrap' }, chip('consistent shading model', 'info'), chip(`radius ${fmt(r.radius_mm, 1)} ± ${fmt(r.sigma_radius_mm, 1)} mm${r.radius_source === 'hint_fixed' ? ' (configured)' : ''}`, 'solid'), chip(`${r.n_views_used} photos used`, 'solid')),
          h('div', { class: 'hint' }, off === null || off === undefined ? '' : `Start of the bend ${fmt(Math.abs(off), 2)} mm ${off > 0 ? 'further into the flat face than' : 'further out than'} the side datum in use (stated ±${fmt(r.sigma_tangent_mm, 2)} mm, direction ±${deg(r.sigma_angle_deg)}°).`),
          cmp && cmp.available ? h('div', { class: 'hint' }, cmp.consistent
            ? `Agrees with the outline method: ${fmt(cmp.offset_difference_mm, 2)} mm apart, ${fmt(cmp.z_offset, 1)} standard deviations.`
            : `Disagrees with the outline method: ${fmt(cmp.offset_difference_mm, 2)} mm apart, ${fmt(cmp.z_offset, 1)} standard deviations. One of the two is wrong; they are not combined.`) : null);
      } else {
        body.push(h('div', { class: 'row wrap' }, chip(r.status === 'needs_view' ? 'needs other photos' : 'no result', 'warn')),
          h('ul', { class: 'hint', style: { margin: '0', paddingLeft: '18px' } }, (r.suggestions && r.suggestions.length ? r.suggestions : r.reasons.map((x) => String(x).replace(/_/g, ' '))).map((t) => h('li', null, t))));
      }
    }
    return h('div', { class: 'card tight', 'data-region': 'photometric' },
      h('div', { class: 'row wrap' }, h('span', { class: 'eyebrow' }, 'Flashlight experiment'), chip('research only', 'info')),
      h('p', { class: 'hint' }, 'Estimates where the rounded edge begins from how the phone light falls off around it, for comparison with the outline method. It is never used for a datum, a check or the target, and it can be wrong without noticing on patterned surfaces.'),
      body,
      h('div', { class: 'row wrap' },
        h('button', { class: 'btn sm', type: 'button', 'data-action': 'photometric-run', onclick: async () => {
          const b = busy('Analysing the photos taken with the light on…');
          try { const out = await api('POST', `api/sessions/${sess.id}/research/photometric`, null, { timeout: 180000 }); setSession(out.session); renderPanel(); } catch (err) { showError(err, 'The experiment could not run'); } finally { b.done(); }
        } }, icon('torch'), rec ? 'Analyse again' : 'Analyse torch photos'),
        h('span', { class: 'hint' }, lit ? `${lit} photo${lit === 1 ? '' : 's'} with the light on` : 'No photo with the light on yet')));
  }

  function renderPanel() {
    const sess = s();
    const reg = sess.registration;
    clear(panel);
    clear(frameSel);
    if (!reg) { panel.append(banner('warn', 'Not analysed yet.', 'Go back to Photos and analyse them.')); return; }
    if (reg.frames_used.length > 1) {
      const cur = activeFrameId();
      const idx = reg.frames_used.indexOf(cur);
      const stepTo = (d) => { const n = reg.frames_used[(idx + d + reg.frames_used.length) % reg.frames_used.length]; setState({ frameId: n }); viewer.userMoved = false; load(); renderPanel(); };
      frameSel.append(
        h('button', { class: 'pill', type: 'button', 'aria-label': 'Previous photo', onclick: () => stepTo(-1) }, '‹'),
        h('span', { class: 'pill' }, `Photo ${idx + 1}/${reg.frames_used.length}`),
        h('button', { class: 'pill', type: 'button', 'aria-label': 'Next photo', onclick: () => stepTo(1) }, '›'));
    }

    // What the operator has to do here is one thing: look whether the two lines are the right ones.  Everything that
    // explains how they were found goes under "Details"; what stands in the way of a result stays in view.
    const details = [];
    panel.append(h('p', { class: 'view-lead', 'data-region': 'lines-lead' },
      'Check the two lines on the picture: ', h('b', { style: { color: 'var(--ov-floor)' } }, 'orange'), ' on the floor line, ',
      h('b', { style: { color: 'var(--ov-side)' } }, 'purple'), ' on the side line of the job. Then confirm.'));

    // --- registration summary
    const checks = Object.values(reg.checks);
    const bad = checks.filter((c) => !c.consistent);
    const excluded = Object.entries(reg.excluded_tags || {});
    const mv = reg.multiview && !reg.multiview.skipped ? reg.multiview : null;
    const cop = mv ? (mv.checks.coplanarity || {}) : {};
    details.push(h('div', { class: 'card tight' },
      h('div', { class: 'row wrap' }, h('span', { class: 'eyebrow' }, 'Registration'),
        chip(`${reg.n_tags} tag${reg.n_tags === 1 ? '' : 's'}`, reg.n_tags >= 2 ? 'ok' : 'warn'),
        chip(`${reg.frames_used.length} photo${reg.frames_used.length === 1 ? '' : 's'}`, 'solid'),
        chip(`fit ${fmt(reg.rms_px, 2)} px`, 'solid'),
        reg.independently_checked ? chip('cross-checked', 'ok') : chip('no independent check', 'warn'),
        cop.verdict === 'coplanar' ? chip('flat in 3-D', 'ok') : cop.verdict === 'not_established' ? chip('3-D check inconclusive', 'warn') : null,
        mv && mv.calibration_check ? chip(mv.calibration_check.consistent ? 'lens profile confirmed' : 'lens profile disagrees', mv.calibration_check.consistent ? 'ok' : 'danger') : null),
      mv && cop.verdict === 'coplanar' ? h('div', { class: 'hint' }, `Tags lie on one plane within ${fmt(cop.worst_center_offset_mm, 2)} mm; a ${fmt(cop.resolving_power_mm, 1)} mm step would have been noticed. Views span ${fmt(mv.checks.diversity.max_view_angle_deg, 0)}°.`) : null,
      mv && cop.verdict === 'not_established' ? h('div', { class: 'hint' }, 'The photos were taken from nearly the same place, so flatness cannot be checked in 3-D. Move further sideways between photos.') : null));
    if (bad.length) panel.append(banner('danger', 'Tags contradict each other.', 'No target is shown until this is resolved.', bad.map((c) => `ID ${c.tag_id}: ${c.reasons.map((r) => CHECK_REASON[r] || r).join('; ')}`)));
    if (excluded.length) panel.append(banner('warn', excluded.length === 1 ? 'A tag was excluded.' : 'Tags were excluded.', null, excluded.map(([id, rs]) => `ID ${id}: ${rs.map((r) => CHECK_REASON[r] || r).join('; ')}`)));
    if (reg.n_tags < 2) panel.append(banner('warn', 'Only one tag.', 'Nothing checks its size or that the panel is flat here. Add a second measured tag to reach “Ready”.'));
    if ((reg.flags || []).includes('high_chi2')) panel.append(banner('warn', 'The tags do not fit one plane as well as their sharpness suggests.', 'Typical causes: lens distortion without a calibration, a bent tag, a panel that is not flat.'));

    // --- datums
    const props = (sess.research || {}).datum_proposals || null;
    panel.append(
      h('div', { class: 'row' }, h('span', { class: 'eyebrow grow' }, 'Lines'),
        h('button', { class: 'btn sm', type: 'button', 'data-action': 'auto-datums', onclick: () => findAutomatically(false) }, icon('refresh'), props ? 'Search again' : 'Find automatically')),
      props && props.hints && props.hints.length ? banner('warn', 'Not everything was found.', null, props.hints) : null,
      datumCard('floor'), datumCard('side'),
      h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: snap ? true : null, onchange: (ev) => { snap = ev.target.checked; } }), h('span', null, 'When drawing by hand, snap to the nearest edge and measure it')));

    const cyl = sess.surface && sess.surface.cylinder;
    if (cyl && cyl.ok) {
      details.push(h('div', { class: 'card tight' },
        h('div', { class: 'row wrap' }, h('span', { class: 'eyebrow' }, 'Rounded edge'), chip(`radius ${fmt(cyl.radius_mm, 1)} mm${cyl.radius_fixed ? ' (configured)' : ` ± ${fmt(cyl.sigma_radius_mm, 1)}`}`, 'solid'), chip(`${cyl.n_views} silhouettes`, 'solid')),
        h('div', { class: 'hint' }, `Tangent line located to ±${fmt(cyl.sigma_tangent_mm, 2)} mm; silhouettes agree with a cylinder within ${fmt(cyl.rms_mm, 2)} mm. A uniform rounded edge fixes the direction and the sideways position only — the floor datum fixes the height.`)));
    }

    const exp = photometricCard();
    if (exp) details.push(exp);

    // --- local frame
    const lf = sess.local_frame;
    const hasBoth = sess.datums.floor && sess.datums.side;
    if (lf) {
      // two lines that do not give a usable frame stand in the way: that is said in view, not under "Details"
      (lf.valid ? details : { push: (n) => panel.append(n) }).push(h('div', { class: 'card tight' },
        h('div', { class: 'row wrap' }, h('span', { class: 'eyebrow' }, 'Local frame'), lf.valid ? chip('valid', 'ok') : chip('not usable', 'danger'), chip(lf.policy.replace(/_/g, ' '), 'solid'), provenanceChip(lf.provenance)),
        h('div', { class: 'hint' }, `Angle between datums ${fmt(lf.raw_angle_deg, 2)}° (nominal ${fmt(lf.nominal_angle_deg, 0)}°, off by ${fmt(lf.axis_disagreement_deg, 2)}°) · side datum on the ${lf.side_of_panel}`),
        lf.valid ? null : h('div', { class: 'hint', style: { color: 'var(--danger)' } }, lf.reasons.map((r) => FRAME_REASON[r] || r).join('; '))));
    } else if (hasBoth) {
      panel.append(banner('danger', 'The local frame cannot be built.', 'The operation record does not define the axis policy or the datum signs.'));
    }
    details.push(h('div', { class: 'legend' },
      h('span', null, h('i', { style: { background: 'var(--ov-tag)' } }), 'accepted tag'),
      h('span', null, h('i', { style: { background: 'var(--ov-floor)' } }), 'floor datum'),
      h('span', null, h('i', { style: { background: 'var(--ov-side)' } }), 'side datum'),
      h('span', null, h('i', { style: { background: 'var(--ov-grid)' } }), 'metric grid')),
      h('p', { class: 'hint' }, view === 'rect' ? 'Rectified view: only things on the panel face are in their true position. The floor, a curved edge or a raised trim appear displaced.' : 'Check that each coloured line lies on the feature the record names. Switch to the rectified view for a square-on look.'));
    panel.append(h('details', { class: 'fold', 'data-region': 'lines-details', open: detailsOpen ? true : null, ontoggle: (ev) => { detailsOpen = ev.target.open; } },
      h('summary', null, 'Details'), h('div', { class: 'body' }, details)));

    const ok = !!(lf && lf.valid && hasBoth);
    const done = ok && sess.datums.floor.confirmed && sess.datums.side.confirmed;
    // a tool, not a stop: the square-on view with the optional rule measurement that checks the entered tag sizes
    if (sess.registration) {
      panel.append(h('button', { class: 'btn sm block', type: 'button', 'data-action': 'scale-check', onclick: () => ctx.go('rectified') }, icon('ruler'), 'Check the scale with a rule (optional)'));
    }
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('capture') },
      done
        ? { id: 'continue', label: 'Continue', kind: 'primary lg', icon: 'arrow', onClick: () => ctx.go('place') }
        : { id: 'confirm-datums', label: 'Confirm lines', kind: 'primary lg', icon: 'check', disabled: !ok, onClick: confirm, title: ok ? '' : 'Both lines are needed first' },
    ]);
  }

  renderPanel();
  // First visit with nothing set: search automatically so the operator starts from a proposal.
  const sess0 = s();
  const fresh = sess0.registration && !sess0.datums.floor && !sess0.datums.side && !((sess0.research || {}).datum_proposals);
  if (fresh) findAutomatically(true); else load();
  return {
    flush: true,
    redraw: () => viewer.draw(),
    destroy() { destroyed = true; viewer.destroy(); },
  };
}

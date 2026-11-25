// Pieces shared by several screens.
import { api, fetchBlob, loadImage } from '../api.js';
import { h, icon, chip, fmt } from '../dom.js';
import { state, lengthOut } from '../store.js';

export const STATUS_TEXT = {
  blocked: { title: 'No guidance', icon: 'block', sub: 'Required geometry is missing, ambiguous or unsupported.' },
  research_preview: { title: 'Research preview', icon: 'alert', sub: 'Shown for inspection only — qualification gates are not met.' },
  prototype_guidance: { title: 'Prototype guidance', icon: 'check', sub: 'All computed gates pass. Not physically validated; not acceptance.' },
  validated: { title: 'Within validated domain', icon: 'check', sub: 'Measurement evidence, not product acceptance.' },
};

export function statusHero(report) {
  const st = STATUS_TEXT[report.status] || STATUS_TEXT.blocked;
  return h('div', { class: `status-hero ${report.status}`, role: 'status', 'data-status': report.status },
    icon(st.icon),
    h('div', { class: 'grow' },
      h('div', { class: 'big' }, st.title),
      h('div', { class: 'small' }, report.notice || st.sub)));
}

export function gateList(report, { onlyFailed = false } = {}) {
  const gates = report.gates.filter((g) => (onlyFailed ? !g.passed : true));
  // failed gates first, blocking before qualifying; then what passed with a note; then the ticks
  const rank = (g) => (g.passed ? (g.caveat ? 2 : 3) : g.severity === 'blocking' ? 0 : 1);
  gates.sort((a, b) => rank(a) - rank(b));
  return h('div', { class: 'gates' }, gates.map((g) =>
    h('div', { class: `gate ${g.passed ? 'pass' : 'fail'} ${g.severity}${g.caveat ? ' caveat' : ''}`, 'data-gate': g.code },
      // a caveat passes but says what was NOT checked: an "i", not a tick
      icon(g.passed ? (g.caveat ? 'info' : 'check') : g.severity === 'blocking' ? 'block' : 'alert'),
      h('div', null, g.message, (!g.passed || g.caveat) && g.suggestion ? h('div', { class: 'fix' }, g.suggestion) : null))));
}

export function budgetTable(u) {
  if (!u) return h('p', { class: 'hint' }, 'No uncertainty was evaluated.');
  const total = u.components.reduce((a, c) => a + c.variance_mm2, 0) || 1;
  const rows = u.components.slice().sort((a, b) => b.variance_mm2 - a.variance_mm2).map((c) =>
    h('tr', null,
      h('td', null, c.name,
        h('span', { class: 'note' }, `${c.kind === 'random' ? 'repeatability' : c.distribution === 'rectangular' ? 'systematic bound' : 'systematic (1σ)'} · ${c.note}`),
        h('div', { class: `bar ${c.kind === 'random' ? '' : 'sys'}` }, h('i', { style: { width: `${Math.round((100 * c.variance_mm2) / total)}%` } }))),
      h('td', { class: 'v' }, `${fmt(c.value_mm, 2)} mm`)));
  return h('div', { class: 'stack' },
    h('div', { class: 'readout' },
      cell('Expanded', fmt(u.expanded_mm, 2), `mm (k=${u.k})`),
      cell('Repeatability', fmt(u.u_random_mm, 2), 'mm 1σ'),
      cell('Bias bound', fmt(u.bias_bound_mm, 2), 'mm rss'),
      cell('Worst case', fmt(u.bias_linear_mm + u.k * u.u_random_mm, 2), 'mm sum')),
    h('table', { class: 'budget' },
      h('thead', null, h('tr', null, h('th', null, 'Contribution'), h('th', { style: { textAlign: 'right' } }, 'At target'))),
      h('tbody', null, rows)),
    u.notes && u.notes.length ? h('p', { class: 'hint' }, u.notes.join(' · ')) : null,
    h('p', { class: 'hint' }, 'Repeatability and bias are reported separately. No probability is attached to the expanded value; its coverage has only been examined on synthetic images.'));
}

export function cell(k, v, u) {
  return h('div', { class: 'cell' }, h('div', { class: 'k' }, k), h('div', null, h('span', { class: 'v' }, v), u ? h('span', { class: 'u' }, u) : null));
}

export function provenanceChip(p) {
  const map = { measured: ['measured', 'ok'], manual: ['manual', 'warn'], configured: ['configured', 'info'], nominal: ['nominal', 'warn'], inferred: ['inferred', 'warn'], simulated: ['simulated', 'info'], unknown: ['unknown', ''] };
  const [t, k] = map[p] || [p, ''];
  return chip(t, k);
}

// Load the picture and overlay of one frame into a viewer.
//   view = 'image'  -> the photo, overlay in frame pixels
//   view = 'rect'   -> the rectified plane view, overlay in rectified pixels
export async function showFrame(viewer, frameId, view, { uncertainty = false, maxEdge = 2600, placards = null, candidates = false, references = false } = {}) {
  const sid = state.session.id;
  const q = `frame=${encodeURIComponent(frameId)}&view=${view}&uncertainty=${uncertainty ? 'true' : 'false'}${placards ? `&placards=${encodeURIComponent(placards.join(','))}` : ''}${candidates ? '&candidates=true' : ''}${references ? '&references=true' : ''}`;
  if (view === 'rect') {
    // The picture and its overlay must come from the *same* rectification:
    // both requests name the same size and the returned specs are compared.
    const [{ blob, headers }, overlay] = await Promise.all([
      fetchBlob(`api/sessions/${sid}/frames/${frameId}/rectified?max=${RECT_MAX}`),
      api('GET', `api/sessions/${sid}/overlay?${q}&rect_max=${RECT_MAX}`),
    ]);
    const spec = JSON.parse(headers.get('X-Lumen-Rect'));
    if (!sameRect(spec, overlay.rect)) {
      throw Object.assign(new Error('The rectified picture and its overlay do not match; reload the view.'), {});
    }
    const bmp = await loadImage(blob);
    viewer.setFrame(bmp, spec.width, spec.height);
    viewer.setOverlay(overlay);
    return { overlay, rect: spec };
  }
  const frame = state.session.frames.find((f) => f.id === frameId);
  const [{ blob }, overlay] = await Promise.all([
    fetchBlob(`api/sessions/${sid}/frames/${frameId}/image?max=${maxEdge}`),
    api('GET', `api/sessions/${sid}/overlay?${q}`),
  ]);
  const bmp = await loadImage(blob);
  viewer.setFrame(bmp, frame.width, frame.height);
  viewer.setOverlay(overlay);
  return { overlay, rect: null };
}

export const RECT_MAX = 1600;

export function sameRect(a, b) {
  if (!a || !b) return false;
  const close = (x, y) => Math.abs(x - y) <= 1e-9 * Math.max(1, Math.abs(x), Math.abs(y));
  return a.width === b.width && a.height === b.height && close(a.mm_per_px, b.mm_per_px)
    && close(a.origin[0], b.origin[0]) && close(a.origin[1], b.origin[1])
    && close(a.e_right[0], b.e_right[0]) && close(a.e_right[1], b.e_right[1])
    && close(a.e_up[0], b.e_up[0]) && close(a.e_up[1], b.e_up[1]);
}

export function viewerShell({ label }) {
  const wrap = h('div', { class: 'viewer-wrap' });
  const hud = h('div', { class: 'viewer-hud' });
  const tools = h('div', { class: 'viewer-tools' });
  const mark = h('div', { class: 'viewer-watermark', 'aria-hidden': 'true' }, 'Research only · not acceptance');
  wrap.append(hud, tools, mark);
  return { wrap, hud, tools, label };
}

// Let the picture take the whole screen area of the step (panel hidden) and back.
export function maximizeButton(viewer, splitEl) {
  const btn = h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Enlarge the picture', 'aria-pressed': 'false', 'data-action': 'maximize' }, icon('maximize'));
  btn.addEventListener('click', () => {
    const on = !splitEl.classList.contains('max');
    splitEl.classList.toggle('max', on);
    btn.setAttribute('aria-pressed', String(on));
    btn.setAttribute('aria-label', on ? 'Show the details panel' : 'Enlarge the picture');
    btn.replaceChildren(icon(on ? 'minimize' : 'maximize'));
    viewer.userMoved = false;
    requestAnimationFrame(() => viewer._resize());
  });
  return btn;
}

export function zoomButtons(viewer) {
  return [
    h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Zoom in', onclick: () => { const r = viewer._box(); viewer.zoomAt(r.width / 2, r.height / 2, 1.6); } }, icon('plus')),
    h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Zoom out', onclick: () => { const r = viewer._box(); viewer.zoomAt(r.width / 2, r.height / 2, 1 / 1.6); } }, icon('minus')),
    h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Fit to screen', onclick: () => viewer.fit() }, icon('zoomfit')),
  ];
}

export function frameQualityChips(f) {
  const out = [];
  const acc = f.markers.filter((m) => m.accepted).length;
  const rej = f.markers.length - acc;
  out.push(chip(`${acc} tag${acc === 1 ? '' : 's'}`, acc >= 2 ? 'ok' : acc === 1 ? 'warn' : 'danger'));
  if (rej) out.push(chip(`${rej} rejected`, 'warn'));
  if (f.quality.sharpness < 40) out.push(chip('soft', 'warn'));
  if (f.quality.clipped_high > 0.05) out.push(chip('glare', 'warn'));
  if (f.calibration_id) out.push(chip('calibrated', 'info'));
  return out;
}

export function targetReadout(p) {
  if (!p) return null;
  return h('div', { class: 'readout' },
    cell('x', p.x_mm === null ? '—' : lengthOut(p.x_mm).split(' ')[0], state.units),
    cell('z', p.z_mm === null ? '—' : lengthOut(p.z_mm).split(' ')[0], state.units),
    cell('Size', p.width_mm === null ? '—' : `${lengthOut(p.width_mm, 0).split(' ')[0]}×${lengthOut(p.height_mm, 0).split(' ')[0]}`, state.units),
    cell('Rotation', p.angle_rad === null ? '—' : fmt((p.angle_rad * 180) / Math.PI, 1), '°'),
    cell('Reference', (p.reference_point || '—').replace('_', ' '), ''),
    cell('Tolerance', p.tol_position_mm === null ? 'none' : `±${lengthOut(p.tol_position_mm, 1).split(' ')[0]}`, p.tol_position_mm === null ? '' : state.units));
}

// One-line capture coaching: an arrow for where to move, and what to do next.
const ARROW_ICON = { left: 'go_left', right: 'go_right', up: 'go_up', down: 'go_down', closer: 'go_closer', back: 'go_back' };
const ARROW_WORD = { left: 'move left', right: 'move right', up: 'tilt up', down: 'tilt down', closer: 'move closer', back: 'move back' };
export function coachLine(advice) {
  const n = advice && advice.next;
  if (!n) return null;
  const kind = n.level === 'ok' ? 'ok' : n.level === 'warn' ? 'warn' : n.level === 'info' ? 'info' : 'todo';
  return h('div', { class: `coach ${kind}`, 'data-coach': n.code, 'data-arrow': n.arrow || '' },
    n.arrow && ARROW_ICON[n.arrow] ? h('span', { class: 'coach-arrow', role: 'img', 'aria-label': ARROW_WORD[n.arrow] }, icon(ARROW_ICON[n.arrow])) : icon(kind === 'ok' ? 'check' : kind === 'warn' ? 'alert' : 'info'),
    h('span', { class: 'grow' }, n.text));
}

export function coachList(advice) {
  if (!advice) return null;
  const rows = advice.items.map((i) => h('li', { class: i.level === 'todo' ? 'todo' : i.level === 'warn' ? 'todo' : 'done', 'data-coach': i.code },
    i.arrow && ARROW_ICON[i.arrow] ? icon(ARROW_ICON[i.arrow]) : icon(i.level === 'info' ? 'info' : 'alert'), h('span', null, i.text)));
  return rows.length ? h('ul', { class: 'checklist' }, rows) : null;
}

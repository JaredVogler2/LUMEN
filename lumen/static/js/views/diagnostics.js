// Diagnostics sheet: the raw numbers behind the current session.
import { h, sheet, chip, banner, fmt } from '../dom.js';
import { state } from '../store.js';

export function openDiagnostics() {
  const s = state.session;
  if (!s) { sheet({ title: 'Diagnostics', content: banner('info', 'No session yet.') }); return; }
  const reg = s.registration;
  const rows = [];
  s.frames.forEach((f) => f.markers.forEach((m) => rows.push(h('tr', null,
    h('td', null, `#${f.index}`), h('td', { class: 'mono' }, `ID ${m.tag_id}`),
    h('td', null, m.accepted ? chip('ok', 'ok') : chip(m.reject_reasons.join(', ') || 'rejected', 'danger')),
    h('td', { class: 'v' }, `${fmt(Math.min(...m.edge_px), 0)} px`),
    h('td', { class: 'v' }, fmt(m.corner_sigma_px, 3)),
    h('td', { class: 'v' }, fmt(m.sigma_blur_px, 2)),
    h('td', { class: 'v' }, m.bias_applied ? fmt(m.bias_px, 3) : '—')))));
  const content = h('div', { class: 'stack' },
    h('div', { class: 'row wrap' }, chip(s.mode === 'demo' ? 'simulated demo' : 'real mode', s.mode === 'demo' ? 'info' : 'solid'), chip(s.algorithm_version, 'solid'), chip(`${s.frames.length} frames`, 'solid')),
    reg ? h('div', { class: 'card tight' }, h('div', { class: 'eyebrow' }, 'Registration'),
      h('dl', { class: 'kv' },
        h('dt', null, 'Tags in layout'), h('dd', null, String(reg.n_tags)),
        h('dt', null, 'Photos registered'), h('dd', null, String(reg.frames_used.length)),
        h('dt', null, 'RMS reprojection'), h('dd', null, `${fmt(reg.rms_px, 3)} px`),
        h('dt', null, 'Reduced χ²'), h('dd', null, `${fmt(reg.reduced_chi2, 2)} (dof ${reg.dof})`),
        h('dt', null, 'Gauge tag'), h('dd', null, `ID ${reg.layout.gauge_id}`),
        h('dt', null, 'Flags'), h('dd', null, (reg.flags || []).join(', ') || 'none'))) : banner('info', 'Not registered yet.'),
    h('div', { class: 'eyebrow' }, 'Tag observations'),
    h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'budget' },
      h('thead', null, h('tr', null, ['Photo', 'Tag', 'Status', 'Edge', 'σ corner px', 'Blur px', 'Edge bias px'].map((t) => h('th', null, t)))),
      h('tbody', null, rows.length ? rows : h('tr', null, h('td', { colspan: '7' }, 'No tag observations.'))))),
    h('p', { class: 'hint' }, 'σ corner is the predicted standard deviation of a tag corner from the edge fits. Edge bias is the dark-side shift estimated from the tag interior and removed from its border.'));
  sheet({ title: 'Diagnostics', content, wide: true });
}

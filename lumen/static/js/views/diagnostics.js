// Diagnostics sheet: the raw numbers behind the current session.
import { api, jobIsCarried } from '../api.js';
import * as carry from '../carry.js';
import { h, clear, sheet, chip, banner, fmt, icon, showError } from '../dom.js';
import { state, activeFrameId } from '../store.js';

// Optional assistant (only when the administrator enabled it): explains the checks of this session in words.
// It is sent the computed report as text — no photo, no session id — and its reply is checked by the server
// before it is shown.  Nothing it says changes a measurement or a status.
function assistantCard() {
  if (!(state.config && state.config.features && state.config.features.assistant)) return null;
  const out = h('div', { class: 'stack', 'aria-live': 'polite', 'data-region': 'assistant-reply' });
  const q = h('input', { class: 'input', type: 'text', maxlength: '400', placeholder: 'e.g. Why is this only a preview?', 'aria-label': 'Question for the assistant', 'data-field': 'assistant-question' });
  const ask = async (kind) => {
    clear(out).append(h('div', { class: 'spinner', style: { alignSelf: 'center' } }));
    try {
      const r = await api('POST', `api/sessions/${state.session.id}/assistant`, { kind, question: q.value.trim(), frame_id: activeFrameId() || null }, { timeout: 60000 });
      clear(out).append(
        banner('info', r.mock ? 'Assistant text (mock adapter, for testing)' : 'Assistant text', r.label),
        r.refusal ? banner('warn', 'The assistant could not answer from the report', r.refusal) : h('p', null, r.answer),
        r.steps && r.steps.length ? h('ol', { class: 'hint', style: { margin: '0', paddingLeft: '20px' } }, r.steps.map((t) => h('li', null, t))) : null,
        r.citations && r.citations.length ? h('details', { class: 'fold' }, h('summary', null, `Quoted from the report (${r.citations.length})`),
          h('div', { class: 'body' }, r.citations.map((c) => h('p', { class: 'hint' }, h('span', { class: 'mono' }, `${c.source}: `), `“${c.quote}”`)))) : null,
        h('p', { class: 'hint' }, `Sent: the report as text (${r.sent.fields.join(', ')}). No photo and no session ID were sent.`));
    } catch (err) { clear(out); showError(err, 'The assistant did not answer'); }
  };
  return h('div', { class: 'card tight', 'data-region': 'assistant' },
    h('div', { class: 'eyebrow' }, 'Assistant (optional)'),
    h('p', { class: 'hint' }, 'Explains the checks of this session in words. Its text is not a measurement and not an acceptance decision.'),
    q,
    h('div', { class: 'row wrap' },
      h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'assistant-explain', onclick: () => ask('explain_capture') }, icon('help'), 'Explain the status'),
      h('button', { class: 'btn sm grow', type: 'button', 'data-action': 'assistant-summary', onclick: () => ask('summarize_diagnostics') }, icon('layers'), 'Summarise the diagnostics')),
    out);
}

// Where the browser carries the job: what it holds, and how the last requests went (how much each sent, how long
// the answer took).  This is what tells a slow connection from a slow server.
function carrierCard() {
  if (!jobIsCarried()) return null;
  const box = h('div', { class: 'card tight', 'data-region': 'carrier' },
    h('div', { class: 'eyebrow' }, 'Where the job is kept'),
    h('p', { class: 'hint' }, 'This server keeps nothing between two requests. The job (its photos and what was measured in them) is on this phone and travels with each request; it is removed when the session ends.'));
  carry.held().then((jobs) => {
    const pics = jobs.reduce((n, j) => n + j.pictures.length, 0);
    const bytes = jobs.reduce((n, j) => n + j.state_bytes + j.pictures.reduce((m, x) => m + x.bytes, 0), 0);
    const last = carry.recent.slice(-8).reverse();
    box.append(
      h('dl', { class: 'kv' },
        h('dt', null, 'On this phone'), h('dd', null, `${pics} photo${pics === 1 ? '' : 's'}, ${fmt(bytes / 1e6, 1)} MB`),
        h('dt', null, 'Requests'), h('dd', null, `${carry.stats.requests}, ${fmt(carry.stats.bytesSent / 1e6, 1)} MB sent`),
        h('dt', null, 'Photos sent again'), h('dd', null, `${carry.stats.picturesSent}${carry.stats.askedAgain ? ` (the server asked ${carry.stats.askedAgain} time${carry.stats.askedAgain === 1 ? '' : 's'})` : ''}`)),
      last.length ? h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'budget', 'data-region': 'carrier-recent' },
        h('thead', null, h('tr', null, ['Request', 'Sent', 'Answer after', 'Result'].map((t) => h('th', null, t)))),
        h('tbody', null, last.map((r) => h('tr', null,
          h('td', { class: 'mono' }, `${r.method} ${r.path.replace(/^api\/(sessions|calibration)\/…\/?/, '') || r.path}`),
          h('td', { class: 'v' }, r.bytes > 1e5 ? `${fmt(r.bytes / 1e6, 2)} MB` : `${fmt(r.bytes / 1e3, 0)} kB`),
          h('td', { class: 'v' }, r.ms === null ? '…' : r.ms >= 1000 ? `${fmt(r.ms / 1000, 1)} s` : `${r.ms} ms`),
          h('td', null, r.status === 200 ? chip('ok', 'ok') : chip(String(r.status === null ? 'waiting' : r.status), r.status === null ? 'solid' : 'danger'))))))) : null);
  });
  return box;
}

export function openDiagnostics() {
  const s = state.session;
  if (!s) { sheet({ title: 'Diagnostics', content: h('div', { class: 'stack' }, banner('info', 'No session yet.'), carrierCard()) }); return; }
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
        h('dt', null, 'Flags'), h('dd', null, (reg.flags || []).join(', ') || 'none'),
        reg.multiview && reg.multiview.excluded_observations && Object.keys(reg.multiview.excluded_observations).length ? h('dt', null, 'Left out') : null,
        reg.multiview && reg.multiview.excluded_observations && Object.keys(reg.multiview.excluded_observations).length
          ? h('dd', null, Object.entries(reg.multiview.excluded_observations).map(([fid, ids]) => ids.map((t) => `tag ${t} in photo ${fid}`).join(', ')).join('; ') + ' (did not fit the other photos of the same tag; the tag is kept)') : null,
        reg.edge_shift ? h('dt', null, 'Edge shift of tag borders') : null,
        reg.edge_shift ? h('dd', null, `${reg.edge_shift.own_estimate} observations with their own estimate`
          + (reg.edge_shift.pooled ? `, ${reg.edge_shift.pooled} too small and corrected with the value from the larger tags (${fmt(reg.edge_shift.pool.delta_px, 3)} ± ${fmt(reg.edge_shift.pool.sigma_px, 3)} px)` : '')
          + (reg.edge_shift.uncorrected ? `, ${reg.edge_shift.uncorrected} too small and not corrected` : '')) : null)) : banner('info', 'Not registered yet.'),
    h('div', { class: 'eyebrow' }, 'Tag observations'),
    h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'budget' },
      h('thead', null, h('tr', null, ['Photo', 'Tag', 'Status', 'Edge', 'σ corner px', 'Blur px', 'Edge bias px'].map((t) => h('th', null, t)))),
      h('tbody', null, rows.length ? rows : h('tr', null, h('td', { colspan: '7' }, 'No tag observations.'))))),
    h('p', { class: 'hint' }, 'σ corner is the predicted standard deviation of a tag corner from the edge fits. Edge bias is the dark-side shift estimated from the tag interior and removed from its border; "—" means the tag was too small for that (it is then corrected with the value from larger tags when there are any).'),
    carrierCard(),
    assistantCard());
  sheet({ title: 'Diagnostics', content, wide: true });
}

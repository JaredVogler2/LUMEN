// Step 1 — identify the operation and the placard.
import { api } from '../api.js';
import { h, clear, icon, chip, banner, toast, showError, busy } from '../dom.js';
import { state, setState, setSession, ensureSession, endSession } from '../store.js';
import { targetReadout } from './common.js';
import { openScanner } from './scan.js';
import { openOperationEditor, nextRevision, issueLabel } from './operation_editor.js';

const issueText = issueLabel;

export function mount(root, ctx) {
  const s = state.session;
  const sel = {
    operationId: s && s.operation ? s.operation.operation.operation_id : null,
    revision: s && s.operation ? s.operation.operation.revision : null,
    placardId: s ? s.active_placard_id : null,
    detail: null,
    query: '',
  };
  const list = h('div', { class: 'stack', role: 'list' });
  const detail = h('div', { class: 'stack' });
  const search = h('input', {
    class: 'input', type: 'search', placeholder: 'Operation, panel or placard ID', 'aria-label': 'Search operations',
    autocomplete: 'off', autocapitalize: 'characters', spellcheck: 'false', enterkeyhint: 'search',
    oninput: (ev) => { sel.query = ev.target.value.trim(); renderList(); },
    onkeydown: (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); resolveCode(sel.query); } },
  });

  const host = (state.config && state.config.hosting) || null;
  root.append(
    // a server that is set to keep jobs on a host that forgets them between requests: nothing would work, so it is said
    host && host.keeps_sessions === false && host.message
      ? banner('danger', 'This copy of LUMEN is set up in a way that cannot work here', host.message)
      : null,
    h('h1', { class: 'view-title' }, 'Choose the job'),
    h('p', { class: 'view-lead' }, 'Pick the operation and the placard to place, or scan its code.'),
    h('div', { class: 'row' },
      h('div', { class: 'grow' }, search),
      h('button', { class: 'btn', type: 'button', 'aria-label': 'Scan a barcode or QR code', 'data-action': 'scan', onclick: () => openScanner(resolveCode) }, icon('scan'), 'Scan')),
    list, detail, libraryCard());

  // Local records: where they are kept decides how much to rely on them.
  function libraryCard() {
    const lib = state.operations || {};
    const st = lib.storage || { editable: !!lib.editable, persistent: null };
    if (!st.editable) {
      return h('div', { class: 'card tight', 'data-region': 'library' }, h('div', { class: 'eyebrow' }, 'Operation library'),
        h('p', { class: 'hint' }, 'This deployment shows the bundled sample records only; records cannot be entered here.'));
    }
    return h('div', { class: 'card tight', 'data-region': 'library' },
      h('div', { class: 'eyebrow' }, 'Operation library'),
      h('p', { class: 'hint' }, st.persistent
        ? 'Records entered here are kept in the directory the administrator configured.'
        : 'Records entered here are kept in temporary storage on this server and can disappear when it restarts or is redeployed. Export a record you want to keep.'),
      h('button', { class: 'btn', type: 'button', 'data-action': 'op-new', onclick: () => edit(null, 'new') }, icon('plus'), 'Enter a new record'));
  }

  async function reloadLibrary() {
    try { setState({ operations: await api('GET', 'api/operations') }); } catch (err) { showError(err); }
  }

  function edit(record, mode) {
    const lib = (state.operations && state.operations.operations) || [];
    let rec = record ? JSON.parse(JSON.stringify(record)) : null;
    if (rec && mode === 'copy') {
      rec = { ...rec, operation_id: '', revision: 'A', status: 'draft', provenance: { ...rec.provenance, source: '', notes: `Copied from ${record.operation_id} rev ${record.revision}; every value is to be checked against its source.` } };
      rec.panel = { ...rec.panel, barcodes: [] };
      rec.placards = rec.placards.map((p) => ({ ...p, barcodes: [], tolerance: { ...p.tolerance, provenance: '' } }));
    } else if (rec && mode === 'revision') {
      rec = { ...rec, revision: nextRevision(rec.revision, lib.filter((o) => o.operation_id === rec.operation_id).map((o) => o.revision)), status: 'draft' };
    }
    openOperationEditor({ record: rec, mode, library: lib, onSaved: async (summary) => {
      await reloadLibrary();
      renderList();
      if (summary) await choose(summary.operation_id, summary.revision);
      else { sel.operationId = null; sel.revision = null; sel.detail = null; clear(detail); renderList(); updateActions(); }
    } });
  }

  if (state.config && state.config.features.demo && !(s && s.frames.length)) {
    root.append(h('div', { class: 'card' },
      h('div', { class: 'eyebrow' }, 'No panel at hand?'),
      h('h3', null, 'Try it with a simulated demo'),
      h('p', { class: 'hint' }, 'Computer-generated photos of an invented panel, measured for real. Marked as simulated throughout.'),
      h('div', { class: 'row wrap' },
        h('button', { class: 'btn grow', type: 'button', 'data-action': 'demo', onclick: () => loadDemo('flat') }, icon('play'), 'Open the demo'),
        h('button', { class: 'btn grow', type: 'button', 'data-action': 'demo-curve', onclick: () => loadDemo('curve') }, icon('play'), 'Rounded edge'))));
  }

  function renderList() {
    clear(list);
    const ops = (state.operations && state.operations.operations) || [];
    const q = sel.query.toLowerCase();
    const shown = ops.filter((o) => !q || [o.operation_id, o.title, o.panel_id, ...o.placard_ids].some((t) => String(t).toLowerCase().includes(q)));
    if (!shown.length) {
      list.append(banner('info', 'No operation matches.', 'Check the ID, or scan the code on the work order. LUMEN only selects exact matches.'));
      return;
    }
    shown.forEach((o) => {
      const card = h('button', {
        class: `card selectable${sel.operationId === o.operation_id && sel.revision === o.revision ? ' selected' : ''}`,
        type: 'button', role: 'listitem', 'data-operation': o.operation_id, 'data-revision': o.revision, 'aria-pressed': sel.operationId === o.operation_id && sel.revision === o.revision ? 'true' : 'false',
        onclick: () => choose(o.operation_id, o.revision),
      },
        h('div', { class: 'row wrap' },
          h('span', { class: 'mono', style: { fontWeight: 800, fontSize: '16px' } }, o.operation_id),
          chip(`rev ${o.revision}`, 'solid'),
          chip(o.status, o.status === 'released' ? 'ok' : o.status === 'superseded' ? 'solid' : 'warn'),
          o.synthetic ? chip('synthetic sample', 'info') : null,
          o.origin === 'local' ? chip('entered here', 'solid') : null,
          o.complete ? null : chip('incomplete', 'danger')),
        h('div', null, o.title),
        h('div', { class: 'hint' }, `Panel ${o.panel_id} · ${o.n_placards} placard${o.n_placards === 1 ? '' : 's'}`));
      list.append(card);
    });
  }

  async function choose(operationId, revision, placardId = null) {
    sel.operationId = operationId; sel.revision = revision;
    renderList();
    clear(detail).append(h('div', { class: 'card' }, h('div', { class: 'spinner', style: { alignSelf: 'center' } })));
    try {
      const d = await api('GET', `api/operations/${encodeURIComponent(operationId)}?revision=${encodeURIComponent(revision)}`);
      sel.detail = d;
      const ids = d.operation.placards.map((p) => p.placard_id);
      sel.placardId = placardId && ids.includes(placardId) ? placardId : (ids.includes(sel.placardId) ? sel.placardId : ids[0] || null);
      renderDetail();
    } catch (err) { showError(err); clear(detail); }
    updateActions();
  }

  function renderDetail() {
    clear(detail);
    const d = sel.detail;
    if (!d) return;
    const op = d.operation;
    const issues = d.summary.issues || {};
    const dat = op.datums || {};
    const datumCard = (key, x) => {
      if (!x) return h('div', { class: 'card tight' }, h('div', { class: 'eyebrow' }, `${key} datum`), banner('danger', 'Not defined in this record.'));
      return h('div', { class: 'card tight' },
        h('div', { class: 'row wrap' }, h('span', { class: 'eyebrow' }, `Datum ${x.id} · ${key}`), chip(x.feature ? x.feature.replace(/_/g, ' ') : 'feature not stated', x.feature ? 'solid' : 'danger')),
        h('div', { style: { fontWeight: 700 } }, x.name),
        h('div', { class: 'hint' },
          x.positive ? `+${key === 'floor' ? 'z' : 'x'} runs ${x.positive.replace(/_/g, ' ')}` : 'Sign not defined',
          x.on_target_plane === true ? ' · on the target plane' : x.on_target_plane === false ? ' · off the target plane' : ' · plane relation not stated',
          x.side_of_panel ? ` · ${x.side_of_panel} side when facing the panel` : '',
          x.radius ? ` · radius ${x.radius.value} ${x.radius.unit}` : ''));
    };
    detail.append(
      h('h2', { class: 'view-title', style: { fontSize: '18px' } }, op.title || op.operation_id),
      Object.keys(issues).length
        ? banner('warn', 'This record is incomplete.', 'It can be looked at, but it cannot reach “Ready”.',
          Object.entries(issues).map(([scope, lst]) => `${scope}: ${lst.map(issueText).join(', ')}`))
        : null,
      h('div', { class: 'stack' },
        datumCard('floor', dat.floor), datumCard('side', dat.side),
        h('div', { class: 'card tight' }, h('div', { class: 'eyebrow' }, 'Axes'),
          h('div', { class: 'hint' }, `${dat.axis_policy ? dat.axis_policy.replace(/_/g, ' ') : 'policy not defined'}${dat.nominal_angle ? ` · nominal ${dat.nominal_angle.value}° between datums` : ''}${dat.max_axis_disagreement ? ` · limit ±${dat.max_axis_disagreement.value}°` : ''}`))),
      h('div', { class: 'eyebrow' }, 'Placard'),
      h('div', { class: 'stack', role: 'radiogroup', 'aria-label': 'Placard' }, d.placards_resolved.map((p) => {
        const pi = issues[`placard:${p.placard_id}`] || [];
        return h('button', {
          class: `card selectable${sel.placardId === p.placard_id ? ' selected' : ''}`, type: 'button', role: 'radio',
          'aria-checked': sel.placardId === p.placard_id ? 'true' : 'false', 'data-placard': p.placard_id,
          onclick: () => { sel.placardId = p.placard_id; renderDetail(); updateActions(); },
        },
          h('div', { class: 'row wrap' }, h('span', { class: 'mono', style: { fontWeight: 800 } }, p.placard_id), pi.length ? chip(pi.map(issueText).join(', '), 'danger') : chip('complete', 'ok')),
          h('div', null, p.title || ''),
          sel.placardId === p.placard_id ? targetReadout(p) : null);
      })),
      h('p', { class: 'hint' }, `Source: ${op.provenance.source || 'not stated'}${op.provenance.notes ? ` — ${op.provenance.notes}` : ''}`),
      editButtons(d));
  }

  function editButtons(d) {
    const lib = state.operations || {};
    if (!(lib.storage ? lib.storage.editable : lib.editable)) return null;
    const op = d.operation;
    const local = d.origin !== 'bundled';
    return h('div', { class: 'row wrap', 'data-region': 'record-actions' },
      local && op.status === 'draft' ? h('button', { class: 'btn sm', type: 'button', 'data-action': 'op-edit', onclick: () => edit(op, 'edit') }, icon('edit'), 'Edit this draft') : null,
      local && op.status !== 'draft' ? h('button', { class: 'btn sm', type: 'button', 'data-action': 'op-revise', onclick: () => edit(op, 'revision') }, icon('edit'), 'New revision') : null,
      h('button', { class: 'btn sm', type: 'button', 'data-action': 'op-copy', onclick: () => edit(op, 'copy') }, icon('copy'), 'Copy to a new record'));
  }

  async function resolveCode(code) {
    if (!code) return;
    try {
      const r = await api('POST', 'api/operations/resolve', { code });
      if (r.status === 'exact') {
        const m = r.matches[0];
        const pm = r.matches.find((x) => x.placard_id);
        toast({ title: 'Exact match', detail: `${m.operation_id} rev ${m.revision}${m.status === 'released' ? '' : ` (${m.status})`}`, kind: 'ok', timeout: 3000 });
        // the code may be a barcode text that the list filter does not know: show the list with the match selected
        sel.query = ''; search.value = '';
        await choose(m.operation_id, m.revision, pm ? pm.placard_id : null);
      } else if (r.status === 'ambiguous') {
        const uniq = [...new Map(r.matches.map((m) => [`${m.operation_id}|${m.revision}`, m])).values()];
        toast({ title: 'More than one record matches', detail: `Choose one: ${uniq.map((m) => `${m.operation_id} rev ${m.revision}`).join(', ')}`, kind: 'error', timeout: 9000 });
        sel.query = code; search.value = code; renderList();
      } else {
        const near = r.near.slice(0, 3).map((m) => m.value).join(', ');
        toast({ title: 'No exact match', detail: near ? `Similar, not selected automatically: ${near}` : 'Nothing in the library has this ID.', kind: 'error', timeout: 9000 });
      }
    } catch (err) { showError(err); }
  }

  async function confirm() {
    const b = busy('Selecting operation…');
    try {
      const sess = await ensureSession('real');
      const out = await api('PUT', `api/sessions/${sess.id}/operation`, { operation_id: sel.operationId, revision: sel.revision, placard_id: sel.placardId });
      setSession(out);
      ctx.go('markers');
    } catch (err) { showError(err); } finally { b.done(); }
  }

  async function loadDemo(scenario = 'flat') {
    const b = busy('Generating the demo session…');
    try {
      if (state.session) await endSession();
      const sess = await api('POST', 'api/sessions', { mode: 'demo' });
      setSession(sess);
      const out = await api('POST', `api/sessions/${sess.id}/demo/load`, { scenario }, { timeout: 180000 });
      setSession(out);
      toast({ title: 'Demo session loaded', detail: 'Simulated photos of an invented panel.', kind: 'ok' });
      setState({});
      ctx.go('capture');
    } catch (err) { showError(err); } finally { b.done(); }
  }

  function updateActions() {
    ctx.setActions([
      { id: 'continue', label: 'Continue', kind: 'primary lg', icon: 'arrow', disabled: !(sel.operationId && sel.placardId), onClick: confirm },
    ]);
  }

  renderList();
  updateActions();
  if (sel.operationId) choose(sel.operationId, sel.revision, sel.placardId);
  return {};
}

// Operation record editor (manual entry of the operation library).
//
// Rules the form keeps:
//  * an empty field stays empty ("not stated") — nothing is defaulted, so a
//    missing tolerance or datum sign shows up as an issue instead of a guess;
//  * every length keeps the unit it was entered in; the server converts once
//    and the preview shows the millimetre values the measurement will use;
//  * a released revision cannot be changed: a change is a new revision.
import { api } from '../api.js';
import { h, clear, icon, sheet, banner, chip, toast, showError, busy, fmt } from '../dom.js';

const FEATURES = [['planar_edge', 'Visible line on the panel face'], ['curve_tangent', 'Where the flat face starts to curve'], ['curve_outer_extreme', 'Outermost line of a rounded edge']];
const DIRECTIONS = [['toward_panel', 'grows into the panel'], ['away_from_panel', 'grows away from the panel']];
const POLICIES = [['floor_primary', 'Floor datum sets the axes'], ['side_primary', 'Side datum sets the axes'], ['perpendicular_offsets', 'Each offset perpendicular to its own datum']];
const REF_POINTS = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right', 'bottom_center', 'top_center', 'left_center', 'right_center'];
const ISSUE = {
  missing_x_offset: 'no x offset', missing_z_offset: 'no z offset', missing_reference_point: 'no reference point', invalid_reference_point: 'unknown reference point',
  missing_x_semantics: 'not stated whether x is measured in the plane or along the surface', missing_size: 'no size', invalid_size: 'size must be positive', implausible_size: 'size over 2 m',
  missing_rotation: 'no rotation', missing_position_tolerance: 'no position tolerance', invalid_position_tolerance: 'position tolerance must be positive',
  missing_rotation_tolerance: 'no rotation tolerance', invalid_rotation_tolerance: 'rotation tolerance must be positive', duplicate_placard_id: 'placard ID used twice',
  missing_floor_datum: 'no floor datum', missing_side_datum: 'no side datum', floor_datum_feature_missing: 'floor datum: feature not stated', side_datum_feature_missing: 'side datum: feature not stated',
  floor_datum_plane_relation_missing: 'floor datum: not stated whether it lies on the panel face', side_datum_plane_relation_missing: 'side datum: not stated whether it lies on the panel face',
  floor_datum_sign_missing: 'floor datum: direction of +z not stated', side_datum_sign_missing: 'side datum: direction of +x not stated',
  axis_policy_missing: 'axis rule not stated', nominal_datum_angle_missing: 'nominal angle between the datums not stated', max_axis_disagreement_missing: 'allowed deviation from the nominal angle not stated',
  no_placards: 'no placard', curved_target_experimental: 'curved target surface (experimental, no guidance)',
};
export const issueLabel = (i) => ISSUE[i] || String(i).replace(/_/g, ' ');

const today = () => new Date().toISOString().slice(0, 10);

export function blankRecord() {
  return {
    operation_id: '', title: '', revision: 'A', status: 'draft', synthetic: false,
    effectivity: { configuration: '', notes: '' },
    provenance: { source: '', author: '', date: today(), notes: '' },
    panel: { panel_id: '', description: '', barcodes: [] },
    surface: { target: 'plane', notes: '' },
    datums: { floor: null, side: null, axis_policy: null, nominal_angle: null, max_axis_disagreement: null },
    placards: [blankPlacard()],
  };
}
function blankPlacard() {
  return { placard_id: '', title: '', barcodes: [], position: { x: null, z: null, reference_point: null, x_semantics: null }, size: { width: null, height: null }, rotation: null, tolerance: { position: null, rotation: null, provenance: '' }, notes: '' };
}
function blankDatum(key) {
  return { id: key === 'floor' ? 'A' : 'B', name: '', feature: null, on_target_plane: null, positive: null, offset: null, radius: null, side_of_panel: null, notes: '' };
}

export function nextRevision(rev, taken) {
  let r = String(rev || 'A');
  for (let i = 0; i < 60; i += 1) {
    if (/^[A-Y]$/.test(r)) r = String.fromCharCode(r.charCodeAt(0) + 1);
    else if (/^\d+$/.test(r)) r = String(Number(r) + 1);
    else { const m = r.match(/^(.*?)(\d+)$/); r = m ? `${m[1]}${Number(m[2]) + 1}` : `${r}2`; }
    if (!taken.includes(r)) return r;
  }
  return `${rev}-new`;
}

// What the server receives: a datum nobody described is "no datum", not an empty one.
function buildRecord(rec) {
  const out = JSON.parse(JSON.stringify(rec));
  ['floor', 'side'].forEach((k) => {
    const d = out.datums[k];
    if (d && !d.name.trim() && !d.feature && d.on_target_plane === null && !d.positive && !d.offset && !d.radius && !d.side_of_panel && !d.notes.trim()) out.datums[k] = null;
    else if (d && k === 'floor') d.side_of_panel = null;
  });
  out.synthetic = false;
  return out;
}

export function openOperationEditor({ record = null, mode = 'new', library = [], onSaved = () => {} } = {}) {
  const rec = record ? JSON.parse(JSON.stringify(record)) : blankRecord();
  ['floor', 'side'].forEach((k) => { if (!rec.datums[k]) rec.datums[k] = blankDatum(k); });
  rec.synthetic = false;
  const original = record ? { id: record.operation_id, revision: record.revision, status: record.status } : null;
  const content = h('div', { class: 'stack op-editor', 'data-region': 'operation-editor' });
  const title = mode === 'edit' ? 'Edit draft record' : mode === 'revision' ? 'New revision' : mode === 'copy' ? 'Copy to a new record' : 'New operation record';
  const ref = sheet({ title, content, wide: true });
  const status = h('div', { class: 'stack', 'data-region': 'editor-status', 'aria-live': 'polite' });
  const actions = h('div', { class: 'stack' });
  let check = null;           // last answer of the server about the record as typed
  let checkError = null;
  let timer = null;
  let seq = 0;
  let jsonMode = false;
  let confirmRelease = false;
  let supersedeOld = true;

  const changed = () => { confirmRelease = false; clearTimeout(timer); timer = setTimeout(validate, 350); };

  // ---------------------------------------------------------------- fields
  const field = (label, control, hint) => h('label', { class: 'field grow' }, h('span', { class: 'label' }, label), control, hint ? h('span', { class: 'hint' }, hint) : null);
  const text = (label, obj, key, attrs = {}) => field(label, h('input', { class: 'input', type: 'text', value: obj[key] || '', autocomplete: 'off', spellcheck: 'false', 'data-field': attrs.name || key, ...attrs,
    oninput: (ev) => { obj[key] = ev.target.value; changed(); } }), attrs.hint);
  const list = (label, obj, key, attrs = {}) => field(label, h('input', { class: 'input', type: 'text', value: (obj[key] || []).join(', '), autocomplete: 'off', spellcheck: 'false', 'data-field': attrs.name || key, placeholder: 'separate several with commas',
    oninput: (ev) => { obj[key] = ev.target.value.split(',').map((x) => x.trim()).filter(Boolean); changed(); } }), attrs.hint);
  const choice = (label, obj, key, options, attrs = {}) => field(label, h('select', { class: 'input', 'data-field': attrs.name || key,
    onchange: (ev) => { const v = ev.target.value; obj[key] = v === '' ? null : v === 'true' ? true : v === 'false' ? false : v; changed(); if (attrs.rerender) render(); } },
  h('option', { value: '', selected: obj[key] === null || obj[key] === undefined ? '' : null }, attrs.empty || '— not stated —'),
  options.map(([v, l]) => h('option', { value: String(v), selected: obj[key] === v ? '' : null }, l))), attrs.hint);
  // a quantity is {value, unit} or null; the unit is kept as entered
  const quantity = (label, obj, key, baseUnits, attrs = {}) => {
    const cur = obj[key];
    // a record may carry a unit the form does not offer (cm, ft, rad ...): it is kept, never replaced by the first option
    const units = cur && !baseUnits.includes(cur.unit) ? [cur.unit, ...baseUnits] : baseUnits;
    const unit = h('select', { class: 'input', 'aria-label': `Unit of ${label}`, 'data-unit': attrs.name || key,
      onchange: () => { if (obj[key]) { obj[key].unit = unit.value; changed(); } } }, units.map((u) => h('option', { value: u, selected: (cur ? cur.unit : units[0]) === u ? '' : null }, u === 'deg' ? '°' : u)));
    const input = h('input', { class: 'input num', type: 'number', inputmode: 'decimal', step: 'any', value: cur ? String(cur.value) : '', placeholder: 'not stated', 'data-field': attrs.name || key,
      oninput: () => { const v = input.value.trim(); obj[key] = v === '' || Number.isNaN(Number(v)) ? null : { value: Number(v), unit: unit.value }; changed(); } });
    return field(label, h('div', { class: 'qty' }, input, unit), attrs.hint);
  };
  const lengthQ = (label, obj, key, attrs) => quantity(label, obj, key, ['mm', 'in'], attrs);
  const angleQ = (label, obj, key, attrs) => quantity(label, obj, key, ['deg'], attrs);
  // Sections stay open or closed as the operator left them when the form is rebuilt.  Their state is read from
  // the page right before rebuilding (the browser reports a toggle only later, which can be after the rebuild).
  const openSections = new Set(['record', `placard-${rec.placards.length - 1}`]);
  const rememberOpen = () => content.querySelectorAll('details[data-section]').forEach((d) => {
    const key = d.getAttribute('data-section');
    if (d.open) openSections.add(key); else openSections.delete(key);
  });
  const fold = (key, titleText, body, attrs = {}) => h('details', { class: 'fold', open: openSections.has(key) ? '' : null, 'data-section': key, ...attrs },
    h('summary', null, titleText), h('div', { class: 'body' }, body));

  function datumFields(key) {
    const d = rec.datums[key];
    const axis = key === 'floor' ? 'z' : 'x';
    return [
      h('div', { class: 'row wrap' }, text('Datum letter', d, 'id', { name: `${key}.id`, style: { maxWidth: '110px' } }), text(key === 'floor' ? 'What realises the floor datum' : 'What realises the side datum', d, 'name', { name: `${key}.name`, placeholder: key === 'floor' ? 'e.g. line where the panel meets the floor' : 'e.g. left joint seam' })),
      choice('Physical feature', d, 'feature', FEATURES, { name: `${key}.feature`, rerender: true, hint: 'A seam, joint or floor line can be seen in one photo. A rounded edge needs several photos from different sides.' }),
      h('div', { class: 'row wrap' },
        choice('Lies on the panel face', d, 'on_target_plane', [[true, 'yes'], [false, 'no']], { name: `${key}.on_target_plane` }),
        choice(`+${axis} from this datum`, d, 'positive', DIRECTIONS, { name: `${key}.positive` })),
      h('div', { class: 'row wrap' },
        lengthQ('Offset from the feature to the datum', d, 'offset', { name: `${key}.offset`, hint: `along +${axis}; usually 0` }),
        d.feature && d.feature !== 'planar_edge' ? lengthQ('Radius of the rounded edge', d, 'radius', { name: `${key}.radius`, hint: 'only if it is known; otherwise it is measured from several photos' }) : null,
        key === 'side' ? choice('Side of the panel', d, 'side_of_panel', [['left', 'left when facing the panel'], ['right', 'right when facing the panel']], { name: 'side.side_of_panel' }) : null),
      text('Notes', d, 'notes', { name: `${key}.notes` }),
    ];
  }

  // the millimetre values the measurement will use (so a value entered in inches can be checked)
  function resolvedText(i) {
    const res = check && check.placards_resolved ? check.placards_resolved[i] : null;
    if (!res) return '';
    const n = (v, digits = 2) => (v === null || v === undefined ? '—' : fmt(v, digits));
    const deg = (v) => (v === null || v === undefined ? '—' : `${fmt((v * 180) / Math.PI, 2)}°`);
    return `As used, in mm: x ${n(res.x_mm)} · z ${n(res.z_mm)} · ${n(res.width_mm)} × ${n(res.height_mm)} · rotation ${deg(res.angle_rad)} · tolerance ±${n(res.tol_position_mm, 3)} mm, ±${deg(res.tol_rotation_rad)}`;
  }

  function placardFields(p, i) {
    return [
      h('div', { class: 'row wrap' }, text('Placard ID', p, 'placard_id', { name: `placard.${i}.placard_id`, autocapitalize: 'characters' }), text('Title', p, 'title', { name: `placard.${i}.title` })),
      list('Codes that identify this placard (barcode / QR text)', p, 'barcodes', { name: `placard.${i}.barcodes` }),
      h('div', { class: 'row wrap' }, lengthQ('x from the side datum', p.position, 'x', { name: `placard.${i}.x` }), lengthQ('z from the floor datum', p.position, 'z', { name: `placard.${i}.z` })),
      h('div', { class: 'row wrap' },
        choice('The offsets locate the placard\'s', p.position, 'reference_point', REF_POINTS.map((r) => [r, r.replace(/_/g, ' ')]), { name: `placard.${i}.reference_point` }),
        choice('x is measured', p.position, 'x_semantics', [['projected', 'in the plane of the panel face'], ['arc_length', 'along the surface (arc length)']], { name: `placard.${i}.x_semantics` })),
      h('div', { class: 'row wrap' }, lengthQ('Width', p.size, 'width', { name: `placard.${i}.width` }), lengthQ('Height', p.size, 'height', { name: `placard.${i}.height` }), angleQ('Rotation', p, 'rotation', { name: `placard.${i}.rotation`, hint: 'turns the width from +x toward +z; 0 = level' })),
      h('div', { class: 'row wrap' }, lengthQ('Position tolerance ±', p.tolerance, 'position', { name: `placard.${i}.tol_position` }), angleQ('Rotation tolerance ±', p.tolerance, 'rotation', { name: `placard.${i}.tol_rotation` })),
      text('Where the tolerance comes from', p.tolerance, 'provenance', { name: `placard.${i}.tol_provenance`, placeholder: 'document, revision, who authorised it' }),
      h('p', { class: 'hint mono', 'data-resolved': String(i) }, resolvedText(i)),
      rec.placards.length > 1 ? h('button', { class: 'btn sm', type: 'button', 'data-action': `placard-remove-${i}`, onclick: () => { rememberOpen(); rec.placards.splice(i, 1); [...openSections].filter((k) => k.startsWith('placard-')).forEach((k) => openSections.delete(k)); changed(); render(false); } }, icon('trash'), 'Remove this placard') : null,
    ];
  }

  // ---------------------------------------------------------------- server check
  async function validate() {
    const my = ++seq;
    try {
      const out = await api('POST', 'api/operations/validate', { operation: buildRecord(rec) });
      if (my !== seq) return;
      check = out; checkError = null;
    } catch (err) {
      if (my !== seq) return;
      check = null;
      checkError = err && err.lumen ? err.lumen.message : String((err && err.message) || err);
    }
    renderStatus();
    content.querySelectorAll('[data-resolved]').forEach((el) => { el.textContent = resolvedText(Number(el.getAttribute('data-resolved'))); });
  }

  const WHERE = { operation_id: 'Operation ID', revision: 'Revision', 'panel.panel_id': 'Panel ID' };
  const whereLabel = (w) => WHERE[w] || w.replace(/^placards\.(\d+)\./, (m, i) => `Placard ${Number(i) + 1}: `).replace(/[._]/g, ' ');

  function renderStatus() {
    clear(status); clear(actions);
    const id = rec.operation_id.trim();
    const otherReleased = library.filter((o) => o.operation_id === id && o.status === 'released' && o.revision !== rec.revision && o.origin !== 'bundled');
    if (checkError) {
      status.append(banner('danger', 'The record could not be checked', checkError.slice(0, 400)));
    } else if (check) {
      const issues = (check.summary && check.summary.issues) || {};
      const ex = check.existing;
      const blocked = ex && (ex.origin === 'bundled' || ((ex.status === 'released' || ex.status === 'superseded')));
      if (check.problems.length) {
        status.append(banner('danger', 'Cannot be saved yet', 'These entries cannot be read as they are:', check.problems.map((p) => `${whereLabel(p.where)}: ${p.message}`)));
      } else if (blocked) {
        status.append(banner('danger', ex.origin === 'bundled' ? 'This ID and revision belong to a bundled sample record' : `Revision ${rec.revision} of ${id} is ${ex.status} and cannot be changed`, 'Give the record another revision (or another ID).'));
      } else if (ex && ex.status === 'draft' && !(original && mode === 'edit' && original.id === id && original.revision === rec.revision)) {
        status.append(banner('warn', `A draft ${id} revision ${rec.revision} already exists`, 'Saving replaces that draft.'));
      }
      if (check.summary) {
        status.append(Object.keys(issues).length
          ? banner('warn', 'Not complete: it can be saved as a draft, not released', 'Measurements with an incomplete or draft record stay at "research preview". Nothing is filled in for you.',
            Object.entries(issues).map(([scope, lst]) => `${scope.replace('placard:', 'Placard ')}: ${lst.map(issueLabel).join('; ')}`))
          : banner('ok', 'Complete', 'Every datum, offset, size and tolerance the measurement needs is stated.'));
      }
      const can = check.can_save;
      if (confirmRelease) {
        actions.append(banner('warn', `Release ${id} revision ${rec.revision}?`, 'A released revision cannot be changed afterwards; a correction is a new revision. Release only values that were checked against their source.'),
          otherReleased.length ? h('label', { class: 'row', style: { gap: '10px' } }, h('input', { type: 'checkbox', checked: supersedeOld ? '' : null, 'data-field': 'supersede', style: { width: '22px', height: '22px' }, onchange: (ev) => { supersedeOld = ev.target.checked; } }),
            h('span', null, `Mark revision ${otherReleased.map((o) => o.revision).join(', ')} as superseded`)) : null,
          h('div', { class: 'row wrap' },
            h('button', { class: 'btn grow', type: 'button', onclick: () => { confirmRelease = false; renderStatus(); } }, 'Not yet'),
            h('button', { class: 'btn primary grow', type: 'button', 'data-action': 'op-release-confirm', onclick: () => save('released', supersedeOld ? otherReleased : []) }, icon('check'), 'Release')));
      } else {
        actions.append(h('div', { class: 'row wrap' },
          h('button', { class: 'btn grow', type: 'button', 'data-action': 'op-save-draft', disabled: can ? null : true, onclick: () => save('draft', []) }, icon('download'), 'Save as draft'),
          h('button', { class: 'btn primary grow', type: 'button', 'data-action': 'op-release', disabled: check.can_release ? null : true, onclick: () => { confirmRelease = true; renderStatus(); } }, icon('check'), 'Release…')));
      }
    } else {
      status.append(h('p', { class: 'hint' }, 'Checking…'));
    }
    actions.append(h('div', { class: 'row wrap' },
      h('button', { class: 'btn sm', type: 'button', 'data-action': 'op-json', 'aria-pressed': String(jsonMode), onclick: () => { jsonMode = !jsonMode; render(); } }, icon('layers'), jsonMode ? 'Back to the form' : 'Show as JSON'),
      h('button', { class: 'btn sm', type: 'button', 'data-action': 'op-export', onclick: exportJson }, icon('download'), 'Export'),
      original && mode === 'edit' ? h('button', { class: 'btn sm', type: 'button', 'data-action': 'op-delete', onclick: removeDraft }, icon('trash'), 'Delete this draft') : null));
  }

  async function save(statusValue, supersede) {
    const b = busy(statusValue === 'released' ? 'Releasing…' : 'Saving…');
    try {
      const out = await api('POST', 'api/operations', { operation: { ...buildRecord(rec), status: statusValue } });
      for (const o of supersede) await api('POST', 'api/operations/supersede', { operation_id: o.operation_id, revision: o.revision });
      toast({ title: statusValue === 'released' ? `Released ${out.summary.operation_id} revision ${out.summary.revision}` : `Draft ${out.summary.operation_id} revision ${out.summary.revision} saved`, kind: 'ok' });
      ref.close();
      onSaved(out.summary);
    } catch (err) { showError(err, 'The record was not saved'); confirmRelease = false; renderStatus(); } finally { b.done(); }
  }

  async function removeDraft() {
    try {
      await api('POST', 'api/operations/delete', { operation_id: original.id, revision: original.revision });
      toast({ title: 'Draft deleted', kind: 'ok' });
      ref.close();
      onSaved(null);
    } catch (err) { showError(err); }
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify(buildRecord(rec), null, 2)], { type: 'application/json' });
    const a = h('a', { href: URL.createObjectURL(blob), download: `${(rec.operation_id || 'operation').replace(/[^A-Za-z0-9._-]+/g, '_')}_rev${rec.revision || ''}.json` });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  function load(obj) {
    const base = blankRecord();
    Object.keys(rec).forEach((k) => delete rec[k]);
    Object.assign(rec, base, obj || {});
    rec.datums = { ...base.datums, ...(obj && obj.datums ? obj.datums : {}) };
    ['floor', 'side'].forEach((k) => { if (!rec.datums[k]) rec.datums[k] = blankDatum(k); });
    if (!Array.isArray(rec.placards) || !rec.placards.length) rec.placards = [blankPlacard()];
  }

  // ---------------------------------------------------------------- layout
  function render(remember = true) {
    if (remember) rememberOpen();
    clear(content);
    if (jsonMode) {
      const area = h('textarea', { class: 'input mono', rows: '18', spellcheck: 'false', 'aria-label': 'Operation record as JSON', 'data-field': 'json', style: { minHeight: '320px', fontSize: '13px', whiteSpace: 'pre', overflow: 'auto' } });
      area.value = JSON.stringify(buildRecord(rec), null, 2);
      const pick = h('input', { type: 'file', accept: 'application/json,.json', hidden: true, 'data-input': 'op-import', onchange: async (ev) => { const f = ev.target.files[0]; if (f) { area.value = await f.text(); apply(); } ev.target.value = ''; } });
      const apply = () => {
        try { const obj = JSON.parse(area.value); if (!obj || typeof obj !== 'object' || Array.isArray(obj)) throw new Error('not an object'); load(obj); changed(); toast({ title: 'JSON applied to the form', kind: 'ok', timeout: 2500 }); jsonMode = false; render(); }
        catch (err) { toast({ title: 'This is not a JSON record', detail: String(err.message || err), kind: 'error' }); }
      };
      content.append(h('p', { class: 'hint' }, 'The same record as text. Paste or import a record, then apply it to the form; it is checked exactly like typed values.'), area, pick,
        h('div', { class: 'row wrap' }, h('button', { class: 'btn primary grow', type: 'button', 'data-action': 'op-json-apply', onclick: apply }, icon('check'), 'Apply to the form'),
          h('button', { class: 'btn grow', type: 'button', 'data-action': 'op-import', onclick: () => pick.click() }, icon('upload'), 'Import a file')),
        status, actions);
      renderStatus();
      return;
    }
    content.append(
      h('p', { class: 'hint' }, 'Enter only what the source document states. A field left empty is reported as missing; LUMEN never fills in a datum, offset or tolerance.'),
      fold('record', 'Record', [
        h('div', { class: 'row wrap' }, text('Operation ID', rec, 'operation_id', { autocapitalize: 'characters', placeholder: 'e.g. OP-1234' }), text('Revision', rec, 'revision', { style: { maxWidth: '120px' }, autocapitalize: 'characters' })),
        text('Title', rec, 'title'),
        h('div', { class: 'row wrap' }, text('Configuration / effectivity', rec.effectivity, 'configuration', { name: 'configuration' }), text('Source document', rec.provenance, 'source', { name: 'source', placeholder: 'where these values come from' })),
        h('div', { class: 'row wrap' }, text('Entered by', rec.provenance, 'author', { name: 'author' }), text('Date', rec.provenance, 'date', { name: 'date', style: { maxWidth: '160px' } })),
        text('Notes', rec.provenance, 'notes', { name: 'provenance.notes' }),
      ]),
      fold('panel', 'Panel', [
        h('div', { class: 'row wrap' }, text('Panel ID', rec.panel, 'panel_id', { autocapitalize: 'characters' }), text('Description', rec.panel, 'description', { name: 'panel.description' })),
        list('Codes that identify this panel (barcode / QR text)', rec.panel, 'barcodes', { name: 'panel.barcodes' }),
        choice('Surface the placards sit on', rec.surface, 'target', [['plane', 'flat region of the panel'], ['curved', 'curved surface (experimental: no guidance)']], { name: 'surface.target', empty: '— choose —' }),
      ]),
      fold('floor', 'Floor datum (sets z)', datumFields('floor')),
      fold('side', 'Side datum (sets x)', datumFields('side')),
      fold('axes', 'Axes', [
        choice('How the two datums define the axes', rec.datums, 'axis_policy', POLICIES, { name: 'axis_policy', hint: 'With one datum setting the axes, the other one only fixes where its coordinate starts; their angle is checked against the nominal one.' }),
        h('div', { class: 'row wrap' }, angleQ('Nominal angle between the datums', rec.datums, 'nominal_angle', { name: 'nominal_angle' }), angleQ('Allowed deviation from it ±', rec.datums, 'max_axis_disagreement', { name: 'max_axis_disagreement' })),
      ]),
      rec.placards.map((p, i) => fold(`placard-${i}`, `Placard ${p.placard_id || i + 1}`, placardFields(p, i), { 'data-placard-index': String(i) })),
      h('button', { class: 'btn sm', type: 'button', 'data-action': 'placard-add', onclick: () => { rememberOpen(); rec.placards.push(blankPlacard()); openSections.add(`placard-${rec.placards.length - 1}`); changed(); render(false); } }, icon('plus'), 'Add another placard on this panel'),
      status, actions);
    renderStatus();
  }

  render();
  validate();
  return ref;
}

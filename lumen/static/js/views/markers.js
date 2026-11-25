// Step 2 — AprilTag sizes.  Scale comes from the measured edge of the black
// square, so this screen is about making that measurement unambiguous.
import { api } from '../api.js';
import { h, clear, icon, chip, banner, toast, showError } from '../dom.js';
import { state, setSession, prefs } from '../store.js';

const SVG = 'http://www.w3.org/2000/svg';

function tagDiagram() {
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('viewBox', '0 0 300 190');
  svg.setAttribute('class', 'diagram');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Measure the edge of the black square of the tag, not the sticker outline.');
  const mk = (tag, a, text) => { const e = document.createElementNS(SVG, tag); for (const k in a) e.setAttribute(k, a[k]); if (text) e.textContent = text; svg.appendChild(e); return e; };
  mk('rect', { x: 60, y: 14, width: 150, height: 160, rx: 6, fill: '#f4f4f1', stroke: '#9aa7bb', 'stroke-width': 1.5, 'stroke-dasharray': '5 4' });
  mk('rect', { x: 75, y: 26, width: 120, height: 120, fill: '#0b0b0b' });
  // a plausible-looking 6x6 interior (illustrative only)
  const bits = ['101101', '010011', '110100', '001011', '101110', '011001'];
  bits.forEach((row, r) => [...row].forEach((b, c) => { if (b === '1') mk('rect', { x: 90 + c * 15, y: 41 + r * 15, width: 15, height: 15, fill: '#f4f4f1' }); }));
  // dimension line
  mk('line', { x1: 75, y1: 160, x2: 195, y2: 160, stroke: '#ffc83d', 'stroke-width': 3 });
  mk('line', { x1: 75, y1: 152, x2: 75, y2: 168, stroke: '#ffc83d', 'stroke-width': 3 });
  mk('line', { x1: 195, y1: 152, x2: 195, y2: 168, stroke: '#ffc83d', 'stroke-width': 3 });
  mk('text', { x: 135, y: 181, 'text-anchor': 'middle', fill: 'currentColor', 'font-size': 13, 'font-weight': 700 }, 'measure this edge');
  mk('text', { x: 222, y: 60, fill: 'currentColor', 'font-size': 12 }, 'sticker');
  mk('text', { x: 222, y: 75, fill: 'currentColor', 'font-size': 12 }, 'outline:');
  mk('text', { x: 222, y: 90, fill: 'currentColor', 'font-size': 12, 'font-weight': 700 }, 'not this');
  mk('line', { x1: 220, y1: 66, x2: 212, y2: 66, stroke: 'currentColor', 'stroke-width': 1.5 });
  return svg;
}

export function mount(root, ctx) {
  const s = state.session;
  const table = JSON.parse(JSON.stringify(s.tag_sizes && Object.keys(s.tag_sizes).length ? s.tag_sizes : prefs.tagSizes));
  table.measured_mm = table.measured_mm || {};
  // offer the sizes remembered on this device when the session has none yet
  const remembered = prefs.tagSizes;
  if (!Object.keys(table.measured_mm).length && remembered && Object.keys(remembered.measured_mm || {}).length) {
    table.measured_mm = { ...remembered.measured_mm };
    table.nominal_mm = remembered.nominal_mm || table.nominal_mm;
  }
  let dirty = false;
  const rows = h('tbody');
  const seen = new Set();
  s.frames.forEach((f) => f.markers.forEach((m) => seen.add(m.tag_id)));

  const nominal = h('input', { class: 'input num', type: 'number', inputmode: 'decimal', step: '0.01', min: '5', max: '500', value: table.nominal_mm, id: 'nominal', 'aria-describedby': 'nominal-hint',
    oninput: (ev) => { table.nominal_mm = parseFloat(ev.target.value); dirty = true; } });
  const newId = h('input', { class: 'input num', type: 'number', inputmode: 'numeric', min: '0', max: '586', step: '1', placeholder: 'ID', 'aria-label': 'Tag ID to add' });
  const newSize = h('input', { class: 'input num', type: 'number', inputmode: 'decimal', step: '0.01', min: '5', max: '500', placeholder: 'mm', 'aria-label': 'Measured edge in millimetres' });

  function renderRows() {
    clear(rows);
    const ids = Object.keys(table.measured_mm).map(Number).sort((a, b) => a - b);
    if (!ids.length) {
      rows.append(h('tr', null, h('td', { colspan: '3', style: { color: 'var(--text-dim)' } }, 'No measured tags yet. Unmeasured tags use the nominal size and stay at research-preview status.')));
    }
    ids.forEach((id) => {
      rows.append(h('tr', { 'data-tag': id },
        h('td', null, h('span', { class: 'mono', style: { fontWeight: 800 } }, `ID ${id}`), seen.has(id) ? h('span', null, ' ', chip('seen', 'ok')) : null),
        h('td', null, h('div', { class: 'input-unit' },
          h('input', { class: 'input num', type: 'number', inputmode: 'decimal', step: '0.01', min: '5', max: '500', value: table.measured_mm[id], 'aria-label': `Measured edge of tag ${id} in millimetres`,
            oninput: (ev) => { table.measured_mm[id] = parseFloat(ev.target.value); dirty = true; } }),
          h('span', { class: 'unit' }, 'mm'))),
        h('td', { style: { width: '56px' } }, h('button', { class: 'icon-btn', type: 'button', 'aria-label': `Remove tag ${id}`, onclick: () => { delete table.measured_mm[id]; dirty = true; renderRows(); } }, icon('trash')))));
    });
  }

  function addTag(id, size) {
    if (!Number.isInteger(id) || id < 0 || id > 586) { toast({ title: 'Tag ID must be 0–586', kind: 'error' }); return; }
    if (!(size >= 5 && size <= 500)) { toast({ title: 'Enter the measured edge in millimetres', kind: 'error' }); return; }
    table.measured_mm[id] = size; dirty = true; renderRows();
  }

  const quick = [...seen].filter((id) => !(id in table.measured_mm)).sort((a, b) => a - b);

  root.append(
    h('h1', { class: 'view-title' }, 'Measure the tags'),
    h('p', { class: 'view-lead' }, 'All scale comes from the tags. Measure the edge of the black square of each sticker with a caliper and enter it for its ID. The size printed on the packaging is only a nominal value.'),
    h('div', { class: 'card' }, tagDiagram(),
      h('ul', { class: 'checklist' },
        h('li', null, icon('check'), h('span', null, 'Use 2–4 tags with different IDs, flat on the same panel face as the placard.')),
        h('li', null, icon('check'), h('span', null, 'Spread them around the placard location; any rotation is fine.')),
        h('li', null, icon('check'), h('span', null, 'Do not put a tag on the curved edge, the floor or a raised trim.')),
        h('li', null, icon('check'), h('span', null, 'Do not move a tag once capture has started.')))),
    h('div', { class: 'card' },
      h('h3', null, 'Measured sizes'),
      h('table', { class: 'tag-table' }, h('thead', null, h('tr', null, h('th', null, 'Tag'), h('th', null, 'Black-square edge'), h('th', null, ''))), rows),
      quick.length ? h('div', { class: 'row wrap' }, h('span', { class: 'label' }, 'Seen in photos:'), quick.map((id) => h('button', { class: 'btn sm', type: 'button', onclick: () => { newId.value = id; newSize.focus(); } }, `ID ${id}`))) : null,
      h('div', { class: 'row' },
        h('div', { style: { width: '104px' } }, newId),
        h('div', { class: 'grow input-unit' }, newSize, h('span', { class: 'unit' }, 'mm')),
        h('button', { class: 'btn', type: 'button', 'data-action': 'add-tag', onclick: () => { addTag(parseInt(newId.value, 10), parseFloat(newSize.value)); newId.value = ''; newSize.value = ''; } }, icon('plus'), 'Add'))),
    h('details', { class: 'fold' },
      h('summary', null, 'Nominal size for unmeasured tags'),
      h('div', { class: 'body' },
        h('div', { class: 'field' }, h('label', { for: 'nominal' }, 'Nominal black-square edge'), h('div', { class: 'input-unit' }, nominal, h('span', { class: 'unit' }, 'mm'))),
        h('p', { class: 'hint', id: 'nominal-hint' }, 'A tag without a measurement uses this value, labelled as nominal. Nominal scale carries a ±1 mm assumed tolerance and cannot support guidance status.'))),
    Object.keys(table.measured_mm).length === 0 ? banner('warn', 'No measured tag yet.', 'You can continue, but results stay a research preview until at least the tags in view are measured.') : null);

  async function save(next) {
    for (const [id, v] of Object.entries(table.measured_mm)) {
      if (!(v >= 5 && v <= 500)) { toast({ title: `Size of tag ${id} is not valid`, kind: 'error' }); return; }
    }
    if (!(table.nominal_mm >= 5 && table.nominal_mm <= 500)) { toast({ title: 'Nominal size is not valid', kind: 'error' }); return; }
    try {
      const body = { nominal_mm: table.nominal_mm, measured_mm: Object.fromEntries(Object.entries(table.measured_mm).map(([k, v]) => [String(k), v])) };
      const current = JSON.stringify({ n: s.tag_sizes.nominal_mm, m: s.tag_sizes.measured_mm });
      const wanted = JSON.stringify({ n: body.nominal_mm, m: body.measured_mm });
      if (dirty || current !== wanted) {
        const out = await api('PUT', `api/sessions/${s.id}/tags`, body);
        setSession(out);
        prefs.tagSizes = body;
      }
      ctx.go(next);
    } catch (err) { showError(err); }
  }

  renderRows();
  ctx.setActions([
    { label: 'Back', back: true, onClick: () => ctx.go('identify') },
    { id: 'continue', label: 'Continue to capture', kind: 'primary lg', icon: 'arrow', onClick: () => save('capture') },
  ]);
  return {};
}

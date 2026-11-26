// Step 5 — metric (rectified) view of the target plane.
import { api } from '../api.js';
import { h, clear, banner, showError, chip, fmt, icon, toast } from '../dom.js';
import { state, setSession, activeFrameId, activePlacard } from '../store.js';
import { Viewer } from '../viewer.js';
import { showFrame, viewerShell, zoomButtons, maximizeButton, statusHero, targetReadout, cell } from './common.js';

export function mount(root, ctx) {
  const shell = viewerShell({ label: 'Rectified view' });
  const panel = h('div', { class: 'panel' });
  const split = h('div', { class: 'split' }, shell.wrap, panel);
  root.append(split);
  const viewer = new Viewer(shell.wrap, { label: 'Rectified metric view of the panel face with grid and target' });
  window.__lumenViewer = viewer;   // hook for automated browser tests
  shell.tools.append(maximizeButton(viewer, split), ...zoomButtons(viewer));
  const layerBtn = (name, text) => h('button', { class: 'pill', type: 'button', 'aria-pressed': 'true',
    onclick: (ev) => { const on = ev.currentTarget.getAttribute('aria-pressed') !== 'true'; ev.currentTarget.setAttribute('aria-pressed', String(on)); ev.currentTarget.style.opacity = on ? '1' : '0.55'; viewer.setLayer(name, on); } }, text);
  shell.hud.append(h('span', { class: 'pill frozen' }, 'Rectified still'), layerBtn('grid', 'Grid'), layerBtn('dims', 'Dimensions'), layerBtn('tags', 'Tags'));
  let destroyed = false;
  let showRefs = false;
  let refInfo = null;

  async function load() {
    try {
      const [out, ref] = await Promise.all([
        showFrame(viewer, activeFrameId(), 'rect', { uncertainty: false, references: showRefs }),
        api('GET', `api/sessions/${state.session.id}/reference`).catch(() => null),
      ]);
      if (destroyed) return;
      refInfo = ref;
      renderPanel(out);
    } catch (err) { showError(err, 'Could not build the rectified view'); }
  }

  // Independent scale check: one ruler measurement between two marked tag corners.
  function scaleCheck() {
    const s = state.session;
    const refs = s.reference_distances || [];
    const sug = refInfo && refInfo.suggestion;
    const box = h('div', { class: 'card tight', 'data-region': 'scale-check' }, h('h3', null, 'Independent scale check'));
    if (!sug) { box.append(h('p', { class: 'hint' }, 'Needs two registered tags.')); return box; }
    const cur = refs.find((r) => r.evaluable);
    if (cur) {
      box.append(cur.consistent
        ? banner('ok', `Agrees: ${fmt(cur.measured_mm, 1)} mm measured, ${fmt(cur.computed_mm, 1)} mm from the tags`, `Difference ${cur.difference_mm > 0 ? '+' : ''}${fmt(cur.difference_mm, 2)} mm (limit ±${fmt(cur.limit_mm, 1)} mm).`)
        : banner('danger', `Contradiction: ${fmt(cur.measured_mm, 1)} mm measured, ${fmt(cur.computed_mm, 1)} mm from the tags`, `The tags make this length ${fmt(cur.relative_difference * 100, 1)} % ${cur.difference_mm > 0 ? 'longer' : 'shorter'} than the ruler. Check the declared tag sizes (black square, not the sticker) and the reading. No target is drawn until this is resolved.`),
        h('button', { class: 'btn sm', type: 'button', 'data-action': 'ref-remove', onclick: async () => { try { setSession(await api('DELETE', `api/sessions/${s.id}/reference/${cur.id}`)); load(); } catch (err) { showError(err); } } }, icon('trash'), 'Remove this measurement'));
      return box;
    }
    const input = h('input', { class: 'input num', type: 'number', inputmode: 'decimal', step: '0.1', min: '5', 'aria-label': 'Measured distance in millimetres', placeholder: 'e.g. 512.5' });
    box.append(
      h('p', { class: 'hint' }, `All lengths come from the declared tag sizes. A single ruler measurement catches an error they all share. Measure between the marked corners of the black squares of ID ${sug.a.tag_id} and ID ${sug.b.tag_id}.`),
      h('button', { class: 'btn sm toggle', type: 'button', 'aria-pressed': String(showRefs), 'data-action': 'ref-show', onclick: () => { showRefs = !showRefs; load(); } }, icon('eye'), showRefs ? 'Hide the two corners' : 'Show the two corners'),
      h('div', { class: 'row' }, h('div', { class: 'input-unit grow' }, input, h('span', { class: 'unit' }, 'mm')),
        h('button', { class: 'btn primary', type: 'button', 'data-action': 'ref-check', onclick: async () => {
          const v = parseFloat(input.value);
          if (!(v > 0)) { toast({ title: 'Enter the measured distance in millimetres', kind: 'error' }); return; }
          try {
            const out = await api('POST', `api/sessions/${s.id}/reference`, { a: { tag_id: sug.a.tag_id, corner: sug.a.corner }, b: { tag_id: sug.b.tag_id, corner: sug.b.corner }, measured_mm: v, u_mm: 0.5, instrument: 'ruler' });
            setSession(out.session); showRefs = true; load();
          } catch (err) { showError(err); }
        } }, 'Check')));
    return box;
  }

  function renderPanel(out) {
    const s = state.session;
    clear(panel);
    const p = activePlacard(s);
    const rep = out.overlay.reports && p ? out.overlay.reports[p.placard_id] : null;
    if (rep) panel.append(statusHero(rep));
    if (!s.local_frame || !s.local_frame.valid) {
      panel.append(banner('warn', 'No local frame yet.', 'The picture is metric, but the grid and the target need both datums. Go back to Datums.'));
    }
    const mmpp = out.rect ? out.rect.mm_per_px : null;
    panel.append(
      h('div', { class: 'eyebrow' }, 'Target'),
      targetReadout(p),
      h('div', { class: 'readout' },
        cell('Grid', out.overlay.grid_step_mm ? fmt(out.overlay.grid_step_mm, 0) : '—', 'mm'),
        cell('Scale', mmpp ? fmt(mmpp, 3) : '—', 'mm/px'),
        cell('Axes', s.local_frame ? s.local_frame.policy.replace(/_/g, ' ') : '—', '')),
      h('div', { class: 'row wrap' },
        s.local_frame ? chip(`datums at ${fmt(s.local_frame.raw_angle_deg, 2)}°`, s.local_frame.valid ? 'ok' : 'danger') : null,
        s.local_frame ? chip(`side datum on the ${s.local_frame.side_of_panel}`, 'solid') : null,
        s.mode === 'demo' ? chip('simulated', 'info') : null),
      scaleCheck(),
      banner('info', 'How to read this view', 'One pixel is the same distance everywhere on the panel face. The grid is computed from the registered tags and the confirmed datums — it is not decoration. Anything off the face (floor, curved edge, raised trim) is drawn where its line of sight meets the face, so it is displaced.'));
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('datums') },
      { id: 'continue', label: 'Continue', kind: 'primary lg', icon: 'arrow', onClick: () => ctx.go('place') },
    ]);
  }

  ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('datums') }]);
  load();
  return { flush: true, redraw: () => viewer.draw(), destroy() { destroyed = true; viewer.destroy(); } };
}

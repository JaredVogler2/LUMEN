// Step 5 — metric (rectified) view of the target plane.
import { h, clear, banner, showError, chip, fmt } from '../dom.js';
import { state, activeFrameId, activePlacard } from '../store.js';
import { Viewer } from '../viewer.js';
import { showFrame, viewerShell, zoomButtons, statusHero, targetReadout, cell } from './common.js';

export function mount(root, ctx) {
  const shell = viewerShell({ label: 'Rectified view' });
  const panel = h('div', { class: 'panel' });
  root.append(h('div', { class: 'split' }, shell.wrap, panel));
  const viewer = new Viewer(shell.wrap, { label: 'Rectified metric view of the panel face with grid and target' });
  window.__lumenViewer = viewer;   // hook for automated browser tests
  shell.tools.append(...zoomButtons(viewer));
  const layerBtn = (name, text) => h('button', { class: 'pill', type: 'button', 'aria-pressed': 'true',
    onclick: (ev) => { const on = ev.currentTarget.getAttribute('aria-pressed') !== 'true'; ev.currentTarget.setAttribute('aria-pressed', String(on)); ev.currentTarget.style.opacity = on ? '1' : '0.55'; viewer.setLayer(name, on); } }, text);
  shell.hud.append(h('span', { class: 'pill frozen' }, 'Rectified still'), layerBtn('grid', 'Grid'), layerBtn('dims', 'Dimensions'), layerBtn('tags', 'Tags'));
  let destroyed = false;

  async function load() {
    try {
      const out = await showFrame(viewer, activeFrameId(), 'rect', { uncertainty: false });
      if (destroyed) return;
      renderPanel(out);
    } catch (err) { showError(err, 'Could not build the rectified view'); }
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
      banner('info', 'How to read this view', 'One pixel is the same distance everywhere on the panel face. The grid is computed from the registered tags and the confirmed datums — it is not decoration. Anything off the face (floor, curved edge, raised trim) is drawn where its line of sight meets the face, so it is displaced.'));
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('datums') },
      { id: 'continue', label: 'Placement', kind: 'primary lg', icon: 'arrow', onClick: () => ctx.go('place') },
    ]);
  }

  ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('datums') }]);
  load();
  return { flush: true, redraw: () => viewer.draw(), destroy() { destroyed = true; viewer.destroy(); } };
}

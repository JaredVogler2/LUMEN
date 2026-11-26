// LUMEN application shell: header (brand, job, result mode), step navigation, view host.
import { api, url } from './api.js';
import { h, clear, icon, logo, toast, showError, sheet, banner } from './dom.js';
import {
  state, setState, subscribe, STEPS, prefs, applyTheme, applyBrand, brandInfo, resultMode, setSession, rememberedSession,
  stepAvailability, stepDone, operationSummary, endSession,
} from './store.js';
import * as identify from './views/identify.js';
import * as markers from './views/markers.js';
import * as capture from './views/capture.js';
import * as datums from './views/datums.js';
import * as rectified from './views/rectified.js';
import * as place from './views/place.js';
import * as verify from './views/verify.js';
import { openDiagnostics } from './views/diagnostics.js';
import { openSettings } from './views/settings.js';

const VIEWS = { identify, markers, capture, datums, rectified, place, verify };

const els = {};
let mounted = null;
let mountedStep = null;

function boot() {
  const bootMeta = document.querySelector('meta[name="lumen-boot"]');
  let bootInfo = {};
  try { bootInfo = JSON.parse(bootMeta ? bootMeta.content : '{}'); } catch (_) { bootInfo = {}; }
  state.boot = bootInfo;
  // light is the look of the application; dark is a choice in the menu and is remembered
  applyTheme(prefs.theme || 'light');
  applyBrand();
  state.units = prefs.units;

  const app = document.getElementById('app');
  clear(app);
  els.header = h('header', { class: 'app-header' });
  els.steps = h('nav', { class: 'steps', 'aria-label': 'Workflow steps' });
  els.view = h('main', { id: 'view', tabindex: '-1' });
  els.actions = h('div', { class: 'actions', role: 'toolbar', 'aria-label': 'Step actions' });
  app.append(els.header, els.steps, els.view, els.actions);

  subscribe(renderShell);
  renderShell();
  init().catch((err) => {
    showError(err, 'LUMEN could not start');
    clear(els.view).append(banner('danger', 'LUMEN could not reach its server.', 'Reload the page. If this persists the service is not running.'));
  });

  // The camera must never keep running in the background.
  document.addEventListener('visibilitychange', () => { if (document.hidden && mounted && mounted.pause) mounted.pause(); });
  window.addEventListener('pagehide', () => { if (mounted && mounted.destroy) mounted.destroy(); });
}

async function init() {
  const [config, ops] = await Promise.all([api('GET', 'api/config'), api('GET', 'api/operations')]);
  state.config = config;
  state.operations = ops;
  applyBrand();
  const sid = rememberedSession();
  if (sid) {
    try {
      const s = await api('GET', `api/sessions/${sid}`);
      setSession(s);
      toast({ title: 'Session restored', detail: 'Continuing where you left off.', kind: 'ok', timeout: 3000 });
    } catch (_) {
      setSession(null);
    }
  }
  go(state.step, true);
}

export function go(step, force = false) {
  const avail = stepAvailability();
  if (!avail[step] || !avail[step].ok) {
    toast({ title: 'Not available yet', detail: (avail[step] && avail[step].why) || '', kind: 'info', timeout: 3500 });
    return;
  }
  if (!force && step === mountedStep) return;
  state.step = step;
  mountView();
  renderShell();
}

function mountView() {
  if (mounted && mounted.destroy) { try { mounted.destroy(); } catch (e) { console.error(e); } }
  mounted = null;
  clear(els.view);
  clear(els.actions);
  els.view.className = '';
  const mod = VIEWS[state.step];
  const ctx = {
    go,
    remount: () => { mountView(); renderShell(); },
    setActions,
  };
  try {
    mounted = mod.mount(els.view, ctx) || {};
  } catch (err) {
    console.error(err);
    els.view.append(banner('danger', 'This screen failed to open.', String(err && err.message ? err.message : err)));
    mounted = {};
  }
  mountedStep = state.step;
  if (mounted.flush) els.view.classList.add('flush');
  els.view.scrollTop = 0;
  els.view.focus({ preventScroll: true });
}

function setActions(list) {
  clear(els.actions);
  (list || []).forEach((a) => {
    if (!a) return;
    const b = h('button', {
      class: `btn ${a.kind || ''} ${a.back ? 'back' : ''}`,
      type: 'button',
      disabled: a.disabled ? true : null,
      title: a.title || null,
      'data-action': a.id || null,
      onclick: a.onClick,
    }, a.icon ? icon(a.icon) : null, a.label);
    els.actions.append(b);
  });
}

// The brand block of the header: the organisation's logo where the installation supplies one, else LUMEN's own mark.
// It is built once per brand, not on every redraw (an <img> that is recreated loads again).
function brandBlock() {
  const b = brandInfo();
  const key = `${b.logo || ''}|${b.name || ''}`;
  if (els.brand && els.brandKey === key) return els.brand;
  els.brandKey = key;
  els.brand = b.logo
    ? h('div', { class: 'brand', 'data-region': 'brand', title: b.name || null },
      h('img', { class: 'brand-logo', src: url(b.logo), alt: b.name ? `${b.name} logo` : 'Organisation logo', 'data-region': 'brand-logo' }),
      h('span', { class: 'brand-divider', 'aria-hidden': 'true' }),
      h('span', { class: 'product' }, 'LUMEN'))
    : h('div', { class: 'brand plain', 'data-region': 'brand', title: b.name || null }, logo(), h('span', { class: 'product' }, 'LUMEN'),
      b.name ? h('span', { class: 'brand-divider', 'aria-hidden': 'true' }) : null, b.name ? h('span', { class: 'brand-name product' }, b.name) : null);
  return els.brand;
}

function openMode() {
  const m = resultMode();
  sheet({ title: m.title, content: h('div', { class: 'stack', 'data-region': 'mode-sheet' }, h('p', null, m.text)) });
}

function renderShell() {
  // header
  const s = state.session;
  const op = operationSummary(s);
  const m = resultMode(s);
  els.header.classList.toggle('no-job', !op);
  clear(els.header).append(
    brandBlock(),
    h('div', { class: 'header-context' },
      h('div', { class: 'op' }, op ? op.operation_id : 'No job selected'),
      h('div', { class: 'sub' }, op ? `${s.active_placard_id || 'no placard'} · rev ${op.revision}` : 'Placard positioning')),
    // one place on every screen says what kind of results these are
    h('button', { class: `mode-chip ${m.kind}`, type: 'button', 'data-region': 'mode', 'data-results': m.kind, 'aria-label': `${m.title}. Tap for details.`, onclick: openMode }, icon(m.icon), m.label),
    h('button', { class: 'icon-btn', 'aria-label': 'Menu', 'data-action': 'menu', onclick: openMenu }, icon('menu')));

  // steps: the stops of the way; a tool reached from a stop is shown as that stop
  const avail = stepAvailability(s);
  const here = (STEPS.find((x) => x.id === state.step) || {}).parent || state.step;
  clear(els.steps);
  STEPS.filter((st) => st.nav !== false).forEach((st) => {
    const a = avail[st.id];
    const done = stepDone(st.id, s);
    const cur = st.id === here;
    els.steps.append(h('button', {
      class: `step${done ? ' done' : ''}${a.ok ? '' : ' blocked'}`,
      type: 'button',
      'aria-current': cur ? 'step' : null,
      'aria-label': `Step ${st.n}: ${st.label}${done ? ' (done)' : ''}${a.ok ? '' : ' (not available yet)'}`,
      'data-step': st.id,
      onclick: () => go(st.id),
    }, h('span', { class: 'dot' }, done && !cur ? icon('check') : String(st.n)), h('span', { class: 'label' }, st.label)));
  });
  const curBtn = els.steps.querySelector('[aria-current="step"]');
  if (curBtn && curBtn.scrollIntoView) curBtn.scrollIntoView({ block: 'nearest', inline: 'center' });
}

function openMenu() {
  const s = state.session;
  let ref = null;
  const item = (label, ic, fn, cls = '') => h('button', { class: `btn block ${cls}`, type: 'button', onclick: () => { ref.close(); fn(); } }, icon(ic), label);
  const content = h('div', { class: 'menu-list' },
    item('Diagnostics', 'bug', () => openDiagnostics()),
    item('Settings & camera calibration', 'gear', () => openSettings()),
    item(state.theme === 'dark' ? 'Light mode' : 'Dark mode', state.theme === 'dark' ? 'sun' : 'moon', () => {
      const t = state.theme === 'dark' ? 'light' : 'dark';
      applyTheme(t); prefs.theme = t; setState({});
      if (mounted && mounted.redraw) mounted.redraw();
    }),
    s ? item('Export session summary', 'download', exportSession) : null,
    s ? item('End session and delete photos', 'trash', confirmEnd, 'danger') : null,
    h('p', { class: 'hint', style: { color: 'var(--text-dim)', fontSize: '13.5px', marginTop: '6px' } },
      `LUMEN ${state.boot.version || ''} · ${state.boot.algorithm || ''}. ${state.boot.notice || ''}`));
  ref = sheet({ title: 'Menu', content });
}

async function exportSession() {
  try {
    const data = await api('GET', `api/sessions/${state.session.id}/export`);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = h('a', { href: URL.createObjectURL(blob), download: `lumen-session-${state.session.id.slice(0, 8)}.json` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 20000);
    toast({ title: 'Summary exported', detail: 'Photos are not included in the summary.', kind: 'ok' });
  } catch (err) { showError(err); }
}

function confirmEnd() {
  let ref = null;
  const content = h('div', { class: 'stack' },
    h('p', null, 'This deletes the session and every photo it holds from the server. This cannot be undone.'),
    h('div', { class: 'row' },
      h('button', { class: 'btn grow', type: 'button', onclick: () => ref.close() }, 'Keep session'),
      h('button', { class: 'btn danger grow', type: 'button', 'data-action': 'confirm-delete', onclick: async () => { ref.close(); await endSession(); go('identify', true); toast({ title: 'Session deleted', kind: 'ok' }); } }, icon('trash'), 'Delete')));
  ref = sheet({ title: 'End session?', content });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();

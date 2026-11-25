// DOM helpers.  Everything is built with createElement/textContent — no
// innerHTML with data — so server or user strings can never inject markup.

const SVG_NS = 'http://www.w3.org/2000/svg';

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'html') throw new Error('raw html is not allowed');
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, String(v));
    }
  }
  append(el, children);
  // The DOM's own append()/replaceChildren() print null, false and arrays as
  // text ("null", "[object HTMLDivElement]").  Views build optional parts with
  // `cond ? node : null`, so elements made here take the forgiving versions.
  el.append = (...more) => { append(el, more); };
  el.replaceChildren = (...more) => { clear(el); append(el, more); };
  return el;
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

// 24x24 stroke icons (hand-drawn paths, no external icon font).
const ICONS = {
  menu: 'M4 7h16M4 12h16M4 17h16',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  x: 'M6 6l12 12M18 6L6 18',
  alert: 'M12 4l9 16H3L12 4zM12 10v4.5M12 17.5v.5',
  info: 'M12 21a9 9 0 100-18 9 9 0 000 18zM12 11v5.5M12 7.5v.5',
  block: 'M12 21a9 9 0 100-18 9 9 0 000 18zM5.6 5.6l12.8 12.8',
  camera: 'M4 8h3l1.5-2.5h7L17 8h3v11H4V8zM12 16.5a3.5 3.5 0 100-7 3.5 3.5 0 000 7z',
  upload: 'M12 16V5M7.5 9.5L12 5l4.5 4.5M5 19h14',
  video: 'M4 7h11v10H4V7zM15 10.5l5-3v9l-5-3',
  trash: 'M5 7h14M9 7V4.5h6V7M7 7l1 12.5h8L17 7M10 10.5v6M14 10.5v6',
  target: 'M12 21a9 9 0 100-18 9 9 0 000 18zM12 15a3 3 0 100-6 3 3 0 000 6zM12 1.5V5M12 19v3.5M1.5 12H5M19 12h3.5',
  grid: 'M4 4h16v16H4V4zM4 9.3h16M4 14.7h16M9.3 4v16M14.7 4v16',
  ruler: 'M3 16L16 3l5 5L8 21l-5-5zM7 12l2 2M10 9l2 2M13 6l2 2',
  tag: 'M4 4h7v7H4V4zM13 4h7v7h-7V4zM4 13h7v7H4v-7zM15.5 15.5h3v3h-3v-3z',
  scan: 'M4 8V5h3M17 5h3v3M20 16v3h-3M7 19H4v-3M7 9v6M10 9v6M13 9v6M16 9v6',
  torch: 'M9 3h6v4l-2 3v10h-2V10L9 7V3zM9 5.5h6',
  zoomfit: 'M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5',
  plus: 'M12 5v14M5 12h14',
  minus: 'M5 12h14',
  arrow: 'M5 12h13M12.5 6.5L18 12l-5.5 5.5',
  back: 'M19 12H6M11.5 6.5L6 12l5.5 5.5',
  sun: 'M12 16.5a4.5 4.5 0 100-9 4.5 4.5 0 000 9zM12 2.5V5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3L7 7M17 17l1.7 1.7M18.7 5.3L17 7M7 17l-1.7 1.7',
  moon: 'M20 14.5A8.5 8.5 0 019.5 4a8.5 8.5 0 1010.5 10.5z',
  download: 'M12 4v11M7.5 11L12 15.5 16.5 11M5 19.5h14',
  line: 'M4 19L20 5M4 19m-1.6 0a1.6 1.6 0 103.2 0 1.6 1.6 0 10-3.2 0M20 5m-1.6 0a1.6 1.6 0 103.2 0 1.6 1.6 0 10-3.2 0',
  layers: 'M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 17l9 5 9-5',
  refresh: 'M20 12a8 8 0 11-2.3-5.6M20 4.5v4.5h-4.5',
  play: 'M7 5l12 7-12 7V5z',
  pause: 'M7 5h3.5v14H7V5zM13.5 5H17v14h-3.5V5z',
  gear: 'M12 15.5a3.5 3.5 0 100-7 3.5 3.5 0 000 7zM19 12l2-1.5-2-3.5-2.4.8-1.7-1-.4-2.5h-4l-.4 2.5-1.7 1L5 7 3 10.5 5 12l-2 1.5L5 17l2.4-.8 1.7 1 .4 2.5h4l.4-2.5 1.7-1 2.4.8 2-3.5-2-1.5z',
  bug: 'M9 8.5a3 3 0 016 0V10H9V8.5zM8 10h8v5a4 4 0 01-8 0v-5zM4 13h4M16 13h4M5 7.5L8 10M19 7.5L16 10M5 19l3-2.5M19 19l-3-2.5',
  flash: 'M13 3L5 13.5h6L10 21l8-10.5h-6L13 3z',
  move: 'M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3',
  eye: 'M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 15a3 3 0 100-6 3 3 0 000 6z',
  help: 'M12 21a9 9 0 100-18 9 9 0 000 18zM9.5 9.5a2.5 2.5 0 114 2c-.9.6-1.5 1.1-1.5 2.2M12 16.8v.4',
  edit: 'M4 20h4.2L19 9.2 14.8 5 4 15.8V20zM13 6.8l4.2 4.2',
  copy: 'M9 8h11v12H9V8zM5 16V4h11',
  maximize: 'M4 10V4h6M20 14v6h-6M4 4l7 7M20 20l-7-7',
  minimize: 'M11 5v6H5M13 19v-6h6M11 11L4 4M13 13l7 7',
  // coaching arrows: where to move the phone
  go_left: 'M19 12H5M11 6l-6 6 6 6',
  go_right: 'M5 12h14M13 6l6 6-6 6',
  go_up: 'M12 19V5M6 11l6-6 6 6',
  go_down: 'M12 5v14M6 13l6 6 6-6',
  go_closer: 'M4 4l5 5M9 5v4H5M20 4l-5 5M15 5v4h4M4 20l5-5M9 19v-4H5M20 20l-5-5M15 19v-4h4',
  go_back: 'M9 9L4 4M4 8V4h4M15 9l5-5M20 8V4h-4M9 15l-5 5M4 16v4h4M15 15l5 5M20 16v4h-4',
};

export function icon(name, cls) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  if (cls) svg.setAttribute('class', cls);
  const p = document.createElementNS(SVG_NS, 'path');
  p.setAttribute('d', ICONS[name] || ICONS.info);
  svg.appendChild(p);
  return svg;
}

export function logo() {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 32 32');
  svg.setAttribute('aria-hidden', 'true');
  const mk = (tag, a) => { const e = document.createElementNS(SVG_NS, tag); for (const k in a) e.setAttribute(k, a[k]); return e; };
  svg.appendChild(mk('rect', { x: 1.5, y: 1.5, width: 29, height: 29, rx: 7, fill: '#ffc83d' }));
  svg.appendChild(mk('path', { d: 'M9 8v16h14', fill: 'none', stroke: '#191200', 'stroke-width': 3.2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
  svg.appendChild(mk('rect', { x: 15, y: 9, width: 8, height: 8, fill: 'none', stroke: '#191200', 'stroke-width': 2 }));
  svg.appendChild(mk('rect', { x: 17.5, y: 11.5, width: 3, height: 3, fill: '#191200' }));
  return svg;
}

// ------------------------------------------------------------------ toasts
let toastHost = null;
export function toast({ title, detail = '', kind = 'info', timeout = 5000 }) {
  if (!toastHost) {
    toastHost = h('div', { class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.appendChild(toastHost);
  }
  const el = h('div', { class: `toast ${kind}` },
    icon(kind === 'error' ? 'alert' : kind === 'ok' ? 'check' : 'info'),
    h('div', { class: 'grow' }, h('div', { class: 't' }, title), detail ? h('div', { class: 'd' }, detail) : null),
    h('button', { class: 'icon-btn', 'aria-label': 'Dismiss', style: { width: '36px', height: '36px' }, onclick: () => el.remove() }, icon('x')));
  toastHost.appendChild(el);
  if (timeout) setTimeout(() => el.remove(), timeout);
  return el;
}

export function showError(err, fallbackTitle = 'Something went wrong') {
  const e = err && err.lumen ? err.lumen : null;
  const title = e ? e.message : (err && err.message) || fallbackTitle;
  const detail = e && e.suggestions && e.suggestions.length ? e.suggestions.join(' ') : '';
  toast({ title, detail, kind: 'error', timeout: 9000 });
}

// ------------------------------------------------------------------ sheets
export function sheet({ title, content, wide = false }) {
  const prev = document.activeElement;
  const close = () => { scrim.remove(); document.removeEventListener('keydown', onKey); if (prev && prev.focus) prev.focus(); };
  const onKey = (ev) => { if (ev.key === 'Escape') close(); };
  const box = h('div', { class: 'sheet', role: 'dialog', 'aria-modal': 'true', 'aria-label': title, tabindex: '-1', style: wide ? { width: 'min(100%, 860px)' } : null },
    h('div', { class: 'grab', 'aria-hidden': 'true' }),
    h('div', { class: 'row' }, h('h2', { class: 'grow' }, title),
      h('button', { class: 'icon-btn', 'aria-label': 'Close', onclick: close }, icon('x'))),
    content);
  const scrim = h('div', { class: 'scrim', onclick: (ev) => { if (ev.target === scrim) close(); } }, box);
  document.body.appendChild(scrim);
  document.addEventListener('keydown', onKey);
  box.focus();
  return { close, box };
}

let busyEl = null;
let busyCount = 0;
export function busy(label = 'Working…') {
  busyCount += 1;
  if (!busyEl) {
    busyEl = h('div', { class: 'busy', role: 'alert', 'aria-busy': 'true' },
      h('div', { class: 'box' }, h('div', { class: 'spinner' }), h('div', { class: 'lbl' }, label)));
    document.body.appendChild(busyEl);
  } else {
    busyEl.querySelector('.lbl').textContent = label;
  }
  let done = false;
  return {
    update(text) { if (busyEl) busyEl.querySelector('.lbl').textContent = text; },
    done() {
      if (done) return;
      done = true;
      busyCount -= 1;
      if (busyCount <= 0 && busyEl) { busyEl.remove(); busyEl = null; busyCount = 0; }
    },
  };
}

export function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

export function banner(kind, title, detail, list) {
  return h('div', { class: `banner ${kind}`, role: kind === 'danger' ? 'alert' : 'note' },
    icon(kind === 'ok' ? 'check' : kind === 'info' ? 'info' : kind === 'danger' ? 'block' : 'alert'),
    h('div', { class: 'grow' },
      h('div', { class: 't' }, title),
      detail ? h('div', { class: 'd' }, detail) : null,
      list && list.length ? h('ul', null, list.map((x) => h('li', null, x))) : null));
}

export function chip(text, kind = '') {
  return h('span', { class: `chip ${kind}` }, text);
}

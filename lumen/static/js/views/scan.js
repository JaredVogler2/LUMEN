// Barcode / QR scanning to identify an operation, panel or placard.
//
// Frames (or a chosen picture) are decoded on the server.  What is read is
// only text for an exact library lookup; the code's size in the picture is
// never used as a measurement.
import { upload } from '../api.js';
import { h, clear, icon, toast, showError } from '../dom.js';
import { Camera } from '../camera.js';
import { fitTransform, frameToView } from '../viewer.js';

export function openScanner(onCode) {
  let closed = false;
  let busy = false;
  let paused = false;
  const prev = document.activeElement;
  const stage = h('div', { class: 'cam-stage' });
  const outline = h('canvas', { class: 'scan-outline', 'aria-hidden': 'true' });
  const pill = h('span', { class: 'pill', 'data-scan-state': 'starting' }, 'Starting camera…');
  const message = h('div', { class: 'cam-message' });
  const coach = h('div', { class: 'cam-coach', role: 'status', 'aria-live': 'polite' });
  const pick = h('input', { type: 'file', accept: 'image/*', hidden: true, 'data-input': 'scan-image', onchange: (ev) => { scanFile(ev.target.files[0]); ev.target.value = ''; } });
  const top = h('div', { class: 'cam-top' },
    h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Close scanner', 'data-action': 'scan-close', onclick: close }, icon('x')), pill);
  const controls = h('div', { class: 'cam-controls' },
    h('div', { class: 'cam-row' },
      h('button', { class: 'btn primary', type: 'button', 'data-action': 'scan-photo', onclick: scanStill }, icon('scan'), 'Scan this view carefully'),
      h('button', { class: 'btn', type: 'button', 'data-action': 'scan-choose', onclick: () => pick.click() }, icon('upload'), 'Choose a picture'),
      h('button', { class: 'btn', type: 'button', 'data-action': 'scan-type', onclick: close }, 'Type the ID instead')),
    h('div', { class: 'cam-hint', style: { textAlign: 'center' } }, 'Fill the frame with the code and hold it level. Only exact matches in the library are selected.'), pick);
  const root = h('div', { class: 'cam', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Scan a code', tabindex: '-1' }, stage, controls);
  const cam = new Camera({ onState });
  stage.append(cam.video, outline, top, coach, message);
  document.body.append(root);
  root.focus();
  const onKey = (ev) => { if (ev.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  window.__lumenScanner = { camera: cam, reads: 0 };

  function setPill(text, cls, st) { pill.textContent = text; pill.className = `pill ${cls || ''}`; pill.dataset.scanState = st; }
  function say(text) { clear(coach); if (text) coach.append(h('div', { class: 'coach' }, icon('info'), h('span', { class: 'grow' }, text))); }

  function onState(st, info) {
    if (closed) return;
    clear(message);
    if (st === 'starting') setPill('Starting camera…', '', 'starting');
    else if (st === 'live') { setPill('Looking for a code…', 'live', 'live'); say('Point the camera at the barcode or QR code.'); loop(); }
    else if (st === 'paused') {
      setPill('Camera paused', 'warn', 'paused');
      message.append(h('div', { class: 'cam-card' }, h('p', null, (info && info.reason) || 'The camera is paused.'),
        h('button', { class: 'btn primary lg', type: 'button', onclick: () => cam.start({ quality: 'standard' }) }, icon('camera'), 'Resume camera')));
    } else if (st === 'error') {
      setPill('Camera unavailable', 'warn', 'error');
      message.append(h('div', { class: 'cam-card', role: 'alert', 'data-cam-error': (info && info.code) || 'error' },
        h('h3', null, (info && info.title) || 'The camera could not be opened'), h('p', null, 'You can still choose a picture of the code, or type the ID.'),
        h('div', { class: 'row wrap', style: { justifyContent: 'center' } },
          h('button', { class: 'btn primary', type: 'button', onclick: () => pick.click() }, icon('upload'), 'Choose a picture'),
          h('button', { class: 'btn', type: 'button', onclick: () => cam.start({ quality: 'standard' }) }, icon('refresh'), 'Try again'))));
    }
  }

  function drawOutline(codes, fw, fh) {
    const r = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    outline.width = Math.round(r.width * dpr); outline.height = Math.round(r.height * dpr);
    const ctx = outline.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    if (!codes || !codes.length || !fw) return;
    const t = fitTransform(fw, fh, r.width, r.height, 0);
    ctx.lineWidth = 4; ctx.strokeStyle = '#3ddc97'; ctx.lineJoin = 'round';
    codes.forEach((c) => {
      ctx.beginPath();
      c.corners.forEach((p, i) => { const [x, y] = frameToView(t, p[0], p[1]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.closePath(); ctx.stroke();
    });
  }

  function handle(res, careful) {
    const codes = res.codes || [];
    window.__lumenScanner.reads += 1;
    drawOutline(codes, res.frame && res.frame.width, res.frame && res.frame.height);
    if (!codes.length) {
      if (careful) say('No code could be read in that picture. Move closer so the code fills the frame, hold it level and avoid glare.');
      return false;
    }
    if (codes.length === 1) { finish(codes[0].text); return true; }
    // several codes in view: the operator chooses
    paused = true;
    setPill(`${codes.length} codes in view`, 'warn', 'choose');
    clear(message).append(h('div', { class: 'cam-card', 'data-region': 'scan-choices' },
      h('h3', null, 'More than one code was read'), h('p', null, 'Choose the one that identifies this job.'),
      h('div', { class: 'stack', style: { alignSelf: 'stretch' } }, codes.map((c) => h('button', { class: 'btn block', type: 'button', 'data-code': c.text, onclick: () => finish(c.text) },
        h('span', { class: 'mono' }, c.text), h('span', { class: 'hint' }, ` ${c.format}${c.resolved.status === 'exact' ? ' · in the library' : c.resolved.status === 'ambiguous' ? ' · matches several records' : ' · not in the library'}`)))),
      h('button', { class: 'btn', type: 'button', onclick: () => { paused = false; clear(message); setPill('Looking for a code…', 'live', 'live'); loop(); } }, 'Scan again')));
    return true;
  }

  function finish(text) {
    close();
    toast({ title: 'Code read', detail: text, kind: 'info', timeout: 2500 });
    onCode(text);
  }

  async function loop() {
    if (busy) return;
    busy = true;
    try {
      while (!closed && !paused && cam.state === 'live') {
        let shot;
        try { shot = await cam.grabReduced(1400, 0.85); } catch (_) { await new Promise((r) => setTimeout(r, 300)); continue; }
        if (shot.bitmap && shot.bitmap.close) shot.bitmap.close();
        try {
          const res = await upload('api/scan?thorough=false', shot.blob, { timeout: 15000 });
          if (closed || paused) break;
          if (handle(res, false)) break;
        } catch (err) {
          if (closed) break;
          await new Promise((r) => setTimeout(r, 900));
        }
        await new Promise((r) => setTimeout(r, 350));
      }
    } finally { busy = false; }
  }

  async function scanStill() {
    if (cam.state !== 'live') { pick.click(); return; }
    paused = true;
    setPill('Reading…', '', 'reading');
    try {
      const shot = await cam.grab({ quality: 0.92 });
      const res = await upload('api/scan?thorough=true', shot.blob, { timeout: 30000 });
      if (closed) return;
      if (!handle(res, true)) { paused = false; setPill('Looking for a code…', 'live', 'live'); loop(); }
    } catch (err) { showError(err, 'Scan failed'); paused = false; setPill('Looking for a code…', 'live', 'live'); loop(); }
  }

  async function scanFile(file) {
    if (!file) return;
    paused = true;
    setPill('Reading the picture…', '', 'reading');
    try {
      let blob = file;
      if (!/^image\/(jpeg|png|webp|bmp)$/.test(file.type)) {
        const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
        const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
        c.getContext('2d').drawImage(bmp, 0, 0);
        blob = await new Promise((ok, bad) => c.toBlob((x) => (x ? ok(x) : bad(new Error('encode failed'))), 'image/jpeg', 0.92));
      }
      const res = await upload('api/scan?thorough=true', blob, { timeout: 30000 });
      if (closed) return;
      if (!(res.codes || []).length) { say('No code could be read in that picture.'); toast({ title: 'No code found in the picture', kind: 'error' }); }
      if (!handle(res, true)) { paused = false; if (cam.state === 'live') { setPill('Looking for a code…', 'live', 'live'); loop(); } else setPill('Camera unavailable', 'warn', 'error'); }
    } catch (err) { showError(err, 'The picture could not be read'); paused = false; }
  }

  function close() {
    if (closed) return;
    closed = true;
    cam.destroy();
    window.__lumenScanner = null;
    document.removeEventListener('keydown', onKey);
    root.remove();
    if (prev && prev.focus) prev.focus();
  }

  cam.start({ quality: 'standard' });
  return { close };
}

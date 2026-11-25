// In-app camera: photos, a recorded sweep, and flash on/off pairs.
import { api, upload } from '../api.js';
import { h, clear, icon, toast, showError } from '../dom.js';
import { state, refreshSession } from '../store.js';
import { Camera } from '../camera.js';
import { coachLine } from './common.js';

let lastConfig = { sid: null, key: null };

// role 'registration' (default): photos, sweep and flash pairs for the capture step.
// role 'verification' with single = true: one photo of the installed placard, then close.
// custom = { url, hint, coach, count(), onResult(out) }: photos go to another endpoint (lens calibration);
// no session is needed then.
export function openCamera({ onChange = () => {}, onFallback = null, role = 'registration', single = false, onCaptured = null, custom = null } = {}) {
  const sid = state.session ? state.session.id : null;
  const lim = state.config.limits;
  let mode = 'photo';            // photo | sweep | pair
  let busyCount = 0;             // uploads in flight
  let recording = null;
  let closed = false;
  let advice = null;
  let pairBusy = false;
  let captured = null;
  const prev = document.activeElement;

  const stage = h('div', { class: 'cam-stage' });
  const pill = h('span', { class: 'pill', 'data-cam-state': 'starting' }, 'Starting camera…');
  const count = h('span', { class: 'pill', 'data-cam-count': '0' }, '0 photos');
  const coach = h('div', { class: 'cam-coach', role: 'status', 'aria-live': 'polite' });
  const message = h('div', { class: 'cam-message' });
  const flashNote = h('div', { class: 'cam-flash', 'aria-hidden': 'true' });
  const top = h('div', { class: 'cam-top' },
    h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Close camera', 'data-action': 'cam-close', onclick: close }, icon('x')),
    pill, count, h('span', { class: 'grow' }),
    h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Camera options', 'data-action': 'cam-options', onclick: options }, icon('gear')));
  const modeSeg = h('div', { class: 'seg', role: 'radiogroup', 'aria-label': 'Capture mode' });
  const torchBtn = h('button', { class: 'btn sm toggle', type: 'button', 'data-action': 'cam-torch', 'aria-pressed': 'false', hidden: true, onclick: toggleTorch }, icon('torch'), 'Light');
  const shutter = h('button', { class: 'cam-shutter', type: 'button', 'data-action': 'cam-shutter', 'aria-label': 'Take photo', disabled: true, onclick: onShutter }, h('i', null));
  const doneBtn = h('button', { class: 'btn primary', type: 'button', 'data-action': 'cam-done', onclick: close }, icon('check'), 'Done');
  const hint = h('div', { class: 'cam-hint' });
  const controls = h('div', { class: 'cam-controls' }, h('div', { class: 'cam-row' }, modeSeg, torchBtn), h('div', { class: 'cam-row main' }, h('span', { class: 'cam-side' }, hint), shutter, h('span', { class: 'cam-side end' }, doneBtn)));
  const root = h('div', { class: 'cam', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Camera', tabindex: '-1' }, stage, controls);
  stage.append(top, coach, message, flashNote);

  const cam = new Camera({ onState: onState });
  stage.prepend(cam.video);
  document.body.append(root);
  root.focus();
  const onKey = (ev) => { if (ev.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  window.__lumenCamera = cam;      // hook for automated browser tests

  function setPill(text, cls, st) { pill.textContent = text; pill.className = `pill ${cls || ''}`; pill.dataset.camState = st; }

  function renderModes() {
    clear(modeSeg);
    const canRecord = !!Camera.recordingType();
    const opts = single || custom ? [['photo', 'Photo']] : [['photo', 'Photo'], ...(canRecord ? [['sweep', 'Sweep video']] : []), ...(cam.torch.supported ? [['pair', 'Flash pair']] : [])];
    modeSeg.hidden = single || !!custom;
    if (!opts.some((o) => o[0] === mode)) mode = 'photo';
    opts.forEach(([id, label]) => modeSeg.append(h('button', {
      type: 'button', role: 'radio', 'aria-checked': mode === id ? 'true' : 'false', 'aria-pressed': mode === id ? 'true' : 'false', 'data-action': `cam-mode-${id}`,
      disabled: recording || pairBusy ? true : null, onclick: () => { mode = id; renderModes(); },
    }, label)));
    torchBtn.hidden = !cam.torch.supported || mode === 'pair';
    shutter.classList.toggle('rec', mode === 'sweep');
    shutter.classList.toggle('recording', !!recording);
    shutter.setAttribute('aria-label', mode === 'sweep' ? (recording ? 'Stop recording' : 'Record sweep') : mode === 'pair' ? 'Take flash pair' : 'Take photo');
    hint.textContent = mode === 'sweep'
      ? (recording ? '' : `Record up to ${Math.min(30, lim.max_video_seconds)} s while moving slowly around the panel.`)
      : mode === 'pair' ? 'Rest the phone on something: two photos, light off then on.'
        : custom ? custom.hint
          : single ? 'One photo: the whole placard and the tags in view.'
            : 'Full frame is captured. Keep every tag in view.';
  }

  function renderCount() {
    const n = custom ? custom.count() : state.session.frames.filter((f) => f.role === role).length;
    count.textContent = `${n} photo${n === 1 ? '' : 's'}${busyCount ? ` · ${busyCount} uploading` : ''}`;
    count.dataset.camCount = String(n);
  }

  async function refreshAdvice() {
    if (custom) {
      clear(coach).append(coachLine({ next: { code: 'custom', level: 'info', text: custom.coach(), arrow: null } }));
      return;
    }
    if (role !== 'registration') {
      clear(coach).append(coachLine({ next: { code: 'verify_photo', level: 'info', text: 'Photograph the installed placard with the tags still in place. Keep the whole placard in view.', arrow: null } }));
      return;
    }
    try {
      advice = await api('GET', `api/sessions/${sid}/capture/advice`);
      if (closed) return;
      clear(coach);
      const line = coachLine(advice);
      if (line) coach.append(line);
    } catch (_) { /* coaching is optional */ }
  }

  function onState(st, info) {
    if (closed) return;
    clear(message);
    shutter.disabled = st !== 'live';
    if (st === 'starting') {
      setPill('Starting camera…', '', 'starting');
      message.append(h('div', { class: 'cam-card' }, h('div', { class: 'spinner' }), h('p', null, 'Starting the camera. If the browser asks, allow camera access.')));
    } else if (st === 'live') {
      setPill('Camera live', 'live', 'live');
      renderModes();
      checkConfig();
      cam.stableAlternative().then((alt) => {
        if (alt && !closed) toast({ title: 'A fixed-lens camera is available', detail: `“${alt.label}” does not switch lenses. Choose it under camera options for consistent geometry.`, kind: 'info', timeout: 9000 });
      });
    } else if (st === 'paused') {
      setPill('Camera paused', 'warn', 'paused');
      message.append(h('div', { class: 'cam-card' }, h('p', null, (info && info.reason) || 'The camera is paused.'),
        h('button', { class: 'btn primary lg', type: 'button', 'data-action': 'cam-resume', onclick: () => cam.start() }, icon('camera'), 'Resume camera')));
    } else if (st === 'error') {
      setPill('Camera unavailable', 'warn', 'error');
      message.append(h('div', { class: 'cam-card', role: 'alert', 'data-cam-error': (info && info.code) || 'error' },
        h('h3', null, (info && info.title) || 'The camera could not be opened'), h('p', null, (info && info.detail) || ''),
        h('div', { class: 'row wrap', style: { justifyContent: 'center' } },
          h('button', { class: 'btn primary', type: 'button', 'data-action': 'cam-retry', onclick: () => cam.start() }, icon('refresh'), 'Try again'),
          onFallback ? h('button', { class: 'btn', type: 'button', 'data-action': 'cam-fallback', onclick: () => { close(); onFallback(); } }, icon('upload'), 'Use the phone camera instead') : null)));
    }
  }

  function checkConfig() {
    const key = cam.configKey();
    const s = cam.settings();
    if (lastConfig.sid === sid && lastConfig.key && lastConfig.key !== key && (custom ? custom.count() : state.session.frames.length)) {
      toast({ title: 'The camera configuration changed', detail: `Now ${s.width}×${s.height}${s.camera_label ? ` (${s.camera_label})` : ''}. Photos from different configurations are not combined in the 3-D checks: use one orientation, lens and zoom for a session.`, kind: 'error', timeout: 10000 });
    }
    lastConfig = { sid, key };
  }

  async function toggleTorch() {
    if (!cam.torch.supported) return;
    torchBtn.disabled = true;
    const want = !(cam.torch.requested);
    const st = await cam.setTorch(want);
    torchBtn.disabled = false;
    torchBtn.setAttribute('aria-pressed', cam.torch.requested ? 'true' : 'false');
    torchBtn.lastChild.textContent = st === 'on' ? 'Light on' : st === 'off' ? 'Light' : 'Light ?';
    torchBtn.title = `Light: ${cam.torchEvidence()}`;
    if (want && st === 'unknown') toast({ title: 'The light could not be confirmed', detail: 'It was requested, but neither the browser nor the picture confirms it. Photos are recorded with light state “unknown”.', kind: 'info', timeout: 7000 });
  }

  function blink() { flashNote.classList.remove('go'); void flashNote.offsetWidth; flashNote.classList.add('go'); }

  async function send(shot, extraMeta = {}, source = 'camera_still') {
    busyCount += 1; renderCount();
    if (custom) {
      try {
        const out = await upload(custom.url, shot.blob, { meta: { ...shot.meta, ...extraMeta } });
        const msg = custom.onResult(out);
        if (msg) toast(msg);
        return out;
      } catch (err) { showError(err, 'Upload failed'); return null; } finally {
        busyCount -= 1;
        if (!closed) { renderCount(); refreshAdvice(); }
      }
    }
    try {
      const out = await upload(`api/sessions/${sid}/frames?role=${role}&source=${source}`, shot.blob, { meta: { ...shot.meta, ...extraMeta } });
      const f = out.frame;
      const acc = f.markers.filter((m) => m.accepted).length;
      if (!acc) toast({ title: 'No usable tag in that photo', detail: role === 'registration' ? 'See the hint at the top; the photo was kept so you can inspect it.' : 'The tags must be in the picture to locate the placard.', kind: 'error', timeout: 6000 });
      captured = f;
      return f;
    } catch (err) {
      showError(err, 'Upload failed');
      return null;
    } finally {
      busyCount -= 1;
      await refreshSession().catch(() => {});
      if (!closed) { renderCount(); refreshAdvice(); }
    }
  }

  async function onShutter() {
    if (cam.state !== 'live') return;
    if (mode === 'photo') {
      if (!custom && state.session.frames.length + busyCount >= lim.max_frames) { toast({ title: 'Frame limit reached', detail: `This session holds at most ${lim.max_frames} frames.`, kind: 'error' }); return; }
      try {
        checkConfig();
        const shot = await cam.grab(); blink();
        if (single) { shutter.disabled = true; setPill('Uploading…', '', 'uploading'); const f = await send(shot); if (f) { close(); return; } if (!closed) { shutter.disabled = cam.state !== 'live'; setPill('Camera live', 'live', 'live'); } } else send(shot);
      } catch (err) { showError(err, 'Could not take the photo'); }
    } else if (mode === 'sweep') {
      if (recording) { recording.stop(); return; }
      await recordSweep();
    } else if (mode === 'pair') {
      await takePair();
    }
  }

  async function takePair() {
    if (pairBusy) return;
    pairBusy = true; shutter.disabled = true; renderModes();
    const pairId = `p${Date.now().toString(36)}`;
    try {
      setPill('Hold still — light off', 'warn', 'pair');
      await cam.setTorch(false, { settleMs: 450 });
      const a = await cam.grab(); blink();
      a.meta.torch = cam.torchState() === 'unknown' ? 'off' : cam.torchState();
      setPill('Hold still — light on', 'warn', 'pair');
      const st = await cam.setTorch(true, { settleMs: 900 });
      const b = await cam.grab(); blink();
      await cam.setTorch(false, { settleMs: 200 });
      if (st !== 'on') {
        toast({ title: 'The light did not come on', detail: 'Neither the browser nor the picture confirms it, so this is not a flash pair. The two photos were not kept.', kind: 'error', timeout: 8000 });
        return;
      }
      setPill('Checking the pair…', '', 'pair');
      await send(a, { pair_id: pairId, torch: 'off' });
      await send(b, { pair_id: pairId, torch: 'on' });
      const adv = await api('GET', `api/sessions/${sid}/capture/advice`);
      const p = (adv.pairs || []).find((x) => x.pair_id === pairId);
      if (p && p.usable) toast({ title: 'Flash pair is aligned', detail: `Camera moved ${p.max_shift_px.toFixed(1)} px between the two photos.`, kind: 'ok' });
      else if (p) toast({ title: 'Flash pair cannot be used as a pair', detail: (p.reasons || []).join(' ') || p.status, kind: 'error', timeout: 9000 });
    } catch (err) {
      showError(err, 'Flash pair failed');
      try { await cam.setTorch(false, { settleMs: 100 }); } catch (_) { /* ignore */ }
    } finally {
      pairBusy = false;
      if (!closed) { shutter.disabled = cam.state !== 'live'; if (cam.state === 'live') setPill('Camera live', 'live', 'live'); renderModes(); }
    }
  }

  async function recordSweep() {
    let rec;
    try {
      rec = cam.startRecording({ maxSeconds: Math.min(30, lim.max_video_seconds), onTick: (s, max) => { setPill(`Recording ${s.toFixed(0)} / ${max.toFixed(0)} s`, 'live', 'recording'); } });
    } catch (err) { showError(err, 'Recording is not available'); return; }
    recording = rec; renderModes();
    let out;
    try { out = await rec.done; } catch (err) { recording = null; renderModes(); showError(err); return; }
    recording = null; renderModes();
    if (closed) return;
    if (out.seconds < 1.5 || !out.blob.size) { setPill('Camera live', 'live', 'live'); toast({ title: 'Recording too short', detail: 'Record a few seconds while moving around the panel.', kind: 'info' }); return; }
    if (out.blob.size > lim.max_video_mb * 1e6) { setPill('Camera live', 'live', 'live'); toast({ title: 'Recording too large', detail: `Limit is ${lim.max_video_mb} MB: record a shorter sweep.`, kind: 'error' }); return; }
    shutter.disabled = true;
    try {
      setPill('Uploading sweep…', '', 'uploading');
      const res = await upload(`api/sessions/${sid}/video?source=live_sweep`, out.blob, { meta: out.meta, timeout: 600000, onProgress: (p) => setPill(`Uploading sweep ${Math.round(p * 100)} %`, '', 'uploading') });
      let job = res.job;
      while (!closed && (job.state === 'queued' || job.state === 'running')) {
        await new Promise((r) => setTimeout(r, 700));
        job = await api('GET', `api/sessions/${sid}/jobs/${job.id}`);
        setPill(`${job.stage || 'Working'} ${Math.round((job.progress || 0) * 100)} %`, '', 'processing');
      }
      if (job.state !== 'done') throw Object.assign(new Error(job.error ? job.error.message : 'The sweep could not be processed.'), { lumen: job.error });
      const r = job.result;
      await refreshSession();
      toast({ title: `${r.selected.length} frame${r.selected.length === 1 ? '' : 's'} chosen from the sweep`, detail: (r.hints && r.hints.length ? r.hints.join(' ') : `from ${r.candidates} candidates. Review them after closing the camera.`), kind: r.selected.length ? 'ok' : 'error', timeout: 9000 });
    } catch (err) { showError(err, 'The sweep could not be used'); } finally {
      if (!closed) { shutter.disabled = cam.state !== 'live'; if (cam.state === 'live') setPill('Camera live', 'live', 'live'); renderCount(); refreshAdvice(); }
    }
  }

  async function options() {
    const cams = await cam.rearCameras();
    const cur = cam.track && cam.track.getSettings ? cam.track.getSettings().deviceId : null;
    const s = cam.settings();
    const box = h('div', { class: 'cam-card options', role: 'dialog', 'aria-label': 'Camera options' },
      h('h3', null, 'Camera options'),
      h('p', { class: 'hint' }, s.width ? `Stream: ${s.width}×${s.height}${s.frame_rate ? ` @ ${Math.round(s.frame_rate)} fps` : ''}${s.zoom !== null ? ` · zoom ${s.zoom}` : ''}. The browser does not report lens calibration.` : 'The camera is not running.'),
      h('div', { class: 'seg' },
        h('button', { type: 'button', 'aria-pressed': cam.quality === 'high' ? 'true' : 'false', 'data-action': 'cam-quality-high', onclick: () => { clear(message); cam.start({ quality: 'high' }); } }, 'Highest resolution'),
        h('button', { type: 'button', 'aria-pressed': cam.quality === 'standard' ? 'true' : 'false', 'data-action': 'cam-quality-standard', onclick: () => { clear(message); cam.start({ quality: 'standard' }); } }, '1080p (faster)')),
      cams.length > 1 ? h('div', { class: 'stack' }, cams.map((d) => h('button', { class: 'btn sm block toggle', type: 'button', 'aria-pressed': d.deviceId === cur ? 'true' : 'false', onclick: () => { clear(message); cam.start({ deviceId: d.deviceId }); } }, d.label))) : null,
      h('p', { class: 'hint' }, cam.torch.supported ? `Light: ${cam.torchEvidence()}.` : 'This browser does not offer control of the phone light. Photos taken with a light from elsewhere can be marked on the Capture screen.'),
      h('button', { class: 'btn', type: 'button', onclick: () => clear(message) }, 'Close options'));
    clear(message).append(box);
  }

  function close() {
    if (closed) return;
    closed = true;
    if (recording) { try { recording.stop(); } catch (_) { /* ignore */ } }
    cam.destroy();
    window.__lumenCamera = null;
    document.removeEventListener('keydown', onKey);
    root.remove();
    if (prev && prev.focus) prev.focus();
    onChange();
    if (captured && onCaptured) onCaptured(captured);
  }

  renderModes(); renderCount(); refreshAdvice();
  cam.start();
  return { close, camera: cam };
}

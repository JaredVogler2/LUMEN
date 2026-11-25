// Live placement guidance.
//
// Each camera frame is sent to the server, which measures the tags in *that
// frame* and returns the overlay for it.  The viewer then shows that same
// frame with its overlay.  A target outline is therefore never drawn on a
// picture it was not computed from; a slow connection only lowers the update
// rate.  When a frame cannot be registered the overlay disappears with it,
// and when no answer arrives for a while the last overlay is removed too.
import { upload } from '../api.js';
import { h, clear, icon, toast } from '../dom.js';
import { state } from '../store.js';
import { Camera } from '../camera.js';
import { coachLine } from './common.js';

const STALE_MS = 2500;

export async function startLive({ viewer, shell, modePill, onStopped = () => {}, onReport = () => {} }) {
  const sid = state.session.id;
  const maxEdge = Math.min(1600, state.config.limits.max_track_long_edge || 1600);
  let running = true;
  let frozen = false;
  let lastUpdate = 0;
  let lastTracked = 0;
  let shownTracking = false;
  let rate = 0;
  let failures = 0;

  const coach = h('div', { class: 'live-coach', role: 'status', 'aria-live': 'polite' });
  const freezeBtn = h('button', { class: 'pill', type: 'button', 'data-action': 'live-freeze', 'aria-pressed': 'false', onclick: toggleFreeze }, 'Freeze');
  shell.wrap.append(coach);
  shell.hud.append(freezeBtn);

  function setPill(text, cls, mode) {
    modePill.textContent = text; modePill.className = `pill ${cls}`; modePill.dataset.mode = mode;
  }
  function setCoach(next) {
    clear(coach);
    const line = next ? coachLine({ next }) : null;
    if (line) coach.append(line);
  }

  const cam = new Camera({
    onState: (st, info) => {
      if (!running) return;
      if (st === 'starting') setPill('Starting camera…', 'warn', 'starting');
      else if (st === 'paused') { hideOverlay(); setPill('Camera paused', 'warn', 'paused'); setCoach({ code: 'paused', level: 'warn', text: `${(info && info.reason) || 'The camera is paused.'} Stop and start live guidance to resume.`, arrow: null }); }
      else if (st === 'error') { toast({ title: (info && info.title) || 'The camera could not be opened', detail: (info && info.detail) || '', kind: 'error', timeout: 9000 }); stop(); }
    },
  });
  window.__lumenLive = { camera: cam, updates: 0, lastResult: null };

  function hideOverlay() {
    viewer.setOverlay({ primitives: [] });
    shownTracking = false;
  }

  function show(shot, res, roundTripMs) {
    const pic = shot.bitmap || shot.canvas;
    const fw = (res.frame && res.frame.width) || shot.width, fh = (res.frame && res.frame.height) || shot.height;
    viewer.setFrame(pic, fw, fh);
    const t = res.tracking || { state: 'lost', message: 'No result.' };
    const now = performance.now();
    rate = lastUpdate ? 0.7 * rate + 0.3 * (1000 / Math.max(now - lastUpdate, 1)) : 0;
    lastUpdate = now;
    window.__lumenLive.updates += 1;
    window.__lumenLive.lastResult = res;
    if (t.state === 'tracking') {
      viewer.setOverlay(res);
      shownTracking = true;
      lastTracked = now;
      setPill(`LIVE · ${(roundTripMs / 1000).toFixed(1)} s delay${rate ? ` · ${rate.toFixed(1)}/s` : ''}`, 'live', 'live');
      setCoach(t.code === 'tracking' ? null : { code: t.code, level: t.code === 'tag_moved' ? 'warn' : 'todo', text: t.message, arrow: t.arrow });
    } else {
      hideOverlay();
      setPill('Tracking lost — no overlay', 'warn', 'lost');
      setCoach({ code: t.code, level: 'todo', text: [t.message, ...(t.suggestions || [])].join(' '), arrow: t.arrow });
    }
    onReport(res);
  }

  // If answers stop coming, the last overlay must not linger on screen.
  const staleTimer = setInterval(() => {
    if (!running || frozen || !shownTracking) return;
    if (performance.now() - lastUpdate > STALE_MS) {
      hideOverlay();
      setPill('No recent update — overlay hidden', 'warn', 'stale');
    }
  }, 300);

  function toggleFreeze() {
    frozen = !frozen;
    freezeBtn.setAttribute('aria-pressed', String(frozen));
    freezeBtn.textContent = frozen ? 'Resume live' : 'Freeze';
    if (frozen) setPill('Frozen frame — not live', 'frozen', 'frozen');
    else { lastUpdate = performance.now(); setPill('Resuming…', 'warn', 'starting'); }
  }

  function stop() {
    if (!running) return;
    running = false;
    clearInterval(staleTimer);
    cam.destroy();
    coach.remove(); freezeBtn.remove();
    window.__lumenLive = null;
    onStopped();
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function loop() {
    while (running) {
      if (frozen || cam.state !== 'live') { await sleep(150); continue; }
      const t0 = performance.now();
      let shot;
      try { shot = await cam.grabReduced(maxEdge); } catch (_) { await sleep(250); continue; }
      const pl = state.session.operation ? state.session.operation.placards_resolved.map((p) => p.placard_id).join(',') : '';
      try {
        const res = await upload(`api/sessions/${sid}/track?placards=${encodeURIComponent(pl)}`, shot.blob, {
          meta: { native_width: shot.native[0], native_height: shot.native[1], ...cam.settings(), torch: cam.torchState() }, timeout: 15000,
        });
        failures = 0;
        if (!running) break;
        if (frozen) continue;                 // an answer for a frame taken before the freeze is dropped
        show(shot, res, performance.now() - t0);
      } catch (err) {
        failures += 1;
        if (!running) break;
        const busy = err && err.lumen && err.lumen.code === 'server_busy';
        if (failures >= 2) {
          hideOverlay();
          setPill(busy ? 'Server busy — overlay hidden' : 'Connection problem — overlay hidden', 'warn', 'stale');
          setCoach({ code: 'connection', level: 'warn', text: (err && err.lumen && err.lumen.message) || 'The server did not answer.', arrow: null });
        }
        if (err && err.lumen && (err.status === 404 || err.status === 409)) { toast({ title: err.lumen.message, kind: 'error' }); stop(); break; }
        await sleep(busy ? 1200 : 700);
      }
    }
  }

  setPill('Starting camera…', 'warn', 'starting');
  const ok = await cam.start({ quality: 'standard' });
  if (!ok) { stop(); return null; }
  hideOverlay();
  loop();
  return { stop, toggleFreeze, get frozen() { return frozen; }, get camera() { return cam; }, lastTrackedAt: () => lastTracked };
}

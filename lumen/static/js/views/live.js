// Live placement guidance.
//
// Each camera frame is sent to the server, which measures the tags in *that
// frame* and returns the overlay for it.  The viewer then shows that same
// frame with its overlay.  A target outline is therefore never drawn on a
// picture it was not computed from; a slow connection only lowers the update
// rate.  When a frame cannot be registered the overlay disappears with it,
// and when no answer arrives for a while the last overlay is removed too.
//
// With `placement` (views/placement.js) every frame also asks where the placard
// is: the request carries what the last answer said about the stream, and each
// answer is handed to the placement view, which shows and sounds it.
import { upload } from '../api.js';
import { h, clear, icon, toast } from '../dom.js';
import { state } from '../store.js';
import { Camera } from '../camera.js';
import { coachLine } from './common.js';

const STALE_MS = 2500;
// No more than about twelve frames a second are asked for: beyond that a reading does not get more useful, and the
// server's time belongs to other operators too.
const MIN_INTERVAL_MS = 80;

export async function startLive({ viewer, shell, modePill, onStopped = () => {}, onReport = () => {}, placement = null }) {
  const sid = state.session.id;
  // Placement wants many answers per second, and what limits its uncertainty is not the pixel size of the frame
  // (docs/BUILD_LOG.md): frames are sent smaller there.  The record is taken at the camera's full size.
  const maxEdge = Math.min(placement ? 1280 : 1600, state.config.limits.max_track_long_edge || 1600);
  let running = true;
  let frozen = false;
  let held = false;                // the loop waits (a record photo is being taken); the picture stays
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
      else if (st === 'paused') { hideOverlay(); gap('The camera is paused.'); setPill('Camera paused', 'warn', 'paused'); setCoach({ code: 'paused', level: 'warn', text: `${(info && info.reason) || 'The camera is paused.'} Stop and start live guidance to resume.`, arrow: null }); }
      else if (st === 'error') { toast({ title: (info && info.title) || 'The camera could not be opened', detail: (info && info.detail) || '', kind: 'error', timeout: 9000 }); stop(); }
    },
  });
  window.__lumenLive = { camera: cam, updates: 0, lastResult: null };

  function hideOverlay() {
    viewer.setOverlay({ primitives: [] });
    shownTracking = false;
  }
  // no current reading: the placement view must stop showing and sounding one
  function gap(text) { if (placement) placement.onGap(text); }

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
      // how old the picture on the screen is, and how many readings a second arrive
      const delay = roundTripMs < 950 ? `${Math.max(10, Math.round(roundTripMs / 10) * 10)} ms` : `${(roundTripMs / 1000).toFixed(1)} s`;
      setPill(`LIVE · ${delay} delay${rate ? ` · ${rate.toFixed(rate < 10 ? 1 : 0)}/s` : ''}`, 'live', 'live');
      setCoach(t.code === 'tracking' ? null : { code: t.code, level: t.code === 'tag_moved' ? 'warn' : 'todo', text: t.message, arrow: t.arrow });
    } else {
      hideOverlay();
      setPill('Tracking lost — no overlay', 'warn', 'lost');
      setCoach({ code: t.code, level: 'todo', text: [t.message, ...(t.suggestions || [])].join(' '), arrow: t.arrow });
    }
    if (placement) placement.onResult(res);
    onReport(res);
  }

  // If answers stop coming, the last overlay must not linger on screen.
  const staleTimer = setInterval(() => {
    if (!running || frozen || held || !shownTracking) return;
    if (performance.now() - lastUpdate > STALE_MS) {
      hideOverlay();
      gap('No recent answer from the server.');
      setPill('No recent update — overlay hidden', 'warn', 'stale');
    }
  }, 300);

  function toggleFreeze() {
    frozen = !frozen;
    freezeBtn.setAttribute('aria-pressed', String(frozen));
    freezeBtn.textContent = frozen ? 'Resume live' : 'Freeze';
    if (frozen) { setPill('Frozen frame — not live', 'frozen', 'frozen'); gap('The picture is frozen: not a current reading.'); }
    else { lastUpdate = performance.now(); setPill('Resuming…', 'warn', 'starting'); }
  }
  // Pause the loop without changing what is shown (while a record photo is taken and measured).
  function hold(on) {
    held = !!on;
    if (!held) lastUpdate = performance.now();
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
      if (frozen || held || cam.state !== 'live') { await sleep(held ? 60 : 150); continue; }
      const t0 = performance.now();
      let shot;
      try { shot = await cam.grabReduced(maxEdge); } catch (_) { await sleep(250); continue; }
      const pl = state.session.operation ? state.session.operation.placards_resolved.map((p) => p.placard_id).join(',') : '';
      try {
        const res = await upload(`api/sessions/${sid}/track?placards=${encodeURIComponent(pl)}${placement ? '&placement=true' : ''}`, shot.blob, {
          meta: { native_width: shot.native[0], native_height: shot.native[1], ...cam.settings(), torch: cam.torchState(), ...(placement ? placement.meta() : {}) }, timeout: 15000,
        });
        failures = 0;
        if (!running) break;
        if (frozen || held) continue;         // an answer for a frame taken before the freeze (or the record) is dropped
        const took = performance.now() - t0;
        show(shot, res, took);
        if (took < MIN_INTERVAL_MS) await sleep(MIN_INTERVAL_MS - took);
      } catch (err) {
        failures += 1;
        if (!running) break;
        const busy = err && err.lumen && err.lumen.code === 'server_busy';
        if (failures >= 2) {
          hideOverlay();
          gap(busy ? 'The server is busy.' : 'The server did not answer.');
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
  return { stop, toggleFreeze, hold, get frozen() { return frozen; }, get camera() { return cam; }, lastTrackedAt: () => lastTracked };
}

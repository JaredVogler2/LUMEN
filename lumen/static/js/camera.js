// Rear-camera access for capture and live guidance.
//
// What a web page can and cannot know about the camera
// ----------------------------------------------------
// * getUserMedia gives a video stream.  It does not give lens intrinsics, RAW
//   data or an exposure lock, and the stream may be cropped, binned or
//   stabilised by the browser.  LUMEN therefore records the stream's pixel
//   size, the device label and the zoom with every capture and never mixes
//   different configurations in one 3-D solution.
// * The torch is controllable only where the browser reports the capability.
//   The control is shown only then, and the state written into a capture is
//   "on"/"off" only when the browser confirms it or the picture itself got
//   brighter; otherwise it is "unknown".
// * The camera is stopped whenever the page is hidden or the screen that uses
//   it closes.  Nothing records in the background.

export function cameraSupport() {
  if (!window.isSecureContext) return { ok: false, code: 'insecure', message: 'The camera needs a secure (HTTPS) page.' };
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return { ok: false, code: 'unsupported', message: 'This browser cannot open the camera from a web page.' };
  return { ok: true };
}

export function describeCameraError(err) {
  const name = (err && err.name) || '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return { code: 'denied', title: 'Camera access was refused', detail: 'Allow the camera for this site (Safari: “aA” → Website Settings → Camera → Allow; or Settings → Safari → Camera), then try again. You can still take photos with the phone camera and upload them.' };
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return { code: 'no_camera', title: 'No rear camera was found', detail: 'Use “Choose photos” to upload pictures instead.' };
  if (name === 'NotReadableError' || name === 'AbortError') return { code: 'busy', title: 'The camera is in use or was interrupted', detail: 'Close other apps that use the camera, then try again.' };
  if (name === 'TimeoutError') return { code: 'timeout', title: 'The camera did not start', detail: 'Try again. If a permission prompt is open, answer it first.' };
  return { code: 'error', title: 'The camera could not be opened', detail: (err && err.message) || 'Unknown error.' };
}

const QUALITY = {
  high: { width: { ideal: 3840 }, height: { ideal: 2160 } },
  standard: { width: { ideal: 1920 }, height: { ideal: 1080 } },
};

function withTimeout(promise, ms, onLate) {
  return new Promise((resolve, reject) => {
    let done = false;
    const timer = setTimeout(() => { if (!done) { done = true; reject(new DOMException('timed out', 'TimeoutError')); } }, ms);
    promise.then((v) => {
      if (done) { if (onLate) onLate(v); return; }
      done = true; clearTimeout(timer); resolve(v);
    }, (e) => { if (!done) { done = true; clearTimeout(timer); reject(e); } });
  });
}

// A lens that never switches is better than a clever one: virtual multi-lens
// devices change lens (and intrinsics) with distance and zoom.
export function preferredRearCamera(devices) {
  const cams = devices.filter((d) => d.kind === 'videoinput');
  const rear = cams.filter((d) => /back|rear|environment/i.test(d.label));
  const plain = rear.find((d) => /^back camera$/i.test(d.label.trim()));
  if (plain) return plain;
  const single = rear.find((d) => !/dual|triple|ultra|tele|wide angle/i.test(d.label));
  return single || rear[0] || null;
}

export class Camera {
  constructor({ onState = null, onInterrupted = null } = {}) {
    this.video = document.createElement('video');
    this.video.setAttribute('playsinline', '');
    this.video.setAttribute('webkit-playsinline', '');
    this.video.muted = true;
    this.video.autoplay = true;
    this.video.disablePictureInPicture = true;
    this.stream = null;
    this.track = null;
    this.state = 'idle';            // idle | starting | live | paused | error
    this.quality = 'high';
    this.deviceId = null;
    this.onState = onState;
    this.onInterrupted = onInterrupted;
    this.torch = { supported: false, requested: false, reported: null, verified: false };
    this.recorder = null;
    this._probe = document.createElement('canvas');
    this._onVis = () => { if (document.hidden && this.state === 'live') this.pause('The camera was stopped because the page went to the background.'); };
    document.addEventListener('visibilitychange', this._onVis);
  }

  _set(state, info) { this.state = state; if (this.onState) this.onState(state, info || null); }

  async start({ quality = this.quality, deviceId = this.deviceId } = {}) {
    const sup = cameraSupport();
    if (!sup.ok) { this._set('error', { code: sup.code, title: sup.message, detail: 'Use “Choose photos” to upload pictures instead.' }); return false; }
    this.stopTracks();
    this.quality = quality;
    this._set('starting');
    const video = { ...QUALITY[quality] };
    if (deviceId) video.deviceId = { exact: deviceId }; else video.facingMode = { ideal: 'environment' };
    try {
      const stream = await withTimeout(navigator.mediaDevices.getUserMedia({ audio: false, video }), 20000, (late) => late.getTracks().forEach((t) => t.stop()));
      this.stream = stream;
      this.track = stream.getVideoTracks()[0];
      this.deviceId = deviceId || null;
      this.video.srcObject = stream;
      await withTimeout(this.video.play(), 8000);
      await this._waitForFrames();
      this.track.addEventListener('ended', () => this._interrupted('The camera stream ended.'));
      this.track.addEventListener('mute', () => { this._muteTimer = setTimeout(() => { if (this.track && this.track.muted) this._interrupted('The camera was interrupted.'); }, 1500); });
      this.track.addEventListener('unmute', () => clearTimeout(this._muteTimer));
      this._probeTorch();
      this._set('live');
      return true;
    } catch (err) {
      this.stopTracks();
      this._set('error', describeCameraError(err));
      return false;
    }
  }

  _waitForFrames() {
    return withTimeout(new Promise((resolve) => {
      const ok = () => this.video.videoWidth > 0 && this.video.readyState >= 2;
      if (ok()) { resolve(); return; }
      const tick = () => { if (ok()) resolve(); else requestAnimationFrame(tick); };
      tick();
    }), 8000);
  }

  _interrupted(why) {
    if (this.state !== 'live') return;
    this.stopTracks();
    this._set('paused', { reason: why });
    if (this.onInterrupted) this.onInterrupted(why);
  }

  pause(why = 'Camera paused.') {
    this.stopTracks();
    this._set('paused', { reason: why });
  }

  stopTracks() {
    clearTimeout(this._muteTimer);
    if (this.recorder && this.recorder.state !== 'inactive') { try { this.recorder.stop(); } catch (_) { /* ignore */ } }
    this.recorder = null;
    if (this.stream) this.stream.getTracks().forEach((t) => { try { t.stop(); } catch (_) { /* ignore */ } });
    this.stream = null; this.track = null;
    this.video.srcObject = null;
    this.torch = { supported: false, requested: false, reported: null, verified: false };
  }

  destroy() {
    document.removeEventListener('visibilitychange', this._onVis);
    this.stopTracks();
    this._set('idle');
    this.onState = null; this.onInterrupted = null;
  }

  // ------------------------------------------------------------- settings
  settings() {
    const st = this.track && this.track.getSettings ? this.track.getSettings() : {};
    return {
      width: this.video.videoWidth, height: this.video.videoHeight,
      frame_rate: st.frameRate || null, facing_mode: st.facingMode || '', zoom: typeof st.zoom === 'number' ? st.zoom : null,
      camera_label: (this.track && this.track.label) || '',
    };
  }
  configKey() {
    const s = this.settings();
    return `${s.camera_label}|${s.width}x${s.height}|${s.zoom === null ? '' : s.zoom}`;
  }
  async rearCameras() {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      return all.filter((d) => d.kind === 'videoinput' && d.label);
    } catch (_) { return []; }
  }
  // If the browser picked a lens-switching virtual camera, offer the fixed one.
  async stableAlternative() {
    const cams = await this.rearCameras();
    const best = preferredRearCamera(cams);
    if (!best || !this.track) return null;
    const cur = this.track.getSettings ? this.track.getSettings().deviceId : null;
    if (best.deviceId === cur) return null;
    if (/dual|triple/i.test(this.track.label || '')) return best;
    return null;
  }

  // ---------------------------------------------------------------- torch
  _probeTorch() {
    let supported = false;
    try {
      const caps = this.track.getCapabilities ? this.track.getCapabilities() : null;
      supported = !!(caps && 'torch' in caps && caps.torch !== false && !(Array.isArray(caps.torch) && !caps.torch.includes(true)));
    } catch (_) { supported = false; }
    this.torch = { supported, requested: false, reported: null, verified: false };
  }

  meanLuma() {
    const v = this.video;
    if (!v.videoWidth) return null;
    const c = this._probe;
    c.width = 48; c.height = 36;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(v, 0, 0, c.width, c.height);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
    return sum / (d.length / 4) / 255;
  }

  // Ask for the torch and find out, as well as a web page can, whether it lit.
  async setTorch(on, { settleMs = 650 } = {}) {
    if (!this.torch.supported || !this.track) return this.torchState();
    const before = this.meanLuma();
    try {
      await this.track.applyConstraints({ advanced: [{ torch: !!on }] });
    } catch (_) {
      this.torch = { ...this.torch, requested: false, reported: null, verified: false, failed: true };
      return this.torchState();
    }
    await new Promise((r) => setTimeout(r, settleMs));
    let reported = null;
    try { const st = this.track.getSettings(); if (typeof st.torch === 'boolean') reported = st.torch; } catch (_) { reported = null; }
    const after = this.meanLuma();
    const brighter = before !== null && after !== null && after > before * 1.06 + 0.004;
    const darker = before !== null && after !== null && after < before * 0.94 - 0.004;
    this.torch = { supported: true, requested: !!on, reported, verified: on ? brighter : darker, failed: false };
    return this.torchState();
  }

  // 'on' / 'off' only when there is evidence; 'unknown' otherwise.
  torchState() {
    const t = this.torch;
    if (!t.supported) return 'unknown';
    if (t.reported !== null) return t.reported ? 'on' : 'off';
    if (t.requested) return t.verified ? 'on' : 'unknown';
    return t.failed ? 'unknown' : 'off';
  }
  torchEvidence() {
    const t = this.torch;
    if (!t.supported) return 'not available in this browser';
    if (t.failed) return 'the browser refused the request';
    if (t.reported !== null) return t.verified ? 'reported by the browser and seen in the picture' : 'reported by the browser';
    if (t.requested) return t.verified ? 'seen in the picture' : 'requested, not confirmed';
    return 'off';
  }

  // --------------------------------------------------------------- capture
  // Still of the current frame at the stream's full pixel size.
  async grab({ quality = 0.95 } = {}) {
    const v = this.video;
    if (this.state !== 'live' || !v.videoWidth) throw new Error('The camera is not running.');
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const blob = await new Promise((ok, bad) => c.toBlob((b) => (b ? ok(b) : bad(new Error('Could not encode the photo.'))), 'image/jpeg', quality));
    c.width = 1; c.height = 1;
    return { blob, meta: { ...this.settings(), torch: this.torchState(), captured_at: new Date().toISOString(), capture_mode: 'browser_stream' } };
  }

  // Frame reduced to `maxEdge` for live tracking; returns the scale used.
  async grabReduced(maxEdge, quality = 0.85) {
    const v = this.video;
    if (this.state !== 'live' || !v.videoWidth) throw new Error('The camera is not running.');
    const k = Math.min(1, maxEdge / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement('canvas');
    c.width = Math.round(v.videoWidth * k); c.height = Math.round(v.videoHeight * k);
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
    const blob = await new Promise((ok, bad) => c.toBlob((b) => (b ? ok(b) : bad(new Error('Could not encode the frame.'))), 'image/jpeg', quality));
    let bitmap = null;
    try { bitmap = await createImageBitmap(c); } catch (_) { bitmap = null; }
    return { blob, bitmap, canvas: bitmap ? null : c, width: c.width, height: c.height, native: [v.videoWidth, v.videoHeight] };
  }

  static recordingType() {
    if (!('MediaRecorder' in window)) return null;
    const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    return types.find((t) => { try { return MediaRecorder.isTypeSupported(t); } catch (_) { return false; } }) || null;
  }

  startRecording({ maxSeconds = 30, onTick = null } = {}) {
    const type = Camera.recordingType();
    if (!type || !this.stream) throw new Error('This browser cannot record a sweep; take photos instead.');
    const chunks = [];
    const rec = new MediaRecorder(this.stream, { mimeType: type, videoBitsPerSecond: 14_000_000 });
    this.recorder = rec;
    const t0 = performance.now();
    const meta = { ...this.settings(), torch: this.torchState(), captured_at: new Date().toISOString(), capture_mode: 'browser_recording' };
    let timer = null;
    const done = new Promise((resolve, reject) => {
      rec.ondataavailable = (ev) => { if (ev.data && ev.data.size) chunks.push(ev.data); };
      rec.onerror = () => { clearInterval(timer); reject(new Error('Recording failed.')); };
      rec.onstop = () => {
        clearInterval(timer);
        const blob = new Blob(chunks, { type: type.split(';')[0] });
        resolve({ blob, meta, seconds: (performance.now() - t0) / 1000 });
      };
    });
    timer = setInterval(() => {
      const s = (performance.now() - t0) / 1000;
      if (onTick) onTick(s, maxSeconds);
      if (s >= maxSeconds && rec.state === 'recording') rec.stop();
    }, 200);
    rec.start(500);
    return { done, stop: () => { if (rec.state !== 'inactive') rec.stop(); } };
  }
}

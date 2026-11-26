// Sound for assisted placement.
//
// While a placard is being positioned the operator looks at the placard, not
// at the phone, so the reading is also heard:
//
//   beeps, faster and higher the nearer the placard is   -> keep moving
//   a steady two-note tone                               -> in position
//   a rising chime                                       -> the placement was recorded
//   a falling pair                                       -> it left its position / the record did not confirm it
//
// A sound only ever follows a reading.  The caller renews it with every result
// (`guide`, `hold`); when no result arrives for STALE_MS the sound stops by
// itself, so a reading that is no longer current cannot keep saying "in
// position".
//
// A web page may start audio only from a tap: `unlock()` is called by the
// button that starts live placement.  On a phone the ring/silent switch and
// the media volume decide whether anything is heard; the page cannot know.

const STALE_MS = 1300;

export class Feedback {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.muted = false;
    this.mode = 'silent';          // silent | guide | hold
    this.interval = 800;
    this.pitch = 660;
    this.renewed = 0;
    this._timer = 0;
    this._hold = null;
    this.log = [];                 // what was sounded, newest last (also read by the browser tests)
  }

  get available() { return !!(window.AudioContext || window.webkitAudioContext); }
  get running() { return !!this.ctx && this.ctx.state === 'running'; }

  // Must be called from a tap.
  unlock() {
    if (!this.available) return false;
    try {
      if (!this.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AC({ latencyHint: 'interactive' });
        this.master = this.ctx.createGain();
        this.master.gain.value = this.muted ? 0 : 0.9;
        this.master.connect(this.ctx.destination);
      }
      // iPhone: play although the ring switch is set to silent, where the browser lets a page say so
      try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch (_) { /* not supported */ }
      if (this.ctx.state !== 'running') this.ctx.resume().catch(() => {});
      // a tone too short and quiet to hear: some browsers only start the clock with the first node
      this._tone(440, 0, 0.02, 0.0001);
      return true;
    } catch (_) { this.ctx = null; return false; }
  }

  setMuted(on) {
    this.muted = !!on;
    if (this.master) this.master.gain.setTargetAtTime(this.muted ? 0 : 0.9, this.ctx.currentTime, 0.01);
    if (this.muted) this._stopHold();
  }

  _note(kind, extra) {
    this.log.push({ t: Math.round(performance.now()), kind, muted: this.muted, ...(extra || {}) });
    if (this.log.length > 200) this.log.splice(0, this.log.length - 200);
  }

  _tone(freq, delay, dur, peak, type = 'sine') {
    if (!this.ctx || !this.master) return;
    const t0 = this.ctx.currentTime + delay;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = type; osc.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(Math.max(peak, 0.0002), t0 + 0.008);
    g.gain.setValueAtTime(Math.max(peak, 0.0002), t0 + Math.max(dur - 0.05, 0.01));
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(g); g.connect(this.master);
    osc.start(t0); osc.stop(t0 + dur + 0.02);
  }

  _tick() {
    clearTimeout(this._timer);
    if (this.mode !== 'guide') return;
    if (performance.now() - this.renewed > STALE_MS) { this.silence('stale'); return; }
    this._tone(this.pitch, 0, Math.min(0.09, this.interval / 1000 * 0.45), 0.5, 'square');
    this._timer = setTimeout(() => this._tick(), this.interval);
  }

  // `q`: how far the placard is from where it must be, in units of what is allowed (0 = on target, 1 = at the limit).
  guide(q) {
    const x = Math.min(Math.max(Number.isFinite(q) ? q : 50, 0), 50);
    // 1100 ms between beeps far away, 150 ms at the limit of the tolerance
    this.interval = Math.round(Math.min(Math.max(150 + 520 * Math.log2(1 + Math.max(x - 1, 0) / 2.5), 150), 1100));
    this.pitch = Math.round(Math.min(Math.max(1040 - 150 * Math.log2(1 + x), 520), 1040));
    this.renewed = performance.now();
    if (this.mode === 'guide') return;
    this._stopHold();
    this.mode = 'guide';
    this._note('guide', { interval: this.interval });
    this._tick();
  }

  // In position: a steady tone until the next result says otherwise or none arrives.
  hold() {
    this.renewed = performance.now();
    if (this.mode === 'hold') return;
    clearTimeout(this._timer);
    this.mode = 'hold';
    this._note('hold');
    if (!this.ctx || !this.master) return;
    const t = this.ctx.currentTime;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.16, t + 0.05);
    g.connect(this.master);
    const oscs = [880, 1320].map((f) => { const o = this.ctx.createOscillator(); o.type = 'sine'; o.frequency.value = f; o.connect(g); o.start(t); return o; });
    this._hold = { g, oscs };
    const watch = () => {
      if (this.mode !== 'hold') return;
      if (performance.now() - this.renewed > STALE_MS) { this.silence('stale'); return; }
      this._timer = setTimeout(watch, 150);
    };
    this._timer = setTimeout(watch, 150);
  }

  _stopHold() {
    const hd = this._hold;
    this._hold = null;
    if (!hd || !this.ctx) return;
    const t = this.ctx.currentTime;
    try {
      hd.g.gain.cancelScheduledValues(t);
      hd.g.gain.setValueAtTime(Math.max(hd.g.gain.value, 0.0002), t);
      hd.g.gain.exponentialRampToValueAtTime(0.0001, t + 0.06);
      hd.oscs.forEach((o) => o.stop(t + 0.08));
    } catch (_) { /* already stopped */ }
  }

  silence(why = '') {
    clearTimeout(this._timer);
    this._stopHold();
    if (this.mode !== 'silent') this._note('silent', why ? { why } : null);
    this.mode = 'silent';
  }

  // One-off signals.  They do not change the running mode.
  chime(kind) {
    this._note(`chime:${kind}`);
    const seq = {
      in: [[1047, 0, 0.11], [1568, 0.1, 0.16]],                                  // reached its position
      out: [[660, 0, 0.12], [440, 0.12, 0.2]],                                   // left it again
      recorded: [[1047, 0, 0.14], [1319, 0.13, 0.14], [1568, 0.26, 0.14], [2093, 0.39, 0.42]],
      fail: [[392, 0, 0.16], [311, 0.17, 0.3]],                                  // the record did not confirm it
    }[kind] || [];
    seq.forEach(([f, d, len]) => this._tone(f, d, len, 0.5, 'triangle'));
    try {
      if (!this.muted && navigator.vibrate) navigator.vibrate(kind === 'recorded' ? [70, 60, 70, 60, 180] : kind === 'in' ? [60] : kind === 'fail' ? [200] : []);
    } catch (_) { /* no vibration here */ }
  }

  // Stops everything that follows a reading at once; a chime that was just started is allowed to end.
  close() {
    this.silence();
    const ctx = this.ctx;
    this.ctx = null; this.master = null;
    if (ctx && ctx.close) setTimeout(() => { ctx.close().catch(() => {}); }, 1200);
  }
}

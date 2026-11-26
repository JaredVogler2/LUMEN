// The carrier: where the server keeps nothing between two requests, this browser carries the job.
//
// On a function platform a request may be served by a copy of the program that has never seen the job.  So the
// job stays here: its state (a document the server wrote and only the server reads) and its pictures.  Every
// request for the job goes to one address, api/carry, wrapped in an envelope together with the state and the
// pictures the step needs; the answer brings the state back as the request left it (storage/carried.py and
// api/carry.py are the other end).
//
// Nothing outside api.js knows about this.  The views ask for "api/sessions/<id>/..." as they always did.
//
// What is kept here is kept for the job: in memory, and in the browser's own database so that a reload (which a
// phone does by itself when memory is short) does not lose the photos.  It is removed when the job ends, and a
// job that was left behind is removed the next time the page starts.

const MAGIC = [0x4c, 0x55, 0x4d, 0x43, 0x01];           // "LUMC", version 1
const CARRY_TYPE = 'application/octet-stream';   // bytes, declared as nothing more: every platform passes that untouched
const DB_NAME = 'lumen-carry';
const LEFT_BEHIND_MS = 2 * 3600 * 1000;       // photos are for the job: one that was left behind does not stay on the phone
const enc = new TextEncoder();
const dec = new TextDecoder();

let cfg = null;                  // config.sessions of a server that leaves the job to the browser
let resolveUrl = (p) => p;
const jobs = new Map();          // id -> job
let dbPromise = null;

export const stats = { requests: 0, picturesSent: 0, askedAgain: 0, bytesSent: 0 };
// The last requests, for the diagnostics sheet (what was asked, how much was sent, how long it took, how it ended).
export const recent = [];
function note(entry) {
  recent.push(entry);
  if (recent.length > 40) recent.shift();
}

export function configure(sessions, { url }) {
  cfg = sessions && sessions.kept_by === 'client' ? { ...(cfg || {}), ...sessions } : null;
  resolveUrl = url;
  if (cfg && !dbPromise) { dbPromise = openDb(); sweep(); }
}
export const active = () => !!cfg;
export const limits = () => cfg;

// A request that belongs to a job: a session or a calibration workspace (the printable board is not one).
export function isJobPath(path) {
  const p = String(path).replace(/^\/+/, '');
  return /^api\/(sessions|calibration)(\/|\?|$)/.test(p) && !/^api\/calibration\/board\.png/.test(p);
}
function idOf(path) {
  const m = /^api\/(?:sessions|calibration)\/([A-Za-z0-9_-]{16,64})(?:\/|\?|$)/.exec(String(path).replace(/^\/+/, ''));
  return m ? m[1] : null;
}

// ---------------------------------------------------------------- the browser's database (optional)
function openDb() {
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('jobs')) db.createObjectStore('jobs', { keyPath: 'id' });
        if (!db.objectStoreNames.contains('pictures')) db.createObjectStore('pictures', { keyPath: 'key' }).createIndex('job', 'id');
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch (_) { resolve(null); }      // private mode, or no database at all: the job lives in memory only
  });
}
function tx(db, stores, mode, fn) {
  return new Promise((resolve) => {
    try {
      const t = db.transaction(stores, mode);
      let out;
      t.oncomplete = () => resolve(out);
      t.onerror = () => resolve(undefined);
      t.onabort = () => resolve(undefined);
      out = fn(t);
    } catch (_) { resolve(undefined); }
  });
}
async function persistState(job) {
  const db = await dbPromise;
  if (db) await tx(db, ['jobs'], 'readwrite', (t) => { t.objectStore('jobs').put({ id: job.id, state: job.state, sig: job.sig, touched: Date.now() }); });
}
// A picture is written as plain bytes: not every browser's database stores a Blob.
async function persistPicture(id, name, blob) {
  const db = await dbPromise;
  if (!db) return;
  let data;
  try { data = await blob.arrayBuffer(); } catch (_) { return; }
  await tx(db, ['pictures'], 'readwrite', (t) => { t.objectStore('pictures').put({ key: `${id}/${name}`, id, name, type: blob.type || 'image/jpeg', data }); });
}
async function forgetPicture(id, name) {
  const db = await dbPromise;
  if (db) await tx(db, ['pictures'], 'readwrite', (t) => { t.objectStore('pictures').delete(`${id}/${name}`); });
}
async function forgetJob(id) {
  const db = await dbPromise;
  if (!db) return;
  await tx(db, ['jobs', 'pictures'], 'readwrite', (t) => {
    t.objectStore('jobs').delete(id);
    const cur = t.objectStore('pictures').index('job').openCursor(IDBKeyRange.only(id));
    cur.onsuccess = () => { const c = cur.result; if (c) { c.delete(); c.continue(); } };
  });
}
// Jobs that were left behind (a closed tab, a phone put away) go when the page next starts: photos are not kept.
async function sweep() {
  const db = await dbPromise;
  if (!db) return;
  const old = [];
  await tx(db, ['jobs'], 'readonly', (t) => {
    const cur = t.objectStore('jobs').openCursor();
    cur.onsuccess = () => { const c = cur.result; if (c) { if (Date.now() - (c.value.touched || 0) > LEFT_BEHIND_MS) old.push(c.value.id); c.continue(); } };
  });
  for (const id of old) { if (!jobs.has(id)) await forgetJob(id); }
}

async function job(id, create = false) {
  let j = jobs.get(id);
  if (j) return j;
  const db = await dbPromise;
  if (db) {
    let rec;
    const pics = [];
    await tx(db, ['jobs', 'pictures'], 'readonly', (t) => {
      const g = t.objectStore('jobs').get(id);
      g.onsuccess = () => { rec = g.result; };
      const cur = t.objectStore('pictures').index('job').openCursor(IDBKeyRange.only(id));
      cur.onsuccess = () => { const c = cur.result; if (c) { pics.push(c.value); c.continue(); } };
    });
    j = jobs.get(id);                   // another request may have loaded it meanwhile
    if (j) return j;
    if (rec && rec.state) {
      j = newJob(id);
      j.state = new Uint8Array(rec.state);
      j.sig = rec.sig || '';
      pics.forEach((p) => { if (p.data) j.pictures.set(p.name, new Blob([p.data], { type: p.type || 'image/jpeg' })); });
      jobs.set(id, j);
      return j;
    }
  }
  if (!create) return null;
  j = newJob(id);
  jobs.set(id, j);
  return j;
}
function newJob(id) {
  return { id, state: null, sig: '', rev: 0, pictures: new Map(), urls: new Map(), queue: Promise.resolve(), packed: null, doc: null };
}

// What the state says about the job's pictures (read only; the state is the server's document).
function docOf(j) {
  if (j.doc && j.doc.rev === j.rev) return j.doc.value;
  let value = {};
  try { value = JSON.parse(dec.decode(j.state)); } catch (_) { value = {}; }
  j.doc = { rev: j.rev, value };
  return value;
}
function frameOf(j, fid) {
  return ((docOf(j).frames || []).find((f) => f.id === fid)) || null;
}

// ---------------------------------------------------------------- envelope
function pack(header, parts) {
  const h = enc.encode(JSON.stringify(header));
  const pre = new Uint8Array(MAGIC.length + 4);
  pre.set(MAGIC);
  new DataView(pre.buffer).setUint32(MAGIC.length, h.length);
  return new Blob([pre, h, ...parts], { type: CARRY_TYPE });
}
function isEnvelope(buffer) {
  const u8 = new Uint8Array(buffer || new ArrayBuffer(0));
  return u8.length >= MAGIC.length + 4 && MAGIC.every((b, i) => u8[i] === b);
}
function unpack(buffer) {
  const u8 = new Uint8Array(buffer);
  if (!isEnvelope(buffer)) throw new Error('not an envelope');
  const n = new DataView(buffer).getUint32(MAGIC.length);
  const start = MAGIC.length + 4;
  const header = JSON.parse(dec.decode(u8.subarray(start, start + n)));
  return { header, rest: u8.subarray(start + n) };
}
async function deflated(j) {
  if (j.packed && j.packed.rev === j.rev) return j.packed;
  let out = { rev: j.rev, bytes: j.state, encoding: 'identity' };
  if (typeof CompressionStream === 'function') {
    try {
      const buf = await new Response(new Blob([j.state]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer();
      out = { rev: j.rev, bytes: new Uint8Array(buf), encoding: 'deflate' };
    } catch (_) { /* sent as it is */ }
  }
  j.packed = out;
  return out;
}

function post(blob, { onProgress, timeout = 60000, signal } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', resolveUrl('api/carry'));
    xhr.responseType = 'arraybuffer';
    xhr.timeout = timeout;
    xhr.setRequestHeader('Content-Type', CARRY_TYPE);
    if (onProgress) xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) onProgress(ev.loaded / ev.total); };
    xhr.onload = () => resolve({ status: xhr.status, type: xhr.getResponseHeader('content-type') || '', buffer: xhr.response });
    xhr.onerror = () => reject(Object.assign(new Error('network'), { kind: 'network' }));
    xhr.ontimeout = () => reject(Object.assign(new Error('timeout'), { kind: 'timeout' }));
    xhr.onabort = () => reject(new DOMException('aborted', 'AbortError'));
    if (signal) signal.addEventListener('abort', () => xhr.abort(), { once: true });
    xhr.send(blob);
  });
}

// ---------------------------------------------------------------- which pictures a step needs
// Steps that look at every photo of the capture (the lines are searched in each of them and checked against all).
const ALL_PHOTOS = /\/(datums\/propose|datums\/select|datums\/(floor|side)|calibration|research\/photometric)(\?|$)/;
function picturesFor(method, path, j, json) {
  const p = String(path).replace(/^\/+/, '');
  const names = new Set();
  if (!j || !j.state) return names;
  const doc = docOf(j);
  const frames = doc.frames || [];
  if (/^api\/sessions\//.test(p)) {
    if (method !== 'GET' && method !== 'DELETE' && ALL_PHOTOS.test(p)) {
      frames.filter((f) => f.role === 'registration').forEach((f) => names.add(f.blob));
    }
    const m = /\/frames\/([A-Za-z0-9_-]+)\/(rectified|image)(\?|$)/.exec(p);
    if (m) { const f = frameOf(j, m[1]); if (f) names.add(f.blob); }
    if (method === 'POST' && /\/verify(\?|$)/.test(p) && json && json.frame_id) { const f = frameOf(j, json.frame_id); if (f) names.add(f.blob); }
  } else {
    const m = /\/photos\/([A-Za-z0-9_-]+)\/image(\?|$)/.exec(p);
    if (m) { const v = (doc.views || []).find((x) => x.observation && x.observation.view_id === m[1]); if (v && v.blob) names.add(v.blob); }
  }
  return new Set([...names].filter((n) => j.pictures.has(n)));
}
// Requests that only read the job: every GET, and the live frame (a POST because it brings a picture).  They need
// not wait for a request that changes the job, and what they bring back is never taken for the job's state when the
// job has changed meanwhile.
const onlyReads = (method, path) => method === 'GET' || /\/track(\?|$)/.test(String(path));

// ---------------------------------------------------------------- one request
// Returns { ok: true, status, json | blob, headers } or { ok: false, status, body }.
export async function request(method, path, { json, body, contentType, meta, onProgress, timeout, signal, expect = 'json' } = {}) {
  const id = idOf(path);
  const j = id ? await job(id) : null;
  const run = () => attempt(method, path, j, { json, body, contentType, meta, onProgress, timeout, signal, expect });
  if (j && !onlyReads(method, path)) {
    // one change of a job at a time: each starts from the state the one before left
    const next = j.queue.then(run, run);
    j.queue = next.then(() => undefined, () => undefined);
    return next;
  }
  return run();
}

async function attempt(method, path, j, o) {
  let payload = new Blob([]);
  let ctype = o.contentType || '';
  if (o.json !== undefined) { payload = new Blob([JSON.stringify(o.json)]); ctype = 'application/json'; } else if (o.body) { payload = o.body; ctype = ctype || o.body.type || 'application/octet-stream'; }
  let names = picturesFor(method, path, j, o.json);
  for (let round = 0; round < 5; round += 1) {
    const baseRev = j ? j.rev : 0;
    const header = { method, path: String(path).replace(/^\/+/, ''), content_type: ctype, states: [], blobs: [], body: payload.size };
    if (o.meta) header.meta = encodeURIComponent(JSON.stringify(o.meta));
    const parts = [];
    if (j && j.state) {
      const z = await deflated(j);
      header.states.push({ id: j.id, size: z.bytes.length, encoding: z.encoding, sig: j.sig || undefined });
      parts.push(z.bytes);
      for (const name of names) {
        const pic = j.pictures.get(name);
        header.blobs.push({ id: j.id, name, size: pic.size });
        parts.push(pic);
      }
      stats.picturesSent += names.size;
    }
    parts.push(payload);
    const envelope = pack(header, parts);
    if (cfg && cfg.max_request_bytes && envelope.size > cfg.max_request_bytes) {
      return { ok: false, status: 413, body: { error: { code: 'request_too_large', message: `This step would send ${(envelope.size / 1e6).toFixed(1)} MB at once, more than this server takes (${(cfg.max_request_bytes / 1e6).toFixed(1)} MB).`, details: {}, suggestions: ['Use fewer photos: remove one and analyse again.'] } } };
    }
    stats.requests += 1;
    stats.bytesSent += envelope.size;
    const started = performance.now();
    const log = { method, path: header.path.split('?')[0].replace(/[A-Za-z0-9_-]{16,64}/, '…'), bytes: envelope.size, pictures: names.size, status: null, ms: null };
    note(log);
    let res;
    try {
      res = await post(envelope, o);
      log.status = res.status;
      log.ms = Math.round(performance.now() - started);
    } catch (err) {
      log.status = (err && (err.kind || err.name)) || 'failed';
      log.ms = Math.round(performance.now() - started);
      if (err && err.name === 'AbortError') throw err;
      const timeoutErr = err && err.kind === 'timeout';
      return { ok: false, status: 0, body: { error: { code: timeoutErr ? 'timeout' : 'network', message: timeoutErr ? 'The server did not answer in time.' : 'Cannot reach the server.', details: {}, suggestions: ['Check the connection and retry.'] } } };
    }
    if (res.status === 200 && isEnvelope(res.buffer)) return take(res.buffer, j, baseRev, payload, method, path, o.expect);
    let errBody = null;
    try { errBody = JSON.parse(dec.decode(new Uint8Array(res.buffer))); } catch (_) { errBody = null; }
    const code = errBody && errBody.error && errBody.error.code;
    if (res.status === 428 && code === 'pictures_needed' && j) {
      // The step needs a picture that was not sent.  Send what it named; where it will want all, all.
      stats.askedAgain += 1;
      const before = names.size;
      (errBody.error.details.blobs || []).forEach((n) => { if (j.pictures.has(n)) names.add(n); });
      if (names.size === before) {
        return { ok: false, status: 409, body: { error: { code: 'pictures_gone', message: 'A photo of this job is no longer on this phone.', details: {}, suggestions: ['Take the photos again.'] } } };
      }
      continue;
    }
    if (res.status === 413 && !errBody) {
      errBody = { error: { code: 'request_too_large', message: 'This step sent more at once than this server takes.', details: {}, suggestions: ['Use fewer photos: remove one and analyse again.'] } };
    }
    return { ok: false, status: res.status, body: errBody };
  }
  return { ok: false, status: 409, body: { error: { code: 'pictures_needed', message: 'The server kept asking for photos of this job.', details: {}, suggestions: ['Reload the page and try again.'] } } };
}

// Take an answer apart and keep what it brings for the job.
async function take(buffer, j, baseRev, payload, method, path, expect) {
  const { header, rest } = unpack(buffer);
  let pos = 0;
  const cut = (n) => { const out = rest.subarray(pos, pos + n); pos += n; return out; };
  const writes = [];
  for (const st of header.states || []) {
    const bytes = cut(st.size).slice();
    let t = jobs.get(st.id);
    if (!t) { t = newJob(st.id); jobs.set(st.id, t); }
    // a reading request that comes back after the job has changed does not put the older state in its place
    if (t === j && onlyReads(method, path) && j.rev !== baseRev) continue;
    t.state = bytes;
    t.sig = st.sig || '';
    t.rev += 1;
    writes.push(persistState(t));
  }
  for (const b of header.blobs || []) {
    const t = jobs.get(b.id);
    const blob = new Blob([cut(b.size).slice()], { type: /\.png$/.test(b.name) ? 'image/png' : 'image/jpeg' });
    if (t) { keep(t, b.name, blob); writes.push(persistPicture(b.id, b.name, blob)); }
  }
  for (const k of header.kept || []) {
    const t = jobs.get(k.id);
    if (t) { keep(t, k.name, payload); writes.push(persistPicture(k.id, k.name, payload)); }
  }
  for (const d of header.dropped || []) {
    const t = jobs.get(d.id);
    if (t) { drop(t, d.name); writes.push(forgetPicture(d.id, d.name)); }
  }
  for (const gone of header.removed || []) {
    const t = jobs.get(gone);
    if (t) { [...t.pictures.keys()].forEach((n) => drop(t, n)); jobs.delete(gone); }
    writes.push(forgetJob(gone));
  }
  const inner = cut(header.body || 0);
  // what was learnt is written down before the answer is handed on: a reload right after must find it
  await Promise.all(writes);
  const headers = header.headers || {};
  const out = { ok: true, status: header.status, headers: { get: (name) => (headers[String(name).toLowerCase()] ?? null) } };
  if (expect === 'blob') out.blob = new Blob([inner.slice()], { type: header.content_type || 'application/octet-stream' });
  else if ((header.content_type || '').includes('application/json')) { try { out.json = JSON.parse(dec.decode(inner)); } catch (_) { out.json = null; } } else out.json = null;
  return out;
}
function keep(j, name, blob) {
  drop(j, name);
  j.pictures.set(name, blob);
}
function drop(j, name) {
  const u = j.urls.get(name);
  if (u) { URL.revokeObjectURL(u); j.urls.delete(name); }
  j.pictures.delete(name);
}

// ---------------------------------------------------------------- pictures the browser already has
// A photo that the server would hand back exactly as it was sent need not be asked for: it is here.
// (The browser prepares every photo itself, upright and without EXIF, so this is the usual case.)
export async function localPicture(path) {
  const p = String(path).replace(/^\/+/, '');
  const id = idOf(p);
  if (!id) return null;
  const j = await job(id);
  if (!j || !j.state) return null;
  let name = null;
  const m = /^api\/sessions\/[^/]+\/frames\/([A-Za-z0-9_-]+)\/image(\?|$)/.exec(p);
  if (m) {
    const f = frameOf(j, m[1]);
    if (f && f.processing_scale === 1 && !f.orientation_applied && /\.jpe?g$/.test(f.blob || '')) name = f.blob;
  }
  // a board photo of a lens calibration is only ever shown small: the browser's copy does
  const c = /^api\/calibration\/[^/]+\/photos\/([A-Za-z0-9_-]+)\/image(\?|$)/.exec(p);
  if (c) {
    const v = (docOf(j).views || []).find((x) => x.observation && x.observation.view_id === c[1]);
    if (v && v.blob) name = v.blob;
  }
  const blob = name ? j.pictures.get(name) : null;
  if (!blob) return null;
  let u = j.urls.get(name);
  if (!u) { u = URL.createObjectURL(blob); j.urls.set(name, u); }
  return { blob, url: u };
}

// ---------------------------------------------------------------- photos made to fit
// All photos of a capture travel in one request when the lines are searched, so each has a size it may have.
// The photo is drawn upright onto a canvas, reduced, and written as JPEG without its EXIF data.
export async function fitPhoto(file, { maxEdge, maxBytes }) {
  let src;
  try { src = await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch (_) { src = await viaImage(file); }
  let w = src.width, h = src.height;
  const s0 = Math.min(1, maxEdge / Math.max(w, h));
  let tw = Math.max(1, Math.round(w * s0)), th = Math.max(1, Math.round(h * s0));
  // reduce in halves first: one large step would skip pixels and soften or alias the tag edges
  let cur = src;
  while (w > 2 * tw && h > 2 * th) {
    w = Math.ceil(w / 2); h = Math.ceil(h / 2);
    cur = draw(cur, w, h);
  }
  for (let shrink = 0; shrink < 6; shrink += 1) {
    const c = draw(cur, tw, th);
    // the size in pixels is kept as long as the quality can give way: a lens profile is for one pixel size
    for (const q of [0.9, 0.84, 0.78, 0.7, 0.62, 0.54, 0.46, 0.38]) {
      const blob = await new Promise((ok, bad) => c.toBlob((b) => (b ? ok(b) : bad(new Error('The photo could not be prepared.'))), 'image/jpeg', q));
      if (blob.size <= maxBytes) { if (src.close) src.close(); return { blob, width: tw, height: th }; }
    }
    tw = Math.max(1, Math.round(tw * 0.85)); th = Math.max(1, Math.round(th * 0.85));
  }
  throw new Error('The photo could not be made small enough for this server.');
}
function draw(src, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = 'high';
  g.drawImage(src, 0, 0, w, h);
  return c;
}
function viaImage(file) {
  return new Promise((ok, bad) => {
    const u = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(u); ok(img); };
    img.onerror = () => { URL.revokeObjectURL(u); bad(new Error('This file is not a picture the browser can read.')); };
    img.src = u;
  });
}

// For tests and the diagnostics sheet: what this browser holds.
export async function held() {
  const out = [];
  jobs.forEach((j) => out.push({ id: j.id, state_bytes: j.state ? j.state.length : 0, pictures: [...j.pictures.entries()].map(([name, b]) => ({ name, bytes: b.size })) }));
  return out;
}

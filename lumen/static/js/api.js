// API client.  Every URL is resolved against the document base, so the app
// works at "/", under "/content/<guid>/" on Posit Connect and behind any
// path-stripping reverse proxy.  Nothing here ever builds an absolute path.

import * as carry from './carry.js';

const BASE = new URL('.', document.baseURI);

export function url(path) {
  return new URL(String(path).replace(/^\/+/, ''), BASE).toString();
}

// Who keeps a job while it is open is the server's to say (config.sessions).  Where it leaves the job to the
// browser, every request that belongs to a job goes through the carrier (carry.js) together with that job; the
// functions below are called as they always were.
export function useCarrier(sessions) {
  carry.configure(sessions, { url });
}
const carried = (path) => carry.active() && carry.isJobPath(path);
export const jobIsCarried = () => carry.active();

export class ApiError extends Error {
  constructor(status, body) {
    const e = (body && body.error) || {};
    super(e.message || `Request failed (${status})`);
    this.status = status;
    this.lumen = { code: e.code || `http_${status}`, message: e.message || `Request failed (${status})`, details: e.details || {}, suggestions: e.suggestions || [] };
  }
}

async function parse(res) {
  const ct = res.headers.get('content-type') || '';
  let body = null;
  if (ct.includes('application/json')) {
    try { body = await res.json(); } catch (_) { body = null; }
  }
  if (!res.ok) throw new ApiError(res.status, body);
  return body;
}

export async function api(method, path, body, { timeout = 60000, signal } = {}) {
  if (carried(path)) {
    const r = await carry.request(method, path, { json: body, timeout, signal });
    if (!r.ok) throw new ApiError(r.status, r.body);
    return r.json;
  }
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(new DOMException('timeout', 'TimeoutError')), timeout);
  if (signal) signal.addEventListener('abort', () => ctl.abort(signal.reason), { once: true });
  try {
    const res = await fetch(url(path), {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      credentials: 'same-origin',
      cache: 'no-store',
      signal: ctl.signal,
    });
    return await parse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      if (signal && signal.aborted) throw err;
      throw new ApiError(0, { error: { code: 'timeout', message: 'The server did not answer in time.', suggestions: ['Check the connection and retry.'] } });
    }
    throw new ApiError(0, { error: { code: 'network', message: 'Cannot reach the server.', suggestions: ['Check the connection and retry.'] } });
  } finally {
    clearTimeout(timer);
  }
}

// Upload a Blob as the raw request body with progress (XHR: fetch has no
// upload progress in Safari).  `meta` travels as URL-encoded JSON.
export function upload(path, blob, { meta, onProgress, signal, timeout = 120000 } = {}) {
  if (carried(path)) return carriedUpload(path, blob, { meta, onProgress, signal, timeout });
  if (carry.active() && /^\/?api\/scan(\?|$)/.test(String(path)) && blob.size > 1_500_000) {
    // a host that leaves the job to the browser also takes no large request: a photo with a code on it is reduced
    return carry.fitPhoto(blob, { maxEdge: 2400, maxBytes: 1_500_000 }).then((fit) => plainUpload(path, fit.blob, { meta, onProgress, signal, timeout }), () => plainUpload(path, blob, { meta, onProgress, signal, timeout }));
  }
  return plainUpload(path, blob, { meta, onProgress, signal, timeout });
}

function plainUpload(path, blob, { meta, onProgress, signal, timeout = 120000 } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url(path));
    xhr.responseType = 'json';
    xhr.timeout = timeout;
    xhr.setRequestHeader('Content-Type', blob.type || 'application/octet-stream');
    if (meta) xhr.setRequestHeader('X-Lumen-Meta', encodeURIComponent(JSON.stringify(meta)));
    if (onProgress) xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) onProgress(ev.loaded / ev.total); };
    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new ApiError(xhr.status, body));
    };
    xhr.onerror = () => reject(new ApiError(0, { error: { code: 'network', message: 'Upload failed: cannot reach the server.', suggestions: ['Check the connection and retry.'] } }));
    xhr.ontimeout = () => reject(new ApiError(0, { error: { code: 'timeout', message: 'Upload timed out.', suggestions: ['Retry on a better connection or use a smaller image.'] } }));
    xhr.onabort = () => reject(new DOMException('aborted', 'AbortError'));
    if (signal) signal.addEventListener('abort', () => xhr.abort(), { once: true });
    xhr.send(blob);
  });
}

// A picture that is kept with the job (a photo of the capture, a board photo, the photo of a record) is made to
// fit before it is sent where the browser carries the job: all photos of a capture travel in one request when the
// lines are searched.  A live frame is not kept and goes as it is.
async function carriedUpload(path, blob, o) {
  let body = blob;
  const p = String(path);
  const lim = carry.limits();
  if (/\/(frames|photos|validate)(\?|$)/.test(p)) {
    const single = /[?&]role=verification/.test(p) || /\/validate(\?|$)/.test(p);
    try {
      body = (await carry.fitPhoto(blob, { maxEdge: lim.photo_max_edge, maxBytes: single ? lim.single_max_bytes : lim.photo_max_bytes })).blob;
    } catch (err) {
      throw new ApiError(0, { error: { code: 'photo_not_usable', message: (err && err.message) || 'The photo could not be prepared.', suggestions: ['Take the photo again.'] } });
    }
  }
  const r = await carry.request('POST', path, { body, contentType: body.type || 'application/octet-stream', meta: o.meta, onProgress: o.onProgress, signal: o.signal, timeout: o.timeout });
  if (!r.ok) throw new ApiError(r.status, r.body);
  return r.json;
}

// Where a picture of the job is shown from: an address on the server, or, where the browser carries the job, the
// picture it holds itself.  Returns a string or a promise of one (views/common.js thumb() takes both).
export function pictureSrc(path) {
  if (!carried(path)) return url(path);
  return carry.localPicture(path).then(async (hit) => {
    if (hit) return hit.url;
    const { blob } = await fetchBlob(path);     // a picture the server has to prepare (rare): through the carrier
    return URL.createObjectURL(blob);
  });
}

// Fetch a binary resource; returns {blob, headers}.
export async function fetchBlob(path, { timeout = 60000 } = {}) {
  if (carried(path)) {
    const hit = await carry.localPicture(path);
    if (hit) return { blob: hit.blob, headers: { get: () => null } };
    const r = await carry.request('GET', path, { timeout, expect: 'blob' });
    if (!r.ok) throw new ApiError(r.status, r.body);
    return { blob: r.blob, headers: r.headers };
  }
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  try {
    const res = await fetch(url(path), { credentials: 'same-origin', signal: ctl.signal });
    if (!res.ok) {
      let body = null;
      try { body = await res.json(); } catch (_) { /* not json */ }
      throw new ApiError(res.status, body);
    }
    return { blob: await res.blob(), headers: res.headers };
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(0, { error: { code: 'network', message: 'Cannot load the image.', suggestions: ['Check the connection and retry.'] } });
  } finally {
    clearTimeout(timer);
  }
}

export async function loadImage(blob) {
  if ('createImageBitmap' in window) {
    try { return await createImageBitmap(blob); } catch (_) { /* fall through */ }
  }
  const src = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.decoding = 'async';
    await new Promise((ok, bad) => { img.onload = ok; img.onerror = () => bad(new Error('image decode failed')); img.src = src; });
    return img;
  } finally {
    setTimeout(() => URL.revokeObjectURL(src), 30000);
  }
}

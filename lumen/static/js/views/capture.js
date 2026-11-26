// Step 3 — capture photos (camera, uploads, video) and review them.
import { api, upload, pictureSrc } from '../api.js';
import { h, clear, icon, chip, banner, toast, showError, busy, sheet } from '../dom.js';
import { state, setSession, refreshSession, registrationFrames, acceptedTags } from '../store.js';
import { Viewer } from '../viewer.js';
import { frameQualityChips, coachLine, coachList, thumb } from './common.js';
import { fetchBlob, loadImage } from '../api.js';
import { openCamera } from './camera_sheet.js';

const REJECT_TEXT = {
  clipped: 'cut off at the image border', too_small: 'too small in the image', too_oblique: 'seen at too steep an angle',
  duplicate_id: 'same ID appears twice', decode_mismatch: 'pattern could not be confirmed', bent_or_distorted: 'edge is not straight (bent tag or lens distortion)',
  glare: 'overexposed / glare', edge_disturbed: 'edge partly hidden or disturbed', edge_low_contrast: 'low contrast', edge_fit_diverged: 'edge not found',
  edge_outside_image: 'cut off at the image border', edge_too_few_pixels: 'too small in the image', degenerate_quad: 'shape not usable', refinement_inconsistent: 'edges not consistent',
};
export const rejectText = (r) => REJECT_TEXT[r] || r.replace(/_/g, ' ');

// Normalise whatever the picker returns into something the server can decode.
async function prepareFile(file) {
  const okType = /^image\/(jpeg|png|webp|bmp)$/.test(file.type);
  if (okType) return { blob: file, transcoded: false };
  // HEIC or unknown: let the browser decode it (Safari can) and re-encode as JPEG.
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  c.getContext('2d').drawImage(bmp, 0, 0);
  if (bmp.close) bmp.close();
  const blob = await new Promise((ok, bad) => c.toBlob((b) => (b ? ok(b) : bad(new Error('encode failed'))), 'image/jpeg', 0.95));
  return { blob, transcoded: true };
}

export function mount(root, ctx) {
  const s = () => state.session;
  const grid = h('div', { class: 'frames', role: 'list', 'aria-label': 'Captured photos' });
  const progress = h('div', { class: 'stack', 'aria-live': 'polite' });
  const checklist = h('ul', { class: 'checklist' });
  const summary = h('div');
  const coachBox = h('div', { class: 'stack', 'data-region': 'coach' });
  const pairBox = h('div', { class: 'stack' });
  let uploading = 0;
  let destroyed = false;
  let adviceSeq = 0;

  const pickMany = h('input', { type: 'file', accept: 'image/*', multiple: true, hidden: true, 'data-input': 'photos', onchange: (ev) => { handleFiles([...ev.target.files], 'photo_upload'); ev.target.value = ''; } });
  const pickCamera = h('input', { type: 'file', accept: 'image/*', capture: 'environment', hidden: true, 'data-input': 'native-camera', onchange: (ev) => { handleFiles([...ev.target.files], 'camera_still'); ev.target.value = ''; } });
  const pickVideo = h('input', { type: 'file', accept: 'video/*', hidden: true, 'data-input': 'video', onchange: (ev) => { handleVideo(ev.target.files[0]); ev.target.value = ''; } });

  // who keeps the job while it is open (config.sessions): where it is the phone, photos are reduced and there is no video
  const sessionsCfg = (state.config && state.config.sessions) || {};
  const carriedJob = sessionsCfg.kept_by === 'client' ? sessionsCfg : null;
  const videoOk = sessionsCfg.video !== false;
  const drop = h('div', { class: 'dropzone',
    ondragover: (ev) => { ev.preventDefault(); drop.classList.add('over'); },
    ondragleave: () => drop.classList.remove('over'),
    ondrop: (ev) => { ev.preventDefault(); drop.classList.remove('over'); handleFiles([...ev.dataTransfer.files].filter((f) => f.type.startsWith('image/') || /\.(heic|heif)$/i.test(f.name)), 'photo_upload'); },
  },
    h('button', { class: 'btn primary lg block', type: 'button', 'data-action': 'live-camera', onclick: () => openCamera({ onChange: render, onFallback: () => pickCamera.click() }) }, icon('camera'), 'Open camera'),
    h('div', { class: `tiles3${videoOk ? '' : ' two'}` },
      h('button', { class: 'btn tile', type: 'button', 'data-action': 'native-photo', onclick: () => pickCamera.click() }, icon('camera'), h('span', null, carriedJob ? 'Phone camera' : 'Full-res photo')),
      h('button', { class: 'btn tile', type: 'button', 'data-action': 'choose-photos', onclick: () => pickMany.click() }, icon('upload'), h('span', null, 'Choose photos')),
      // a video is worked on after its upload has ended: only a server that keeps the job can do that
      videoOk ? h('button', { class: 'btn tile', type: 'button', 'data-action': 'choose-video', onclick: () => pickVideo.click() }, icon('video'), h('span', null, 'Use a video')) : null),
    carriedJob
      ? h('div', { class: 'hint', 'data-region': 'photos-reduced' }, `On this server the phone carries the job, so every photo is reduced to ${carriedJob.photo_max_edge} px before it is sent. That is enough to place a placard; full-size photos, on a server that keeps the job, measure more finely.`)
      : h('div', { class: 'hint' }, 'Full-resolution photos from the phone camera give the best accuracy. The in-app camera is quicker and coaches the sweep.'),
    pickMany, pickCamera, pickVideo);

  root.append(
    h('h1', { class: 'view-title' }, 'Photos of the panel'),
    h('p', { class: 'view-lead' }, 'Take 3 to 9 photos from different positions. Each must show the tags; together they must show the floor line and the side line the job refers to.'),
    summary, coachBox, drop, progress,
    h('div', { class: 'card' }, h('h3', null, 'Before you analyse'), checklist),
    pairBox,
    h('div', { class: 'eyebrow' }, 'Photos'), grid);

  // What to do next, from the server's look at the photos taken so far.
  async function loadAdvice() {
    const seq = ++adviceSeq;
    try {
      const adv = await api('GET', `api/sessions/${s().id}/capture/advice`);
      if (destroyed || seq !== adviceSeq) return;
      clear(coachBox);
      const line = coachLine(adv);
      if (line) coachBox.append(line);
      const rest = { ...adv, items: adv.items.filter((i) => !adv.next || i.code !== adv.next.code) };
      const list = coachList(rest);
      if (list) coachBox.append(h('details', { class: 'fold' }, h('summary', null, `${rest.items.length} more hint${rest.items.length === 1 ? '' : 's'}`), h('div', { class: 'body' }, list)));
      if (adv.spread_deg !== null) coachBox.append(h('p', { class: 'hint' }, `Viewpoints differ by up to ${adv.spread_deg.toFixed(0)}° (${adv.spread_source === 'registered' ? 'from the analysis' : 'rough estimate from the tags'}).`));
      clear(pairBox);
      if (adv.pairs && adv.pairs.length) {
        pairBox.append(h('div', { class: 'card tight' }, h('h3', null, 'Flash pairs'),
          adv.pairs.map((p) => h('div', { class: 'row wrap', 'data-pair': p.pair_id },
            chip(p.usable ? 'aligned' : p.status.replace(/_/g, ' '), p.usable ? 'ok' : 'warn'),
            h('span', { class: 'hint' }, p.usable
              ? `camera moved ${p.max_shift_px.toFixed(1)} px between light off and on · brightness ×${p.brightness_on_over_off.toFixed(2)}${p.exposure && p.exposure.source === 'unknown' ? ' (exposure not known to the browser)' : ''}`
              : (p.reasons || []).join(' '))))));
      }
    } catch (_) { /* coaching is optional; the checklist below still works */ }
  }

  async function handleFiles(files, source) {
    if (!files.length) return;
    const lim = state.config.limits;
    for (const file of files) {
      if (destroyed) return;
      if (s().frames.length + uploading >= lim.max_frames) {
        toast({ title: 'Frame limit reached', detail: `This session holds at most ${lim.max_frames} frames.`, kind: 'error' });
        break;
      }
      const bar = h('progress', { max: '1', value: '0' });
      const row = h('div', { class: 'card tight' }, h('div', { class: 'row' }, h('span', { class: 'grow' }, file.name || 'photo'), h('span', { class: 'hint' }, 'uploading…')), bar);
      progress.append(row);
      uploading += 1;
      try {
        const { blob, transcoded } = await prepareFile(file);
        if (blob.size > lim.max_image_mb * 1e6) throw Object.assign(new Error(`Photo is larger than ${lim.max_image_mb} MB.`), {});
        const meta = { captured_at: new Date(file.lastModified || Date.now()).toISOString(), transcoded_by_client: transcoded };
        const out = await upload(`api/sessions/${s().id}/frames?role=registration&source=${source}`, blob, { meta, onProgress: (p) => { bar.value = p; } });
        row.querySelector('.hint').textContent = 'measuring tags…';
        const f = out.frame;
        const acc = f.markers.filter((m) => m.accepted).length;
        if (!acc) toast({ title: 'No usable tag in this photo', detail: f.markers.length ? f.markers.map((m) => `ID ${m.tag_id}: ${m.reject_reasons.map(rejectText).join(', ')}`).join(' · ') : 'Make sure the tags are in view, in focus and not washed out.', kind: 'error', timeout: 8000 });
      } catch (err) {
        showError(err, 'Upload failed');
      } finally {
        uploading -= 1;
        row.remove();
      }
      await refreshSession();
      render();
    }
  }

  async function handleVideo(file) {
    if (!file) return;
    const lim = state.config.limits;
    if (file.size > lim.max_video_mb * 1e6) { toast({ title: 'Video too large', detail: `Limit is ${lim.max_video_mb} MB.`, kind: 'error' }); return; }
    const b = busy('Uploading video…');
    try {
      const out = await upload(`api/sessions/${s().id}/video`, file, { onProgress: (p) => b.update(`Uploading video… ${Math.round(p * 100)} %`), timeout: 600000 });
      b.update('Selecting frames…');
      let job = out.job;
      while (job.state === 'queued' || job.state === 'running') {
        await new Promise((r) => setTimeout(r, 900));
        job = await api('GET', `api/sessions/${s().id}/jobs/${job.id}`);
        b.update(`${job.stage || 'Working'}… ${Math.round((job.progress || 0) * 100)} %`);
      }
      if (job.state !== 'done') throw Object.assign(new Error(job.error ? job.error.message : 'Video processing failed'), { lumen: job.error });
      await refreshSession();
      const r = job.result;
      toast({ title: `${r.selected.length} frame${r.selected.length === 1 ? '' : 's'} chosen from the video`, detail: [`${r.candidates} candidates, sharpest frame of each ${r.window_s.toFixed(2)} s.`, ...(r.hints || []), 'Review them below and remove any you do not want.'].join(' '), kind: r.selected.length ? 'ok' : 'error', timeout: 10000 });
    } catch (err) { showError(err, 'Video could not be used'); } finally { b.done(); render(); }
  }

  async function removeFrame(fid) {
    try { setSession(await api('DELETE', `api/sessions/${s().id}/frames/${fid}`)); render(); } catch (err) { showError(err); }
  }

  async function inspect(f) {
    const wrap = h('div', { class: 'viewer-wrap', style: { height: '56dvh' } });
    const info = h('div', { class: 'stack' },
      f.markers.length ? f.markers.map((m) => h('div', { class: 'row wrap' },
        h('span', { class: 'mono', style: { fontWeight: 800 } }, `ID ${m.tag_id}`),
        m.accepted ? chip('accepted', 'ok') : chip('rejected', 'danger'),
        chip(`${Math.round(Math.min(...m.edge_px))} px`, 'solid'),
        chip(`${m.size_mm} mm ${m.size_provenance}`, m.size_provenance === 'measured' ? 'ok' : 'warn'),
        m.accepted ? chip(`σ ${m.corner_sigma_px.toFixed(3)} px`, 'solid') : null,
        m.reject_reasons.length ? h('span', { class: 'hint' }, m.reject_reasons.map(rejectText).join(', ')) : null))
        : banner('warn', 'No AprilTag found in this photo.'),
      f.notes.length ? h('p', { class: 'hint' }, f.notes.join(' · ')) : null,
      h('p', { class: 'hint' }, `${f.width}×${f.height} px analysed · native ${f.native_width}×${f.native_height}${f.calibration_note ? ` · ${f.calibration_note}` : ''}${f.video_time_s !== null && f.video_time_s !== undefined ? ` · video frame at ${f.video_time_s.toFixed(2)} s` : ''}${f.camera_label ? ` · ${f.camera_label}` : ''}`),
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Phone light / flash in this photo'),
        h('div', { class: 'seg', role: 'radiogroup', 'aria-label': 'Light state' }, ['off', 'on', 'unknown'].map((t) =>
          h('button', { type: 'button', role: 'radio', 'aria-checked': f.torch === t ? 'true' : 'false', 'aria-pressed': f.torch === t ? 'true' : 'false', 'data-torch': t,
            onclick: async (ev) => {
              try {
                setSession(await api('PATCH', `api/sessions/${s().id}/frames/${f.id}`, { torch: t }));
                ev.currentTarget.parentNode.querySelectorAll('button').forEach((b) => { const on = b === ev.currentTarget; b.setAttribute('aria-pressed', String(on)); b.setAttribute('aria-checked', String(on)); });
                render();
              } catch (err) { showError(err); }
            } }, t === 'unknown' ? 'Not known' : `Light ${t}`)))),
      f.pair_id ? h('p', { class: 'hint' }, `Part of flash pair ${f.pair_id}.`) : null);
    const ref = sheet({ title: `Photo ${f.index}`, content: h('div', { class: 'stack' }, wrap, info), wide: true });
    const viewer = new Viewer(wrap, { label: `Photo ${f.index} with detected tags` });
    try {
      const { blob } = await fetchBlob(`api/sessions/${s().id}/frames/${f.id}/image?max=2400`);
      viewer.setFrame(await loadImage(blob), f.width, f.height);
      viewer.setOverlay({ primitives: f.markers.map((m) => ({ kind: m.accepted ? 'tag' : 'tag_rejected', points: m.corners_px, closed: true, label: `ID ${m.tag_id}` })) });
    } catch (err) { showError(err); }
    const obs = new MutationObserver(() => { if (!document.body.contains(ref.box)) { viewer.destroy(); obs.disconnect(); } });
    obs.observe(document.body, { childList: true });
  }

  function render() {
    if (destroyed) return;
    const sess = s();
    const frames = registrationFrames(sess);
    clear(grid);
    if (!frames.length) grid.append(h('p', { class: 'hint' }, 'No photos yet.'));
    frames.forEach((f) => {
      const acc = acceptedTags(f).length;
      const cls = acc === 0 ? ' bad' : (f.markers.some((m) => !m.accepted) || f.quality.sharpness < 40) ? ' issue' : '';
      grid.append(h('div', { class: `frame-tile${cls}${f.selected ? '' : ' excluded'}`, role: 'listitem', 'data-frame': f.id },
        h('button', { type: 'button', style: { all: 'unset', cursor: 'pointer', display: 'block', width: '100%' }, 'aria-label': `Inspect photo ${f.index}`, onclick: () => inspect(f) },
          thumb(pictureSrc(`api/sessions/${sess.id}/frames/${f.id}/image?max=360`), `Photo ${f.index}`)),
        h('span', { class: 'idx' }, `#${f.index}`),
        h('button', { class: 'x', type: 'button', 'aria-label': `Remove photo ${f.index}`, onclick: () => removeFrame(f.id) }, icon('trash')),
        h('div', { class: 'meta' }, h('div', { class: 'badges' }, frameQualityChips(f),
          f.torch === 'on' ? chip('light on', 'info') : null,
          f.pair_id ? chip('pair', 'solid') : null,
          f.video_time_s !== null && f.video_time_s !== undefined ? chip(`${f.video_time_s.toFixed(1)} s`, 'solid') : null))));
    });
    loadAdvice();

    const withTags = frames.filter((f) => acceptedTags(f).length > 0);
    const ids = new Set();
    withTags.forEach((f) => acceptedTags(f).forEach((m) => ids.add(m.tag_id)));
    const unmeasured = [...ids].filter((id) => !(String(id) in (sess.tag_sizes.measured_mm || {})));
    const item = (ok, text) => h('li', { class: ok ? 'done' : 'todo' }, icon(ok ? 'check' : 'alert'), h('span', null, text));
    clear(checklist).append(
      item(withTags.length >= 1, `${withTags.length} photo${withTags.length === 1 ? '' : 's'} with accepted tags (1 is the minimum, 3–9 recommended)`),
      item(ids.size >= 2, `${ids.size} different tag${ids.size === 1 ? '' : 's'} seen (2 or more allow an independent check)`),
      item(unmeasured.length === 0 && ids.size > 0, unmeasured.length ? `Tags without a measured size: ${unmeasured.map((i) => `ID ${i}`).join(', ')}` : 'All seen tags have measured sizes'),
      h('li', null, icon('eye'), h('span', null, 'Check yourself: the floor seam and the side edge are visible in at least one photo, on the same flat face as the tags.')));

    clear(summary);
    if (sess.registration) {
      summary.append(banner('ok', 'Analysis is up to date.', `${sess.registration.n_tags} tags, ${sess.registration.frames_used.length} photo(s) registered. Adding or removing photos requires a new analysis.`));
    }
    ctx.setActions([
      { label: 'Back', back: true, onClick: () => ctx.go('markers') },
      sess.registration
        ? { id: 'continue', label: 'Continue', kind: 'primary lg', icon: 'arrow', onClick: () => ctx.go('datums') }
        : { id: 'analyze', label: withTags.length ? `Analyse ${withTags.length} photo${withTags.length === 1 ? '' : 's'}` : 'Analyse', kind: 'primary lg', icon: 'target', disabled: withTags.length === 0, onClick: analyze },
    ]);
  }

  async function analyze() {
    const b = busy('Registering tags and views…');
    try {
      const out = await api('POST', `api/sessions/${s().id}/register`, undefined, { timeout: 180000 });
      setSession(out.session);
      const reg = out.registration;
      if (!reg.consistent) toast({ title: 'Tags contradict each other', detail: 'See the Lines step for details.', kind: 'error', timeout: 8000 });
      ctx.go('datums');
    } catch (err) { showError(err, 'Analysis failed'); render(); } finally { b.done(); }
  }

  render();
  return { destroy() { destroyed = true; } };
}

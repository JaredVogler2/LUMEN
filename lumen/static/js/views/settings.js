// Settings: display units and lens profiles (camera calibration from a printed board).
//
// A lens profile describes one camera configuration of one phone, so it is
// kept in this browser (it never needs an account) and attached to a session
// when the operator chooses.  The server keeps the calibration photos and
// points only as long as a session.
import { api, upload, url } from '../api.js';
import { thumb } from './common.js';
import { h, clear, append, icon, sheet, banner, chip, toast, showError, busy, fmt } from '../dom.js';
import { state, setState, setSession, prefs } from '../store.js';
import { openCamera } from './camera_sheet.js';

const REASON = {
  board_not_found: 'no board found in the photo', checkerboard_partly_visible: 'the checkerboard is partly outside the photo', too_few_corners: 'too few corners visible',
  corners_collinear: 'the visible corners are on one line', board_too_small_in_picture: 'the board is too small in the photo', too_blurred: 'blurred',
  glare_or_uneven_lighting: 'glare or uneven light', marker_size_mismatch: 'the marker size does not match the description',
  too_few_views: 'too few usable photos', fronto_parallel_only: 'all photos are square-on: the focal length cannot be determined', same_tilt: 'all photos have the same tilt',
  inconsistent_views: 'the photos contradict each other', board_not_square: 'the print is stretched in one direction', focal_length_not_determined: 'the focal length is not determined',
  principal_point_not_determined: 'the image centre is not determined', uncertainty_not_determined: 'the uncertainty could not be determined', high_reprojection_error: 'the fit is poor',
  insufficient_coverage: 'the board did not cover enough of the picture', solver_failed: 'the solver failed', reprojection_outlier: 'does not fit the other photos', duplicate_photo: 'same photo twice',
};
const why = (r) => REASON[r] || String(r).replace(/_/g, ' ');
// Notes of a finished calibration that come without their own advice sentence.
const NOTE = {
  views_excluded: 'Photos that did not fit the others were left out.',
  jackknife_unavailable: 'Too few photos to check how much the result depends on any one of them; the stated uncertainty is the formal one.',
  rotation_ambiguous_180: 'This checkerboard looks the same after half a turn, so the image centre is checked less strictly. A board with an odd and an even number of squares avoids that.',
  rotation_ambiguous_90: 'This checkerboard looks the same after a quarter turn. A board with different numbers of squares across and down avoids that.',
};

function saveProfile(p) {
  const all = prefs.calibrations.filter((x) => x.id !== p.id);
  all.push(p);
  prefs.calibrations = all;
}
function removeProfile(id) { prefs.calibrations = prefs.calibrations.filter((x) => x.id !== id); }

export function openSettings() {
  const content = h('div', { class: 'stack', 'data-region': 'settings' });
  const ref = sheet({ title: 'Settings', content, wide: true });
  let ws = null;            // calibration workspace being built
  let boardForm = { kind: 'charuco', squares_x: 9, squares_y: 6, square_mm: '', marker_mm: '', dictionary: 'DICT_5X5_100' };

  const unit = (u, label) => h('button', { type: 'button', 'aria-pressed': state.units === u ? 'true' : 'false',
    onclick: (ev) => { prefs.units = u; setState({ units: u }); ev.currentTarget.parentNode.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === ev.currentTarget))); } }, label);

  function profileCard(p) {
    const sess = state.session;
    const attached = sess && sess.calibration && sess.calibration.id === p.id;
    return h('div', { class: `card tight${attached ? ' selected' : ''}`, 'data-profile': p.id },
      h('div', { class: 'row wrap' }, h('strong', { class: 'grow' }, p.label || p.id), attached ? chip('used in this session', 'ok') : null, chip(p.method, 'solid')),
      h('p', { class: 'hint' }, `${p.width}×${p.height} px · ${p.device || 'camera not named'}${p.zoom !== null && p.zoom !== undefined ? ` · zoom ${p.zoom}` : ''} · ${p.n_views || '?'} photos · RMS ${p.rms_px === null || p.rms_px === undefined ? '—' : fmt(p.rms_px, 3)} px · coverage ${p.coverage === null || p.coverage === undefined ? '—' : Math.round(p.coverage * 100)} % · ${p.created ? p.created.slice(0, 10) : ''}`),
      h('div', { class: 'row wrap' },
        sess ? h('button', { class: 'btn sm grow', type: 'button', 'data-action': attached ? 'profile-detach' : 'profile-use', onclick: async () => {
          const b = busy(attached ? 'Removing the lens profile…' : 'Applying the lens profile and re-measuring the tags…');
          try { setSession(await api('PUT', `api/sessions/${sess.id}/calibration`, { profile: attached ? null : p }, { timeout: 180000 })); toast({ title: attached ? 'Lens profile removed from this session' : 'Lens profile attached', detail: attached ? '' : 'It applies to photos of the same camera, size and orientation. Analyse again.', kind: 'ok' }); render(); } catch (err) { showError(err); } finally { b.done(); }
        } }, attached ? 'Stop using it in this session' : 'Use in this session') : null,
        h('button', { class: 'btn sm', type: 'button', 'data-action': 'profile-delete', 'aria-label': `Delete profile ${p.label || p.id}`, onclick: () => { removeProfile(p.id); render(); } }, icon('trash'))));
  }

  // Where the board has been in the picture so far (each photo's corner extent, as fractions of the picture).
  function coverageMap(views) {
    const ok = views.filter((v) => v.ok && v.bbox);
    if (!ok.length) return null;
    const [W, H] = ok[0].image_size;
    const cw = 168, ch = Math.round((cw * H) / W);
    const c = h('canvas', { width: String(cw * 2), height: String(ch * 2), role: 'img', 'data-region': 'cal-coverage', 'aria-label': `Parts of the picture covered by the board in ${ok.length} photo(s)`,
      style: { width: `${cw}px`, height: `${ch}px`, flex: 'none', border: '1px solid var(--border-strong)', borderRadius: '6px', background: 'var(--surface-3)' } });
    const ctx = c.getContext('2d');
    ctx.scale(2, 2);
    ctx.fillStyle = 'rgba(31, 157, 104, 0.22)'; ctx.strokeStyle = 'rgba(31, 157, 104, 0.9)'; ctx.lineWidth = 1;
    ok.forEach((v) => {
      const [x0, y0, x1, y1] = v.bbox;
      ctx.fillRect(x0 * cw, y0 * ch, (x1 - x0) * cw, (y1 - y0) * ch);
      ctx.strokeRect(x0 * cw + 0.5, y0 * ch + 0.5, (x1 - x0) * cw - 1, (y1 - y0) * ch - 1);
    });
    return h('div', { class: 'row', style: { alignItems: 'flex-start' } }, c,
      h('p', { class: 'hint grow', style: { margin: '0' } }, 'Green: where the board has been in the picture. Fill all of it, corners included.'));
  }

  async function startWorkspace() {
    const sq = parseFloat(boardForm.square_mm), mk = parseFloat(boardForm.marker_mm);
    if (!(sq > 0)) { toast({ title: 'Enter the measured square size', detail: 'Measure across several squares of the print and divide by their number.', kind: 'error' }); return; }
    if (boardForm.kind === 'charuco' && !(mk > 0 && mk < sq)) { toast({ title: 'Enter the measured marker size', detail: 'The black marker inside a white square, smaller than the square.', kind: 'error' }); return; }
    try {
      const board = { kind: boardForm.kind, squares_x: Number(boardForm.squares_x), squares_y: Number(boardForm.squares_y), square_mm: sq };
      if (boardForm.kind === 'charuco') { board.marker_mm = mk; board.dictionary = boardForm.dictionary; }
      ws = await api('POST', 'api/calibration', { board });
      render();
    } catch (err) { showError(err, 'The board description was refused'); }
  }

  async function addFiles(files) {
    for (const file of files) {
      const b = busy(`Reading ${file.name || 'photo'}…`);
      try {
        let blob = file;
        if (!/^image\/(jpeg|png|webp|bmp)$/.test(file.type)) {
          const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
          const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
          c.getContext('2d').drawImage(bmp, 0, 0);
          blob = await new Promise((ok, bad) => c.toBlob((x) => (x ? ok(x) : bad(new Error('encode failed'))), 'image/jpeg', 0.95));
        }
        const out = await upload(`api/calibration/${ws.id}/photos`, blob, { meta: { capture_mode: 'photo' } });
        ws = out.calibration;
      } catch (err) { showError(err, 'The photo could not be used'); } finally { b.done(); }
      render();
    }
  }

  async function solve() {
    const b = busy('Calibrating…');
    try { ws = await api('POST', `api/calibration/${ws.id}/solve`, { label: '' }, { timeout: 180000 }); } catch (err) { showError(err, 'Calibration failed'); } finally { b.done(); }
    render();
  }

  async function check(file) {
    if (!file) return;
    const b = busy('Checking the profile on this photo…');
    try { const out = await upload(`api/calibration/${ws.id}/validate`, file, {}); ws = out.calibration; } catch (err) { showError(err, 'The check could not be made'); } finally { b.done(); }
    render();
  }

  function boardStep() {
    const num = (key, label, attrs = {}) => h('label', { class: 'field' }, h('span', { class: 'label' }, label),
      h('input', { class: 'input num', type: 'number', inputmode: 'decimal', value: boardForm[key], 'data-field': key, ...attrs, oninput: (ev) => { boardForm[key] = ev.target.value; } }));
    const q = () => `kind=${boardForm.kind}&squares_x=${boardForm.squares_x}&squares_y=${boardForm.squares_y}&square_mm=30${boardForm.kind === 'charuco' ? '&marker_mm=21&dictionary=DICT_5X5_100' : ''}`;
    return h('div', { class: 'card', 'data-region': 'cal-board' },
      h('h3', null, 'New lens profile — 1. The board'),
      h('p', { class: 'hint' }, 'A web page is not told the lens parameters of the camera, so they are measured from photos of a printed board. Do this once for each phone, camera (lens), zoom, resolution and orientation you measure with.'),
      h('div', { class: 'seg', role: 'radiogroup', 'aria-label': 'Board type' },
        ['charuco', 'checkerboard'].map((k) => h('button', { type: 'button', role: 'radio', 'aria-checked': String(boardForm.kind === k), 'aria-pressed': String(boardForm.kind === k), 'data-action': `cal-kind-${k}`,
          onclick: () => { boardForm.kind = k; if (k === 'checkerboard') { boardForm.squares_x = 10; boardForm.squares_y = 7; } else { boardForm.squares_x = 9; boardForm.squares_y = 6; } render(); } }, k === 'charuco' ? 'ChArUco (recommended)' : 'Checkerboard'))),
      h('a', { class: 'btn sm', href: url(`api/calibration/board.png?${q()}`), download: '', 'data-action': 'cal-download' }, icon('download'), 'Download a printable board'),
      h('p', { class: 'hint' }, 'Print it at full size on stiff, flat material (or glue it to a flat plate). Printers do not keep the size: measure the print with a steel rule across as many squares as possible and divide by their number.'),
      h('div', { class: 'row wrap' }, num('squares_x', 'Squares across', { min: '3', step: '1' }), num('squares_y', 'Squares down', { min: '3', step: '1' })),
      h('div', { class: 'row wrap' }, num('square_mm', 'Measured square size, mm', { step: '0.01', min: '2', placeholder: 'e.g. 29.86' }),
        boardForm.kind === 'charuco' ? num('marker_mm', 'Measured marker size, mm', { step: '0.01', min: '1', placeholder: 'e.g. 20.90' }) : null),
      h('button', { class: 'btn primary block', type: 'button', 'data-action': 'cal-start', onclick: startWorkspace }, icon('arrow'), 'Continue to photos'));
  }

  function photoStep() {
    const usable = ws.views.filter((v) => v.ok).length;
    const pick = h('input', { type: 'file', accept: 'image/*', multiple: true, hidden: true, 'data-input': 'cal-photos', onchange: (ev) => { addFiles([...ev.target.files]); ev.target.value = ''; } });
    const pickCheck = h('input', { type: 'file', accept: 'image/*', hidden: true, 'data-input': 'cal-check', onchange: (ev) => { check(ev.target.files[0]); ev.target.value = ''; } });
    const res = ws.result, cal = res && res.calibration;
    const box = h('div', { class: 'card', 'data-region': 'cal-photos' },
      h('h3', null, 'New lens profile — 2. Photos of the board'),
      h('p', { class: 'hint' }, `Take 10–15 photos with the same camera, zoom and orientation you will measure with. Move so the board appears in every part of the picture, and tilt the phone 20–40° left, right, up and down. ${ws.board.kind === 'checkerboard' ? 'The whole checkerboard must be inside each photo.' : 'The board may be partly outside the picture.'}`),
      h('div', { class: 'row wrap' },
        h('button', { class: 'btn grow', type: 'button', 'data-action': 'cal-camera', onclick: () => openCamera({
          custom: { url: `api/calibration/${ws.id}/photos`, hint: 'Board photos for the lens profile.', count: () => ws.views.length,
            coach: () => `${ws.views.filter((v) => v.ok).length} usable photo(s). Tilt 20–40° and put the board in a new part of the picture each time.`,
            onResult: (out) => { ws = out.calibration; const v = out.view; return v.ok ? null : { title: 'That photo cannot be used', detail: `${v.reasons.map(why).join(', ')}. ${(v.suggestions || []).join(' ')}`, kind: 'error', timeout: 8000 }; } },
          onChange: render,
        }) }, icon('camera'), 'Open camera'),
        h('button', { class: 'btn grow', type: 'button', 'data-action': 'cal-choose', onclick: () => pick.click() }, icon('upload'), 'Choose photos')),
      pick, pickCheck,
      h('div', { class: 'row wrap' }, chip(`${usable} usable`, usable >= 10 ? 'ok' : usable >= 6 ? 'warn' : 'danger'), chip(`${ws.views.length - usable} refused`, ws.views.length - usable ? 'warn' : 'solid'), h('span', { class: 'hint' }, 'at least 6, better 10 or more')),
      coverageMap(ws.views),
      ws.views.length ? h('div', { class: 'frames' }, ws.views.map((v) => h('div', { class: `frame-tile${v.ok ? '' : ' bad'}`, 'data-view': v.view_id },
        thumb(url(`api/calibration/${ws.id}/photos/${v.view_id}/image?max=320`), `Board photo ${v.view_id}`),
        h('button', { class: 'x', type: 'button', 'aria-label': `Remove board photo ${v.view_id}`, onclick: async () => { try { ws = await api('DELETE', `api/calibration/${ws.id}/photos/${v.view_id}`); render(); } catch (err) { showError(err); } } }, icon('trash')),
        h('div', { class: 'meta' }, h('div', { class: 'badges' }, v.ok ? chip(`${v.n_points} corners`, 'ok') : chip(v.reasons.map(why).join(', ') || 'refused', 'danger'),
          v.ok && v.quality.tilt_deg !== null && v.quality.tilt_deg !== undefined ? chip(`tilt ${Math.round(v.quality.tilt_deg)}°`, 'solid') : null))))) : null,
      h('button', { class: 'btn primary block', type: 'button', 'data-action': 'cal-solve', disabled: usable >= 1 ? null : true, onclick: solve }, icon('target'), 'Calibrate'));
    if (res) {
      if (res.ok && ws.profile) {
        const p = ws.profile;
        box.append(
          banner('ok', 'Lens profile computed', `${p.n_views} photos · RMS ${fmt(p.rms_px, 3)} px · coverage ${Math.round((p.coverage || 0) * 100)} % · focal length ${fmt(p.fx, 1)} px ± ${p.param_sigma && p.param_sigma[0] !== null ? fmt(p.param_sigma[0], 2) : '?'} · distortion terms: ${(cal.distortion_model.estimated || []).join(', ')}.`,
            (res.warnings || []).filter((w) => NOTE[w]).map((w) => NOTE[w]).concat(res.suggestions || [])),
          h('div', { class: 'row wrap' },
            h('button', { class: 'btn primary grow', type: 'button', 'data-action': 'cal-save', onclick: () => { saveProfile(p); toast({ title: 'Lens profile saved on this phone', detail: 'Attach it to a session from this list.', kind: 'ok' }); ws = null; render(); } }, icon('check'), 'Save this profile'),
            h('button', { class: 'btn grow', type: 'button', 'data-action': 'cal-check', onclick: () => pickCheck.click() }, icon('eye'), 'Check with one more photo')));
        (ws.validations || []).slice(-1).forEach((v) => box.append(banner(v.ok ? 'ok' : 'warn', v.ok ? 'Check passed on a photo that was not used' : 'Check: larger residuals than expected',
          `Board corners after lens correction fit a plane to ${fmt(v.with_model.rms_px, 3)} px (${fmt(v.with_model.rms_mm, 3)} mm) RMS; without the lens model ${fmt(v.without_model.rms_px, 3)} px (${fmt(v.without_model.rms_mm, 3)} mm).`)));
      } else {
        box.append(banner('danger', 'No profile: the calibration was refused', (res.reasons || []).map(why).join('; '), res.suggestions || []));
      }
    }
    box.append(h('button', { class: 'btn sm', type: 'button', 'data-action': 'cal-cancel', onclick: async () => { try { await api('DELETE', `api/calibration/${ws.id}`); } catch (_) { /* expired */ } ws = null; render(); } }, 'Discard this calibration'));
    return box;
  }

  function render() {
    clear(content);
    const sess = state.session;
    const profiles = prefs.calibrations;
    // (the DOM's own append() would print null and arrays as text; the helper skips and flattens them)
    append(content, [
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Display units (calculations are always in millimetres)'),
        h('div', { class: 'seg' }, unit('mm', 'Millimetres'), unit('in', 'Inches'))),
      h('div', { class: 'eyebrow' }, 'Lens profiles on this phone'),
      sess && sess.calibration && !profiles.some((p) => p.id === sess.calibration.id)
        ? h('div', { class: 'card tight' }, h('div', { class: 'row wrap' }, h('strong', { class: 'grow' }, sess.calibration.label || sess.calibration.id), chip('used in this session', 'ok'), chip(sess.calibration.provenance, sess.calibration.provenance === 'simulated' ? 'info' : 'solid')),
          h('p', { class: 'hint' }, `${sess.calibration.width}×${sess.calibration.height} px · ${sess.calibration.method}`)) : null,
      profiles.length ? profiles.map(profileCard) : h('p', { class: 'hint' }, 'None saved. Without a lens profile LUMEN still measures on the panel plane, but lens distortion is only bounded, and results stay at "research preview".'),
      ws ? photoStep() : boardStep(),
      h('p', { class: 'hint' }, 'A profile is applied only to photos of exactly the same pixel size and orientation, and not when the photo reports a different camera, lens or zoom. Calibration photos are kept on the server only as long as a session.')]);
  }
  render();
  return ref;
}

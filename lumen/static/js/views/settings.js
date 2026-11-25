// Settings sheet.
import { h, sheet, banner } from '../dom.js';
import { state, setState, prefs } from '../store.js';

export function openSettings() {
  const unit = (u, label) => h('button', { type: 'button', 'aria-pressed': state.units === u ? 'true' : 'false',
    onclick: (ev) => { prefs.units = u; setState({ units: u }); ev.currentTarget.parentNode.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === ev.currentTarget))); } }, label);
  const content = h('div', { class: 'stack' },
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Display units (calculations are always in millimetres)'),
      h('div', { class: 'seg' }, unit('mm', 'Millimetres'), unit('in', 'Inches'))),
    banner('info', 'Camera calibration', 'Calibration profiles are managed here in a later build step.'));
  sheet({ title: 'Settings', content });
}

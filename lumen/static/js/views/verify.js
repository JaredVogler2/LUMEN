// Step 7 — post-installation verification (implemented in verify_impl.js).
import { h, banner } from '../dom.js';

export function mount(root, ctx) {
  root.append(
    h('h1', { class: 'view-title' }, 'Verify the placement'),
    banner('info', 'Verification is not part of this build yet.', 'The placement overlay and its checks are available on the previous step.'));
  ctx.setActions([{ label: 'Back', back: true, onClick: () => ctx.go('place') }]);
  return {};
}

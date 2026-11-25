// In-app camera.
import { toast } from '../dom.js';

export function openCamera(_opts) {
  toast({ title: 'The in-app camera is not part of this build yet', detail: 'Use “Full-resolution photo” or “Choose photos”.', kind: 'info' });
}

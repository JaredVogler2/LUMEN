// Barcode / QR scanning for operation identification.
import { toast } from '../dom.js';

export function openScanner(_onCode) {
  toast({ title: 'Scanning is not part of this build yet', detail: 'Type the ID into the search field instead.', kind: 'info' });
}

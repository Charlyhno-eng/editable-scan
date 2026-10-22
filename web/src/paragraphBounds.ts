import { Util } from 'pdfjs-dist';
import type { PDFPageProxy } from 'pdfjs-dist';

type Box = { x: number; y: number; width: number; height: number };
type Passage = Box & { id: string; text: string };
const normalize = (text: string) => text.normalize('NFC').replace(/\s/g, '');

export async function paragraphBounds<T extends Passage>(page: PDFPageProxy, passages: T[], edits: { id: string; text: string }[]): Promise<T[]> {
  if (!passages.some(passage => edits.some(edit => edit.id === passage.id))) return passages;
  const content = await page.getTextContent();
  const viewport = page.getViewport({ scale: 1 });
  const unit = Math.hypot(viewport.transform[0], viewport.transform[1]);
  const runs: (Box & { start: number; end: number })[] = [];
  let text = '';
  for (const item of content.items) {
    if (!('str' in item) || !normalize(item.str)) continue;
    const start = text.length;
    text += normalize(item.str);
    const matrix = Util.transform(viewport.transform, item.transform);
    const angle = Math.atan2(matrix[1], matrix[0]);
    const height = Math.hypot(matrix[2], matrix[3]);
    const style = content.styles[item.fontName];
    const points = [style?.ascent ?? 0.8, style?.descent ?? -0.2].flatMap(offset => [0, item.width * unit].map(width => ({
      x: matrix[4] + Math.cos(angle) * width + Math.sin(angle) * offset * height,
      y: matrix[5] + Math.sin(angle) * width - Math.cos(angle) * offset * height,
    })));
    const x = Math.min(...points.map(point => point.x));
    const y = Math.min(...points.map(point => point.y));
    runs.push({ start, end: text.length, x, y, width: Math.max(...points.map(point => point.x)) - x, height: Math.max(...points.map(point => point.y)) - y });
  }
  return passages.map(passage => {
    const edit = edits.find(item => item.id === passage.id);
    if (!edit) return passage;
    const replacement = normalize(edit.text);
    if (!replacement) return passage; // Keep a way to restore a deleted passage.
    const prefix = replacement.slice(0, 24);
    let best: { boxes: Box[]; length: number; distance: number } | undefined;
    let start = text.indexOf(prefix);
    while (start !== -1) {
      const first = runs.find(run => run.start <= start && run.end > start)!;
      const distance = Math.hypot(first.x - passage.x, first.y - passage.y);
      // Match at the source location, not a repeated sentence elsewhere.
      if (distance <= Math.max(48, Math.min(passage.width, passage.height) * 2)) {
        let length = prefix.length;
        while (length < replacement.length && text[start + length] === replacement[length]) length++;
        if (!best || length > best.length || (length === best.length && distance < best.distance)) {
          best = { length, distance, boxes: runs.filter(run => run.end > start && run.start < start + length) };
        }
      }
      start = text.indexOf(prefix, start + 1);
    }
    if (!best) return passage;
    const x = Math.min(...best.boxes.map(box => box.x));
    const y = Math.min(...best.boxes.map(box => box.y));
    return { ...passage, x, y, width: Math.max(...best.boxes.map(box => box.x + box.width)) - x, height: Math.max(...best.boxes.map(box => box.y + box.height)) - y };
  });
}

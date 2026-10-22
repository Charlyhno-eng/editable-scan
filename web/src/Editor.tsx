import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import type { PDFDocumentProxy, RenderTask } from 'pdfjs-dist';
import worker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { errorMessage } from './api';
import './editor.css';

GlobalWorkerOptions.workerSrc = worker;
type Overflow = 'wrap' | 'shrink' | 'page';
const fonts = [
  ['LiberationSans', 'Liberation Sans'], ['LiberationSerif', 'Liberation Serif'], ['LiberationMono', 'Liberation Mono'],
  ['DejaVuSans', 'DejaVu Sans'], ['DejaVuSerif', 'DejaVu Serif'], ['DejaVuSansMono', 'DejaVu Sans Mono'],
];
type Edit = { id: string; page: number; text: string; bold: boolean; color: string; overflow?: Overflow; font?: string };
type FontInfo = { name: string; size: number; exact: boolean; mixed: boolean; similar: string[] };
type Passage = Edit & { x: number; y: number; width: number; height: number; originalFont: string; fontExact: boolean; scanned: boolean; memberIds?: string[]; boxes?: { x: number; y: number; width: number; height: number }[]; lineCount?: number };
type History = { past: Edit[][]; present: Edit[]; future: Edit[][] };

function PagePreview({ pdf, page, zoom, passages, selected, edited, onSelect, onError }: {
  pdf: PDFDocumentProxy; page: number; zoom: number; passages: Passage[]; selected: string | null; edited: Set<string>;
  onSelect: (id: string) => void; onError: (error: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [available, setAvailable] = useState(700);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [ready, setReady] = useState(false);
  const scale = dimensions.width ? Math.min(1.5, available / dimensions.width) * zoom : 1;
  useEffect(() => {
    const observer = new ResizeObserver(entries => setAvailable(Math.max(200, entries[0].contentRect.width - 32)));
    observer.observe(host.current!);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    let cancelled = false;
    let render: RenderTask | undefined;
    setReady(false);
    async function draw() {
      const current = await pdf.getPage(page);
      if (cancelled) return;
      const base = current.getViewport({ scale: 1 });
      if (base.width !== dimensions.width || base.height !== dimensions.height) {
        setDimensions({ width: base.width, height: base.height });
        return;
      }
      const viewport = current.getViewport({ scale });
      const density = Math.min(window.devicePixelRatio || 1, 3, Math.sqrt(16_000_000 / (viewport.width * viewport.height)));
      const element = document.createElement('canvas');
      element.width = Math.ceil(viewport.width * density);
      element.height = Math.ceil(viewport.height * density);
      render = current.render({ canvas: element, viewport, transform: [density, 0, 0, density, 0, 0] });
      await render.promise;
      if (!cancelled) {
        const target = canvas.current!;
        target.width = element.width; target.height = element.height;
        target.getContext('2d')!.drawImage(element, 0, 0);
        setReady(true);
      }
    }
    void draw().catch(cause => { if (!cancelled && cause?.name !== 'RenderingCancelledException') onError('The PDF preview could not be rendered.'); });
    return () => { cancelled = true; render?.cancel(); };
  }, [pdf, page, dimensions, scale, onError]);
  return <div className="preview-scroll" ref={host}>
    {!ready && <div className="preview-loading" role="status">Loading page…</div>}
    <div className="editor-paper" data-ready={ready} style={{ width: dimensions.width * scale, height: dimensions.height * scale }} aria-label={`Page ${page} preview`}>
      <canvas ref={canvas} />
      <div className="source-text-layer" aria-label="Original text passages">
        {passages.flatMap(item => (item.boxes || [item]).map((box, index) => <button key={`${item.id}:${index}`} className={`source-passage ${selected === item.id ? 'selected' : ''} ${edited.has(item.id) ? 'edited' : ''}`}
          style={{ left: box.x * scale, top: box.y * scale, width: box.width * scale, height: box.height * scale }}
          aria-label={index === 0 ? `Edit original text: ${item.text}` : `Select paragraph continuation ${index}`}
          aria-pressed={selected === item.id} onClick={() => onSelect(item.id)} title={item.text} />))}
      </div>
    </div>
  </div>;
}

export default function Editor() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [original, setOriginal] = useState<PDFDocumentProxy | null>(null);
  const [preview, setPreview] = useState<PDFDocumentProxy | null>(null);
  const [filename, setFilename] = useState('document');
  const [history, setHistory] = useState<History>({ past: [], present: [], future: [] });
  const [passages, setPassages] = useState<Passage[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [units, setUnits] = useState<Record<number, 'paragraph' | 'line'>>({});
  const [fontInfo, setFontInfo] = useState<FontInfo | null>(null);
  const [fontError, setFontError] = useState('');
  const fontCache = useRef(new Map<string, FontInfo>());
  const [page, setPage] = useState(1);
  const unit = units[page] || 'paragraph';
  const [zoom, setZoom] = useState(1);
  const [error, setError] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [loadingPassages, setLoadingPassages] = useState(true);
  const [saveStatus, setSaveStatus] = useState('Saved');
  const [previewing, setPreviewing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const group = useRef('');
  const saved = useRef('');
  const saveQueue = useRef<Promise<unknown>>(Promise.resolve());
  const edits = history.present;
  const currentEdits = useRef(edits);
  currentEdits.current = edits;
  const source = passages.find(item => item.id === selected);
  const current = source ? edits.find(item => item.id === source.id) || source : null;
  const json = JSON.stringify({ edits });
  const latest = useRef({ loaded, json });
  latest.current = { loaded, json };
  const editedIds = new Set(edits.map(item => item.id));
  const pageCount = (preview || original)?.numPages || 1;

  useEffect(() => { setPage(value => Math.min(value, pageCount)); }, [pageCount]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const task = getDocument({ url: `/api/jobs/${id}/download?original=true`,
      cMapUrl: '/pdfjs/cmaps/', cMapPacked: true, standardFontDataUrl: '/pdfjs/standard_fonts/', wasmUrl: '/pdfjs/wasm/', iccUrl: '/pdfjs/iccs/',
    });
    async function load() {
      const [pdfDocument, draftResponse, jobResponse] = await Promise.all([
        task.promise,
        fetch(`/api/jobs/${id}/editor`, { signal: controller.signal, cache: 'no-store' }),
        fetch(`/api/jobs/${id}`, { signal: controller.signal, cache: 'no-store' }),
      ]);
      if (!draftResponse.ok) throw new Error(await errorMessage(draftResponse));
      if (!jobResponse.ok) throw new Error(await errorMessage(jobResponse));
      const draft: { edits?: Edit[] } = await draftResponse.json();
      const job: { filename: string } = await jobResponse.json();
      if (cancelled) return;
      setOriginal(pdfDocument);
      setFilename(job.filename);
      setHistory({ past: [], present: draft.edits || [], future: [] });
      saved.current = JSON.stringify({ edits: draft.edits || [] });
      setLoaded(true);
    }
    void load().catch(cause => { if (!cancelled) setError(cause instanceof Error ? cause.message : 'The editor could not be loaded.'); });
    return () => { cancelled = true; controller.abort(); void task.destroy(); };
  }, [id]);

  useEffect(() => {
    if (!loaded) return;
    const controller = new AbortController();
    setLoadingPassages(true); setPassages([]); setSelected(null);
    if (original && page > original.numPages) { setLoadingPassages(false); return; }
    void fetch(`/api/jobs/${id}/passages?page=${page}&unit=${unit}`, { signal: controller.signal, cache: 'no-store' }).then(async response => {
      if (!response.ok) throw new Error(await errorMessage(response));
      const result: (Passage & { font: string })[] = await response.json();
      if (!controller.signal.aborted) {
        // Old drafts and pages edited as individual fragments must reopen in
        // that mode, rather than hiding edits inside a newly grouped paragraph.
        if (unit === 'paragraph' && currentEdits.current.some(edit => edit.page === page && !result.some(item => item.id === edit.id))) {
          setUnits(previous => ({ ...previous, [page]: 'line' }));
          return;
        }
        setPassages(result.map(item => ({ ...item, originalFont: item.font, font: 'auto', overflow: 'wrap' })));
        setLoadingPassages(false);
      }
    }).catch(cause => { if (!controller.signal.aborted) { setError(cause.message); setLoadingPassages(false); } });
    return () => controller.abort();
  }, [id, loaded, page, original, unit]);

  useEffect(() => {
    setFontInfo(null); setFontError('');
    if (!selected || !source || source.page !== page) return;
    const cacheKey = `${id}:${page}:${selected}`;
    if (fontCache.current.has(cacheKey)) { setFontInfo(fontCache.current.get(cacheKey)!); return; }
    const controller = new AbortController();
    void fetch(`/api/jobs/${id}/font?page=${page}&passage=${encodeURIComponent(selected)}`, { signal: controller.signal, cache: 'no-store' }).then(async response => {
      if (!response.ok) throw new Error(await errorMessage(response));
      const result: FontInfo = await response.json();
      if (!controller.signal.aborted) { fontCache.current.set(cacheKey, result); setFontInfo(result); }
    }).catch(cause => { if (!controller.signal.aborted) setFontError(cause.message); });
    return () => controller.abort();
  }, [id, page, selected, source]);

  const save = useCallback((body: string) => {
    const request = saveQueue.current.catch(() => {}).then(async () => {
      if (saved.current === body) return;
      const response = await fetch(`/api/jobs/${id}/editor`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body });
      if (!response.ok) throw new Error(await errorMessage(response));
      saved.current = body;
    });
    saveQueue.current = request;
    return request;
  }, [id]);

  useEffect(() => {
    if (!loaded || saved.current === json) { setSaveStatus('Saved'); return; }
    let cancelled = false;
    setSaveStatus('Unsaved changes');
    const timer = setTimeout(() => {
      setSaveStatus('Saving…');
      void save(json).then(() => { if (!cancelled) { setSaveStatus('Saved'); setError(''); } }).catch(cause => { if (!cancelled) { setSaveStatus('Not saved'); setError(cause.message); } });
    }, 650);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [json, loaded, save]);

  useEffect(() => {
    if (!loaded) return;
    if (!edits.length) { setPreview(null); setPreviewing(false); return; }
    let cancelled = false;
    const controller = new AbortController();
    let task: ReturnType<typeof getDocument> | undefined;
    let published = false;
    setPreviewing(true);
    const timer = setTimeout(() => {
      void (async () => {
        const response = await fetch(`/api/jobs/${id}/export`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: json, signal: controller.signal });
        if (!response.ok) throw new Error(await errorMessage(response));
        const data = await response.arrayBuffer();
        if (cancelled) return;
        task = getDocument({ data, cMapUrl: '/pdfjs/cmaps/', cMapPacked: true, standardFontDataUrl: '/pdfjs/standard_fonts/', wasmUrl: '/pdfjs/wasm/', iccUrl: '/pdfjs/iccs/' });
        const result = await task.promise;
        if (!cancelled) { published = true; setPreview(result); setPreviewing(false); }
      })().catch(cause => { if (!cancelled) { setPreviewing(false); setError(cause.message); } });
    }, 400);
    return () => { cancelled = true; clearTimeout(timer); controller.abort(); if (task && !published) void task.destroy(); };
  }, [id, loaded, json, edits.length]);
  useEffect(() => () => { if (preview) void preview.loadingTask.destroy(); }, [preview]);
  useEffect(() => () => { if (latest.current.loaded && saved.current !== latest.current.json) void save(latest.current.json).catch(() => {}); }, [save]);
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => { if (loaded && saved.current !== json) event.preventDefault(); };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [json, loaded]);

  function commit(update: (items: Edit[]) => Edit[], editGroup = '') {
    setError('');
    const coalesce = !!editGroup && group.current === editGroup;
    group.current = editGroup;
    setHistory(previous => ({ past: coalesce ? previous.past : [...previous.past.slice(-49), previous.present], present: update(previous.present), future: [] }));
  }
  function patch(change: Partial<Edit>, editGroup = '') {
    if (!current || !source) return;
    const next: Edit = { id: source.id, page: source.page, text: current.text, bold: current.bold, color: current.color, overflow: current.overflow || 'wrap', font: current.font || 'auto', ...change };
    commit(items => {
      const rest = items.filter(item => item.id !== next.id);
      return next.text === source.text && next.bold === source.bold && next.color === source.color && next.font === 'auto' && next.overflow === 'wrap' ? rest : [...rest, next];
    }, editGroup);
  }
  function undo() { group.current = ''; setHistory(previous => previous.past.length ? { past: previous.past.slice(0, -1), present: previous.past.at(-1)!, future: [previous.present, ...previous.future] } : previous); }
  function redo() { group.current = ''; setHistory(previous => previous.future.length ? { past: [...previous.past.slice(-49), previous.present], present: previous.future[0], future: previous.future.slice(1) } : previous); }

  async function download() {
    setExporting(true); setError('');
    try {
      await save(json);
      const response = await fetch(`/api/jobs/${id}/export`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: json });
      if (!response.ok) throw new Error(await errorMessage(response));
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a'); link.href = url; link.download = `${filename.replace(/\.[^.]+$/, '')}-edited.pdf`;
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 30_000); setSaveStatus('Saved');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'The PDF could not be exported.'); }
    finally { setExporting(false); }
  }

  return <section className="document-editor" aria-label="Document editor" onKeyDown={event => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes((event.target as HTMLElement).tagName)) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); event.shiftKey ? redo() : undo(); }
    if (event.key === 'Escape') setSelected(null);
  }}>
    <div className="editor-heading"><div><span className="eyebrow">YOUR DOCUMENT</span><h1>Edit your document</h1><p className="filename" title={filename}>{filename}</p></div>
      <div className="editor-actions"><button className="button secondary" onClick={async () => { try { if (loaded) await save(json); navigate(`/document/${id}`); } catch (cause) { setError((cause as Error).message); } }}>Back</button><button className="button primary" onClick={() => void download()} disabled={!loaded || exporting}>{exporting ? 'Exporting…' : 'Download PDF'}</button></div>
    </div>
    {error && <p className="error editor-error" role="alert">{error}</p>}
    {!loaded ? <div className="panel editor-pending">{error ? <Link className="text-link" to={`/document/${id}`}>Return to document</Link> : <p role="status">Loading editor…</p>}</div> : <>
      <div className="editor-toolbar" role="toolbar" aria-label="Document tools">
        <span className="toolbar-label">Edit original text</span><span className="toolbar-divider" />
        <button className="tool-button" aria-label="Undo" disabled={!history.past.length} onClick={undo}>↶</button><button className="tool-button" aria-label="Redo" disabled={!history.future.length} onClick={redo}>↷</button>
        <span className="toolbar-spacer" />
        <label className="zoom-control">Zoom <select aria-label="Zoom" value={zoom} onChange={event => setZoom(Number(event.target.value))}><option value={0.75}>75%</option><option value={1}>Fit</option><option value={1.5}>150%</option><option value={2}>200%</option></select></label>
        <span className="save-status" role="status">{saveStatus}</span>
      </div>
      <div className="editor-layout"><div className="preview-column">
        <div className="page-controls"><button className="tool-button" aria-label="Previous page" disabled={page === 1} onClick={() => setPage(page-1)}>←</button><span>Page {page} / {pageCount}</span><button className="tool-button" aria-label="Next page" disabled={page >= pageCount} onClick={() => setPage(page+1)}>→</button></div>
        <PagePreview pdf={preview || original!} page={Math.min(page, pageCount)} zoom={zoom} passages={passages} selected={selected} edited={editedIds} onSelect={setSelected} onError={setError} />
        <p className="editor-hint" role="status">{previewing ? 'Updating preview…' : `Click an existing ${unit === 'paragraph' ? 'paragraph' : 'line'} to edit its text, font, weight or color.`}</p>
      </div><aside className="text-properties" aria-label="Text properties">
        <label className="field-label editing-unit">Edit as<select aria-label="Editing unit" value={unit} disabled={edits.some(edit => edit.page === page)} onChange={event => { const value = event.target.value as 'paragraph' | 'line'; setUnits(previous => ({ ...previous, [page]: value })); }}><option value="paragraph">Paragraphs</option><option value="line">Individual lines</option></select>{edits.some(edit => edit.page === page) && <span>Restore this page’s edits to change the selection mode.</span>}</label>
        <div className="properties-title"><h2>{current ? (source?.lineCount && source.lineCount > 1 ? 'Original paragraph' : 'Original text') : 'Select text'}</h2>{current && editedIds.has(current.id) && <button className="tool-button" onClick={() => commit(items => items.filter(item => item.id !== current.id))}>Restore original</button>}</div>
        {current ? <><label className="field-label">Text<textarea aria-label="Text content" value={current.text} maxLength={4000} rows={source?.lineCount && source.lineCount > 1 ? 8 : 5} onChange={event => patch({ text: event.target.value }, `${selected}:text`)} onBlur={() => { group.current = ''; }} /></label>
          <label className="field-label overflow-field">When text is longer<select aria-label="Text overflow" value={current.overflow || 'wrap'} onChange={event => patch({ overflow: event.target.value as Overflow })}>
            <option value="wrap">Wrap in free space</option><option value="shrink">Slightly reduce text size</option><option value="page">Continue on a new page</option>
          </select><span>{(current.overflow || 'wrap') === 'wrap' ? 'Keep the original size and wrap within the text column. Stop before existing content; any remaining text continues on an extra page.' : current.overflow === 'shrink' ? 'Use the full column width, then reduce by up to 15% to fit the original height. Any remaining text wraps or continues on an extra page.' : 'Use the full column width at the original size and height. Any text that does not fit continues on an extra page.'}</span></label>
          <label className="field-label font-field">Font<select aria-label="Text font" value={current.font || 'auto'} onChange={event => patch({ font: event.target.value })}>
            <option value="auto">{fontInfo ? `Automatic · ${fontInfo.name}` : 'Automatic · matching original…'}</option>{fontInfo ? <><optgroup label="Similar fonts">{fontInfo.similar.map(value => <option key={value} value={value}>{fonts.find(item => item[0] === value)?.[1] || value}</option>)}</optgroup><optgroup label="Other fonts">{fonts.filter(([value]) => !fontInfo.similar.includes(value)).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</optgroup></> : fonts.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select><span>{fontInfo ? `${fontInfo.exact ? 'Original' : 'Estimated'}: ${fontInfo.name} · ${fontInfo.size} pt${fontInfo.mixed ? ' · mixed styles' : ''}` : fontError || 'Matching the original font and size…'}</span></label>
          <div className="format-controls"><button className={`tool-button format-bold ${current.bold ? 'active' : ''}`} aria-label="Bold" aria-pressed={current.bold} onClick={() => patch({ bold: !current.bold })}>B</button><label className="field-label color-field">Color<input type="color" aria-label="Text color" value={current.color} onChange={event => patch({ color: event.target.value }, `${selected}:color`)} /></label></div>
          <p className="properties-note">{source!.scanned ? 'Bold and color preserve the scanned letter shapes. Rewritten paragraphs use the estimated font or your chosen alternative. An exact font match cannot be guaranteed from an image.' : `Original font: ${source!.originalFont}. Embedded glyphs are reused when available; unavailable glyphs use a matching fallback.`}</p>
        </> : <p className="properties-note">{page > original!.numPages ? 'Continuation page. Return to the original page to edit this text.' : loadingPassages ? 'Finding original text…' : passages.length ? 'Choose a paragraph on the page or below.' : 'No editable horizontal text was recognized on this page.'}</p>}
        {passages.length > 0 && <div className="text-box-list"><span className="field-label">Original text on this page</span>{passages.map(item => <button key={item.id} className={`text-box-entry ${selected === item.id ? 'active' : ''}`} onClick={() => setSelected(item.id)}>{edits.find(edit => edit.id === item.id)?.text ?? item.text}</button>)}</div>}
      </aside></div>
      <p className="draft-note">Drafts expire after one hour of inactivity. Download your PDF to keep your changes.</p>
    </>}
  </section>;
}

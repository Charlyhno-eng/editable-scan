import { StrictMode, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Link, Route, Routes, useNavigate, useParams } from 'react-router';
import './style.css';

type Job = {
  id: string;
  filename: string;
  status: 'uploading' | 'queued' | 'processing' | 'done' | 'error';
  done: number;
  total: number;
  error: string | null;
  report?: {
    similarity_percent: number;
    method: string;
    source_bytes: number;
    output_bytes: number;
    size_reduction_percent: number;
    pages: { page: number; similarity_percent: number; summary: string }[];
  } | null;
};

function fileSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function QualityReport({ report }: { report: NonNullable<Job['report']> }) {
  return <div className="quality-report">
    <div className="quality-metrics"><div><strong>{report.similarity_percent.toFixed(1)}%</strong><span>Visual similarity</span></div><div><strong>{fileSize(report.output_bytes)}</strong><span>PDF size</span></div></div>
    <p className="quality-note">Compared with the source. Similarity measures appearance, not text accuracy.</p>
    <details><summary>Comparison details</summary>
      <p>Source: {fileSize(report.source_bytes)} · PDF: {fileSize(report.output_bytes)}{report.size_reduction_percent > 0 ? ` · ${report.size_reduction_percent}% smaller` : ''}</p>
      {report.pages.map(page => <p key={page.page}><b>Page {page.page} · {page.similarity_percent.toFixed(1)}%</b><br />{page.summary}</p>)}
      <p>{report.method}</p>
    </details>
  </div>;
}

function Icon({ name, className = '' }: { name: 'upload' | 'file' | 'download' | 'check' | 'arrow' | 'text'; className?: string }) {
  const paths = {
    upload: <><path d="M12 16V4m-4 4 4-4 4 4" /><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" /></>,
    file: <><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z" /><path d="M14 3v6h6M8 14h8m-8 3h5" /></>,
    download: <><path d="M12 3v12m-4-4 4 4 4-4" /><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    text: <path d="M5 5h14M12 5v14m-4 0h8" />,
  };
  return <svg className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

async function errorMessage(response: Response) {
  try {
    const body = await response.json();
    return typeof body.detail === 'string' ? body.detail : 'The file could not be processed.';
  } catch {
    return 'The server is unavailable. Try again shortly.';
  }
}

function Upload() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const busy = useRef(false);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [filename, setFilename] = useState('');

  async function convert(files: FileList | null) {
    if (!files?.length || busy.current) return;
    setError('');
    if (files.length > 1) { setError('Upload one file at a time.'); return; }
    const file = files[0];
    if (!file.size) { setError('This file is empty.'); return; }
    if (file.size > 50 * 1024 * 1024) { setError('The file exceeds 50 MB.'); return; }
    busy.current = true;
    setFilename(file.name);
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch('/api/jobs', { method: 'POST', body: form });
      if (!response.ok) throw new Error(await errorMessage(response));
      const job: Job = await response.json();
      navigate(`/document/${job.id}`);
    } catch (cause) {
      setError(cause instanceof TypeError ? 'Unable to connect. Make sure the server is running.' : cause instanceof Error ? cause.message : 'Conversion failed.');
    } finally {
      busy.current = false;
      setUploading(false);
      if (input.current) input.current.value = '';
    }
  }

  return <>
    <div className="intro"><span className="eyebrow">FROM SCAN TO TEXT</span><h1>Your documents,<br /><span>ready to copy.</span></h1><p>An image or a PDF. Text you can copy.</p></div>
    <section className={`panel dropzone ${dragging ? 'dragging' : ''}`} aria-label="Drop a document"
      onDragEnter={event => { event.preventDefault(); dragDepth.current++; setDragging(true); }}
      onDragOver={event => event.preventDefault()}
      onDragLeave={event => { event.preventDefault(); dragDepth.current--; if (dragDepth.current === 0) setDragging(false); }}
      onDrop={event => { event.preventDefault(); dragDepth.current = 0; setDragging(false); void convert(event.dataTransfer.files); }}>
      <input ref={input} id="document" type="file" accept=".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp,.gif,application/pdf,image/png,image/jpeg,image/webp,image/tiff,image/bmp,image/gif" className="visually-hidden" tabIndex={-1} aria-label="Choose a document" onChange={event => void convert(event.target.files)} disabled={uploading} />
      <div className={`upload-art ${uploading ? 'working' : ''}`}><div className="paper-back" /><div className="paper-front"><Icon name="file" /></div><span className="art-badge"><Icon name="upload" /></span></div>
      <div aria-live="polite">{uploading ? <><h2>Uploading document…</h2><p className="filename">{filename}</p></> : <><h2>{dragging ? 'Drop your file here' : 'Drag your document here'}</h2><p>or choose one from your device</p></>}</div>
      <button className="button primary" onClick={() => input.current?.click()} disabled={uploading}>{uploading ? <><span className="spinner" /> Uploading</> : <>Choose a file <Icon name="arrow" /></>}</button>
      <span className="formats">PDF, PNG, JPG, WebP, TIFF, BMP, GIF · 50 MB max.</span>
    </section>
    {error && <p className="error" role="alert">{error}</p>}
    <div className="output-note"><Icon name="text" /><span>Sharper scans. Selectable text.</span></div>
  </>;
}

function Document() {
  const { id } = useParams();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setError('');
    setJob(null);
    async function poll() {
      try {
        const response = await fetch(`/api/jobs/${id}`, { signal: controller.signal, cache: 'no-store' });
        if (!response.ok) throw new Error(await errorMessage(response));
        const next: Job = await response.json();
        if (controller.signal.aborted) return;
        setJob(next);
        if (next.status !== 'done' && next.status !== 'error') timer = setTimeout(poll, 800);
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause instanceof TypeError ? 'Connection lost. Your document is still being processed.' : cause instanceof Error ? cause.message : 'Unable to retrieve the document.');
      }
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [id, attempt]);

  const done = job?.status === 'done';
  const failed = !!error || job?.status === 'error';
  const progress = job?.total ? Math.round(job.done / job.total * 100) : 0;
  return <>
    <div className="intro compact"><span className="eyebrow">FROM SCAN TO TEXT</span><h1>{done ? <>Your document, <span>ready.</span></> : failed ? 'Something went wrong.' : <>Your text <span>is taking shape.</span></>}</h1></div>
    <section className="panel result" aria-live="polite" aria-busy={!done && !failed}>
      <div className={`result-icon ${done ? 'success' : failed ? 'failure' : 'processing'}`}>{done ? <Icon name="check" /> : failed ? <span>!</span> : <Icon name="text" />}</div>
      <h2>{done ? 'Your PDF is ready' : failed ? 'Conversion interrupted' : job?.status === 'queued' ? 'Waiting in queue…' : job?.done && job.done === job.total ? 'Comparing visual quality…' : 'Recognizing text…'}</h2>
      {job && <p className="filename" title={job.filename}>{job.filename}</p>}
      {failed ? <><p className="error" role="alert">{error || job?.error}</p>{error && <button className="button secondary" onClick={() => setAttempt(value => value + 1)}>Try again</button>}</> : done ? <><p className="result-caption">{job.total} {job.total > 1 ? 'pages' : 'page'} · Selectable text</p><a className="button primary" href={`/api/jobs/${id}/download`} download><Icon name="download" /> Download PDF</a>{job.report && <QualityReport report={job.report} />}</> : <div className="progress-area"><div className={`progress-track ${!job?.done ? 'indeterminate' : ''}`} role="progressbar" aria-label="Pages processed" aria-valuemin={0} aria-valuemax={100} aria-valuenow={job?.done ? progress : undefined}><span style={{ width: `${progress}%` }} /></div><p>{job?.done ? `${job.done} / ${job.total} pages` : 'Preparing document'}</p></div>}
    </section>
    <Link className="text-link" to="/">{done || failed ? 'Convert another file' : 'Back to home'}</Link>
  </>;
}

function NotFound() {
  return <div className="intro"><h1>Page not found.</h1><Link className="button primary" to="/">Choose a document</Link></div>;
}

function App() {
  return <div className="app"><header><Link className="brand" to="/" aria-label="Editable Scan — home"><span className="brand-icon"><Icon name="text" /></span>editable<span className="brand-light">scan</span></Link><span className="local-label"><i /> Local processing</span></header><main><Routes><Route path="/" element={<Upload />} /><Route path="/document/:id" element={<Document />} /><Route path="*" element={<NotFound />} /></Routes></main><footer><span>Figures and layout preserved.</span><span>Editable Scan</span></footer></div>;
}

createRoot(document.getElementById('root')!).render(<StrictMode><BrowserRouter><App /></BrowserRouter></StrictMode>);

import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const image = fileURLToPath(new URL('../../tests/fixtures/scan.png', import.meta.url));
const pdf = fileURLToPath(new URL('../../tests/fixtures/scan.pdf', import.meta.url));
const python = fileURLToPath(new URL('../../.venv/bin/python', import.meta.url));

async function openEditor(page: import('@playwright/test').Page, file = image) {
  await page.goto('/');
  await page.getByLabel('Choose a document').setInputFiles(file);
  await page.getByRole('link', { name: 'Edit document' }).click({ timeout: 45_000 });
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  await expect(page.getByRole('button', { name: /Edit original text: Bonjour/ })).toBeVisible();
}

test('edit original text, style, undo, reload and export without duplicate old text', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await openEditor(page);
  await expect(page.getByRole('button', { name: /Add text/ })).toHaveCount(0);
  await page.getByRole('button', { name: /Edit original text: Bonjour/ }).click();
  await expect(page.getByRole('textbox', { name: 'Text content' })).toHaveValue('Bonjour, été français.');
  await page.getByRole('button', { name: 'Bold', exact: true }).click();
  await page.getByLabel('Text color').fill('#bb2255');
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  await page.screenshot({ path: '/tmp/editable-scan-original-style.png', fullPage: true });
  await page.getByRole('textbox', { name: 'Text content' }).fill('Bonjour, nouvel essai.');
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  await page.reload();
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  await page.getByRole('button', { name: /Edit original text: Bonjour/ }).click();
  await expect(page.getByRole('textbox', { name: 'Text content' })).toHaveValue('Bonjour, nouvel essai.');
  await expect(page.getByRole('button', { name: 'Bold', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('textbox', { name: 'Text content' }).fill('漢');
  await expect(page.getByRole('alert')).toContainText('does not support');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Text content' })).toHaveValue('Bonjour, nouvel essai.');
  await page.getByRole('button', { name: 'Restore original', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Text content' })).toHaveValue('Bonjour, été français.');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Text content' })).toHaveValue('Bonjour, nouvel essai.');
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  await page.screenshot({ path: '/tmp/editable-scan-original-rewrite.png', fullPage: true });
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download PDF' }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('scan-edited.pdf');
  const output = testInfo.outputPath('edited.pdf');
  await download.saveAs(output);
  const text = execFileSync(python, ['-c', 'import pymupdf,sys; d=pymupdf.open(sys.argv[1]); print(d[0].get_text())', output], { encoding: 'utf8' });
  expect(text).not.toContain('Bonjour, été français.');
  expect(text.split('Bonjour, nouvel essai.')).toHaveLength(2);
  expect(text).toContain('Un document');
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(page.getByText('Measured before your text edits.', { exact: false })).toBeVisible();
  await page.screenshot({ path: '/tmp/editable-scan-result-actions.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('edit separate original pages and use editor on mobile', async ({ page }) => {
  await openEditor(page, pdf);
  await page.getByRole('button', { name: /Edit original text: Bonjour/ }).click();
  await page.getByLabel('Text color').fill('#bb2255');
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  await page.getByRole('button', { name: 'Next page' }).click();
  await expect(page.getByRole('button', { name: /Edit original text: Bonjour/ })).toBeVisible();
  await page.getByRole('button', { name: /Edit original text: Bonjour/ }).click();
  await expect(page.getByLabel('Text color')).toHaveValue('#000000');
  await page.getByRole('button', { name: 'Bold', exact: true }).click();
  await page.getByRole('button', { name: 'Previous page' }).click();
  await page.getByRole('button', { name: /Edit original text: Bonjour/ }).click();
  await expect(page.getByLabel('Text color')).toHaveValue('#bb2255');
  await expect(page.getByRole('button', { name: 'Bold', exact: true })).toHaveAttribute('aria-pressed', 'false');
  await page.getByLabel('Zoom', { exact: true }).selectOption('2');
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  await page.getByLabel('Zoom', { exact: true }).selectOption('1');
  await page.setViewportSize({ width: 375, height: 812 });
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await page.screenshot({ path: '/tmp/editable-scan-original-mobile.png', fullPage: true });
});

test('report editor expiry clearly', async ({ page }) => {
  await page.goto('/document/missing/edit');
  await expect(page.getByRole('alert')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Return to document' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Download PDF' })).toBeDisabled();
});

test('serve PDF.js resources at the URLs requested by its worker', async ({ request }) => {
  const resources = [
    '/pdfjs/wasm/qcms_bg.wasm',
    '/pdfjs/wasm/openjpeg.wasm',
    '/pdfjs/wasm/jbig2.wasm',
    '/pdfjs/iccs/CGATS001Compat-v2-micro.icc',
    '/pdfjs/standard_fonts/LiberationSans-Regular.ttf',
    '/pdfjs/cmaps/Adobe-Japan1-UCS2.bcmap',
  ];
  for (const url of resources) {
    const response = await request.get(url);
    expect(response.status(), url).toBe(200);
    expect(response.headers()['content-type'], url).not.toContain('text/html');
    const body = await response.body();
    expect(body.length, url).toBeGreaterThan(100);
    if (url.endsWith('.wasm')) expect([...body.subarray(0, 4)]).toEqual([0, 97, 115, 109]);
  }
});

test('save long replacements, preview continuation pages and undo without 422 errors', async ({ page }) => {
  const failures: string[] = [];
  page.on('response', response => { if (response.url().includes('/api/') && response.status() >= 400) failures.push(`${response.status()} ${response.url()}`); });
  await openEditor(page);
  await page.getByRole('button', { name: /Edit original text: Bonjour/ }).click();
  const replacement = 'A longer replacement keeps the original text size. '.repeat(60).trim();
  const saved = page.waitForResponse(response => response.url().endsWith('/editor') && response.request().method() === 'PUT' && response.ok(), { timeout: 30_000 });
  const rendered = page.waitForResponse(response => response.url().endsWith('/export') && response.ok(), { timeout: 30_000 });
  await page.getByRole('textbox', { name: 'Text content' }).fill(replacement);
  await Promise.all([saved, rendered]);
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Next page' })).toBeEnabled({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Next page' }).click();
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByRole('button', { name: 'Previous page' }).click();
  await page.getByRole('button', { name: /Edit original text: Bonjour/ }).click();
  await expect(page.getByRole('textbox', { name: 'Text content' })).toHaveValue(replacement);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Next page' })).toBeDisabled();
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  expect(failures).toEqual([]);
});

test('edit a complete paragraph, choose a font and control each overflow mode', async ({ page }, testInfo) => {
  const input = testInfo.outputPath('paragraph.pdf');
  execFileSync(python, ['-c', 'from tests.test_paragraphs import paragraph_source; import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(paragraph_source())', input]);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByLabel('Choose a document').setInputFiles(input);
  await page.getByRole('link', { name: 'Edit document' }).click({ timeout: 45_000 });
  const paragraph = page.getByRole('button', { name: /Edit original text: First paragraph/ });
  await paragraph.click();
  await expect(page.getByRole('heading', { name: 'Original paragraph' })).toBeVisible();
  await expect(page.getByLabel('Text content')).toHaveValue('First paragraph line with several words. Second line in the same paragraph.');
  await expect(page.getByText('Original: Helvetica · 12 pt', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Editing unit')).toHaveValue('paragraph');

  async function saveChange(action: () => Promise<unknown>) {
    const response = page.waitForResponse(response => response.url().endsWith('/editor') && response.request().method() === 'PUT', { timeout: 30_000 });
    await action();
    expect((await response).status()).toBe(200);
    await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  }
  // A layout preference can be chosen before changing the original text.
  await saveChange(() => page.getByLabel('Text overflow').selectOption('shrink'));
  await expect(page.getByLabel('Text overflow')).toHaveValue('shrink');
  await saveChange(() => page.getByLabel('Text font').selectOption('LiberationSerif'));
  // Three full-width lines fit the original height only after slight shrinking.
  const replacement = 'A complete paragraph with new words and the selected font. '.repeat(3).trim();
  await saveChange(() => page.getByLabel('Text content').fill(replacement));
  await expect(page.getByRole('button', { name: 'Next page' })).toBeDisabled();
  await saveChange(() => page.getByLabel('Text overflow').selectOption('page'));
  await expect(page.getByRole('button', { name: 'Next page' })).toBeEnabled();
  await saveChange(() => page.getByLabel('Text overflow').selectOption('wrap'));
  await expect(page.getByRole('button', { name: 'Next page' })).toBeDisabled();
  await saveChange(() => page.getByLabel('Text overflow').selectOption('page'));
  await page.reload();
  await paragraph.click();
  await expect(page.getByLabel('Text content')).toHaveValue(replacement);
  await expect(page.getByLabel('Text font')).toHaveValue('LiberationSerif');
  await expect(page.getByLabel('Text overflow')).toHaveValue('page');
  await expect(page.getByRole('button', { name: 'Next page' })).toBeEnabled();
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  await page.screenshot({ path: '/tmp/editable-scan-paragraph-editor.png', fullPage: true });

  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download PDF' }).click();
  const output = testInfo.outputPath('paragraph-edited.pdf');
  await (await downloadEvent).saveAs(output);
  const text = execFileSync(python, ['-c', 'import pymupdf,sys; d=pymupdf.open(sys.argv[1]); print(" ".join(p.get_text() for p in d))', output], { encoding: 'utf8' });
  expect(text).not.toContain('First paragraph line');
  expect(text).not.toContain('Second line in the same');
  expect(text.replaceAll(/\s/g, '')).toContain(replacement.replaceAll(/\s/g, ''));
  expect(text).toContain('Separate paragraph');
  await saveChange(() => page.getByRole('button', { name: 'Restore original', exact: true }).click());
  await expect(page.getByLabel('Text font')).toHaveValue('auto');
  await expect(page.getByRole('button', { name: 'Next page' })).toBeDisabled();
  await page.getByLabel('Editing unit').selectOption('line');
  await expect(page.getByRole('button', { name: 'Edit original text: First paragraph line with several words.', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('expand a short paragraph to a full line without premature wrapping', async ({ page }, testInfo) => {
  const input = testInfo.outputPath('short-paragraph.pdf');
  execFileSync(python, ['-c', 'from tests.test_paragraphs import paragraph_source; import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(paragraph_source(phrases=["Two words", "Short end"]))', input]);
  await page.goto('/');
  await page.getByLabel('Choose a document').setInputFiles(input);
  await page.getByRole('link', { name: 'Edit document' }).click({ timeout: 45_000 });
  await page.getByRole('button', { name: /Edit original text: Two words/ }).click();
  const replacement = 'Two words followed by enough words to fill the available line.';
  const saved = page.waitForResponse(response => response.url().endsWith('/editor') && response.request().method() === 'PUT');
  const rendered = page.waitForResponse(response => response.url().endsWith('/export') && response.request().method() === 'POST');
  await page.getByLabel('Text content').fill(replacement);
  expect((await saved).status()).toBe(200);
  expect((await rendered).status()).toBe(200);
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Next page' })).toBeDisabled();
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  await expect(page.getByRole('status').filter({ hasText: /^Updating preview…$/ })).toHaveCount(0);
  await page.screenshot({ path: '/tmp/editable-scan-flow-editor.png', fullPage: true });
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download PDF' }).click();
  const output = testInfo.outputPath('full-line.pdf');
  await (await downloadEvent).saveAs(output);
  const script = 'import pymupdf,sys,json; d=pymupdf.open(sys.argv[1]); s=[s for b in d[0].get_text("dict")["blocks"] for l in b.get("lines",[]) for s in l["spans"] if s["text"] != "Separate paragraph"]; print(json.dumps({"pages": len(d), "spans": s}))';
  const result = JSON.parse(execFileSync(python, ['-c', script, output], { encoding: 'utf8' }));
  expect(result.pages).toBe(1);
  expect(result.spans).toHaveLength(1);
  expect(result.spans[0].text).toBe(replacement);
  expect(result.spans[0].size).toBeCloseTo(12);
  expect(result.spans[0].bbox[2]).toBeLessThanOrEqual(460.5);
});

test('save status remains visible while a slower preview is loading', async ({ page }) => {
  await openEditor(page);
  await page.getByRole('button', { name: /Edit original text: Bonjour/ }).click();
  let release: () => void = () => {};
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/export', async route => { await gate; await route.continue(); });
  await page.getByLabel('Text content').fill('Bonjour, revised text.');
  await expect(page.getByRole('status').filter({ hasText: /^Updating preview…$/ })).toBeVisible();
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  release();
  await expect(page.getByRole('status').filter({ hasText: /^Updating preview…$/ })).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('remember paragraph and line selection modes independently on different pages', async ({ page }, testInfo) => {
  const input = testInfo.outputPath('two-paragraph-pages.pdf');
  execFileSync(python, ['-c', 'from tests.test_paragraphs import paragraph_source; import pymupdf,sys; s=pymupdf.open(stream=paragraph_source(),filetype="pdf"); d=pymupdf.open(); d.insert_pdf(s); d.insert_pdf(s); d.save(sys.argv[1])', input]);
  const failures: number[] = [];
  page.on('response', response => { if (response.url().includes('/api/') && response.status() >= 400) failures.push(response.status()); });
  await page.goto('/');
  await page.getByLabel('Choose a document').setInputFiles(input);
  await page.getByRole('link', { name: 'Edit document' }).click({ timeout: 45_000 });
  await expect(page.locator('.editor-paper')).toHaveAttribute('data-ready', 'true');
  await page.getByLabel('Editing unit').selectOption('line');
  const line = page.getByRole('button', { name: 'Edit original text: First paragraph line with several words.', exact: true });
  await line.click();
  await page.getByLabel('Text content').fill('First page line replacement');
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  await page.getByRole('button', { name: 'Next page' }).click();
  await expect(page.getByLabel('Editing unit')).toHaveValue('paragraph');
  const paragraph = page.getByRole('button', { name: /Edit original text: First paragraph/ });
  await paragraph.click();
  await expect(page.getByRole('heading', { name: 'Original paragraph' })).toBeVisible();
  await page.getByLabel('Text content').fill('Second page paragraph replacement');
  await expect(page.getByRole('status').filter({ hasText: /^Saved$/ })).toBeVisible();
  await page.getByRole('button', { name: 'Previous page' }).click();
  await expect(page.getByLabel('Editing unit')).toHaveValue('line');
  await line.click();
  await expect(page.getByLabel('Text content')).toHaveValue('First page line replacement');
  await page.reload();
  await expect(page.getByLabel('Editing unit')).toHaveValue('line');
  await line.click();
  await expect(page.getByLabel('Text content')).toHaveValue('First page line replacement');
  await page.getByRole('button', { name: 'Next page' }).click();
  await expect(page.getByLabel('Editing unit')).toHaveValue('paragraph');
  await paragraph.click();
  await expect(page.getByLabel('Text content')).toHaveValue('Second page paragraph replacement');
  expect(failures).toEqual([]);
});

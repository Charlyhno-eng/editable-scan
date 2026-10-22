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

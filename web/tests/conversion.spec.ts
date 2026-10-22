import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const image = fileURLToPath(new URL('../../tests/fixtures/scan.png', import.meta.url));
const pdf = fileURLToPath(new URL('../../tests/fixtures/scan.pdf', import.meta.url));

test('convert an image, reload the result, and download a PDF', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Your documents');
  await page.getByLabel('Choose a document').setInputFiles(image);
  await expect(page).toHaveURL(/\/document\/[a-f0-9]+/);
  await expect(page.getByText('Your PDF is ready')).toBeVisible({ timeout: 45_000 });
  await page.reload();
  await expect(page.getByText('Your PDF is ready')).toBeVisible();
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download PDF' }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('scan-text.pdf');
  await page.getByRole('link', { name: 'Convert another file' }).click();
  await expect(page.getByRole('button', { name: 'Choose a file' })).toBeVisible();
});

test('convert a multipage PDF', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Choose a document').setInputFiles(pdf);
  await expect(page.getByText('Your PDF is ready')).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText('3 pages · Selectable text')).toBeVisible();
});

test('accept a dropped image', async ({ page }) => {
  await page.goto('/');
  const transfer = await page.evaluateHandle(bytes => {
    const data = new DataTransfer();
    data.items.add(new File([new Uint8Array(bytes)], 'scan.png', { type: 'image/png' }));
    return data;
  }, Array.from(readFileSync(image)));
  await page.getByRole('region', { name: 'Drop a document' }).dispatchEvent('drop', { dataTransfer: transfer });
  await expect(page.getByText('Your PDF is ready')).toBeVisible({ timeout: 45_000 });
});

test('show an invalid-file error and allow retrying', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Choose a document').setInputFiles({ name: 'broken.pdf', mimeType: 'application/pdf', buffer: Buffer.from('broken') });
  await expect(page.getByRole('alert')).toContainText('Unreadable file');
  await expect(page.getByRole('button', { name: 'Choose a file' })).toBeEnabled();
});

test('show an expired result and a way back', async ({ page }) => {
  await page.goto('/document/unknown');
  await expect(page.getByRole('alert')).toContainText('expired');
  await page.getByRole('link', { name: 'Convert another file' }).click();
  await expect(page).toHaveURL('/');
});

test('mobile upload page has no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Choose a file' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await page.screenshot({ path: '/tmp/editable-scan-mobile.png', fullPage: true });
});

test('desktop preview', async ({ page }) => {
  await page.goto('/');
  await page.screenshot({ path: '/tmp/editable-scan-desktop.png', fullPage: true });
});

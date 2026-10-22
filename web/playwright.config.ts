import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 60_000,
  fullyParallel: false,
  use: {
    baseURL: 'http://127.0.0.1:8123',
    ...devices['Desktop Chrome'],
    channel: process.env.PLAYWRIGHT_CHANNEL,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'cd .. && .venv/bin/python -m src.web',
    url: 'http://127.0.0.1:8123/api/health',
    env: { PORT: '8123' },
    reuseExistingServer: false,
  },
});

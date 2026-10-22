![Editable Scan banner](assets/editable-scan-banner.png)

# Editable Scan

Turn scanned images, document photos, and PDFs into searchable PDFs with selectable text. Crop and straighten pages, improve scans, and reduce file size while preserving figures and layout. Edit and format existing text in the built-in editor before downloading.

Forked from [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF).

## Quick start

Install (requires Python 3.12, uv, and Node.js 24):

```sh
uv sync --python 3.12 && npm ci --prefix web && npm run build --prefix web
```

Run:

```sh
uv run editable-scan
```

Open [localhost:8000](http://localhost:8000).

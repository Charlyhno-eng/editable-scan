![Editable Scan banner](assets/editable-scan-banner.png)

# Editable Scan

Turn scanned images, document photos, and PDFs into searchable PDFs with selectable text. Automatically crop and straighten photographed pages, improve scans, and reduce file size while preserving figures and layout.

Forked from [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF).

## Quick start

Install (requires Python 3.12, uv, and Node.js):

```sh
uv sync --python 3.12 && npm ci --prefix web && npm run build --prefix web
```

Run:

```sh
uv run editable-scan
```

Open [localhost:8000](http://localhost:8000).

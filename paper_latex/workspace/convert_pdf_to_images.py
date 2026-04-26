#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
    print("Using PyMuPDF (fitz)")
except ImportError:
    print("PyMuPDF not found, trying pdf2image...")
    try:
        from pdf2image import convert_from_path
        print("Using pdf2image")
    except ImportError:
        print("Neither PyMuPDF nor pdf2image available. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pymupdf"], check=True)
        import fitz

# Convert using PyMuPDF
pdf_path = Path("paper.pdf")
output_dir = Path("paper_pages")
output_dir.mkdir(exist_ok=True)

print(f"Converting {pdf_path} to PNG images at 150 DPI...")
doc = fitz.open(pdf_path)

print(f"Total pages: {len(doc)}")
zoom = 150 / 72  # Convert 150 DPI to zoom factor (72 DPI is default)

for i in range(len(doc)):
    page = doc[i]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    output_path = output_dir / f"page_{i+1:02d}.png"
    pix.save(output_path)
    print(f"Saved page {i+1} to {output_path}")

doc.close()
print(f"All pages converted to {output_dir}/")

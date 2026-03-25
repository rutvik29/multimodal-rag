"""PDF parser: extracts text chunks, images, and tables."""
from __future__ import annotations
import base64
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pdfplumber
import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def describe_image_with_gpt4v(image_bytes: bytes, context: str = "") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe this image from a document in detail. Context: {context}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        max_tokens=500
    )
    return response.choices[0].message.content


class PDFParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, pdf_path: str) -> List[Dict[str, Any]]:
        chunks = []
        doc = fitz.open(pdf_path)
        filename = Path(pdf_path).name

        for page_num, page in enumerate(doc, 1):
            # Text extraction
            text = page.get_text()
            if text.strip():
                chunks.append({
                    "type": "text",
                    "content": text,
                    "source": filename,
                    "page": page_num,
                    "metadata": {"type": "text", "page": page_num}
                })

            # Image extraction + GPT-4V description
            for img_idx, img_ref in enumerate(page.get_images()):
                xref = img_ref[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                description = describe_image_with_gpt4v(img_bytes, context=f"Page {page_num} of {filename}")
                chunks.append({
                    "type": "image",
                    "content": description,
                    "source": filename,
                    "page": page_num,
                    "metadata": {"type": "image", "page": page_num, "img_idx": img_idx}
                })

        # Table extraction with pdfplumber
        with pdfplumber.open(pdf_path) as plumb_pdf:
            for page_num, page in enumerate(plumb_pdf.pages, 1):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if table:
                        table_text = "\n".join([" | ".join([str(c) for c in row if c]) for row in table])
                        chunks.append({
                            "type": "table",
                            "content": f"Table on page {page_num}:\n{table_text}",
                            "source": filename,
                            "page": page_num,
                            "metadata": {"type": "table", "page": page_num, "table_idx": t_idx}
                        })
        return chunks

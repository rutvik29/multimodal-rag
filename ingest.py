"""Document ingestion pipeline for multimodal RAG."""
import argparse
import os
from pathlib import Path
from src.parsers.pdf_parser import PDFParser
from src.parsers.image_parser import ImageParser
from src.indexing.indexer import MultimodalIndexer


def ingest_directory(source_dir: str, collection_prefix: str = "docs"):
    parser = PDFParser()
    image_parser = ImageParser()
    indexer = MultimodalIndexer(collection_prefix=collection_prefix)

    source = Path(source_dir)
    pdf_files = list(source.glob("**/*.pdf"))
    img_files = list(source.glob("**/*.{png,jpg,jpeg}"))

    print(f"Found {len(pdf_files)} PDFs, {len(img_files)} images")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        chunks = parser.parse(str(pdf_path))
        indexer.index_chunks(chunks, source=str(pdf_path))

    for img_path in img_files:
        print(f"Processing image: {img_path.name}")
        description = image_parser.describe(str(img_path))
        indexer.index_image(description, source=str(img_path))

    print(f"Indexed {indexer.total_chunks} chunks total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source directory")
    parser.add_argument("--collection", default="docs", help="Collection name prefix")
    args = parser.parse_args()
    ingest_directory(args.source, args.collection)

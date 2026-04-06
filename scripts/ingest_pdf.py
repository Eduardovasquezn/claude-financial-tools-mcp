"""Simple CLI script to ingest PDF(s) into Qdrant."""

import sys
from pathlib import Path

from src.config import Config
from src.helpers.logs import get_logger
from src.scripts.pdf_ingestion import PDFIngestionPipeline

logger = get_logger()


def ingest_single_pdf(pipeline: PDFIngestionPipeline, pdf_path: Path, index: int = 0, total: int = 1) -> bool:
    """
    Ingest a single PDF file.
    
    Args:
        pipeline: The PDFIngestionPipeline instance
        pdf_path: Path to the PDF file
        index: Current file index (for progress display)
        total: Total number of files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        prefix = f"[{index}/{total}]" if total > 1 else ""
        print(f"\n{prefix} 📄 Processing: {pdf_path.name}")
        
        result = pipeline.ingest(pdf_path)
        
        total_chunks = result.get('total_chunks_ingested', result.get('chunks_ingested', 0))
        print(f"   ✅ Success: {total_chunks} chunks ingested")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        logger.error(f"Failed to ingest {pdf_path}: {e}")
        return False


def main():
    """Main entry point for PDF ingestion."""
    if len(sys.argv) < 2:
        print("❌ Usage: python ingest_pdf.py <path_to_pdf_or_folder>")
        print("   Single file: python ingest_pdf.py data/nvidia/NVDA-Q1-2026.pdf")
        print("   Folder:      python ingest_pdf.py data/nvidia/")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # Determine if it's a file or folder
    if not input_path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        sys.exit(1)
    
    # Collect PDF files
    pdf_files = []
    if input_path.is_file():
        if input_path.suffix.lower() != '.pdf':
            print(f"❌ Error: Not a PDF file: {input_path}")
            sys.exit(1)
        pdf_files = [input_path]
    elif input_path.is_dir():
        # Recursively find all PDFs in subdirectories
        pdf_files = sorted(input_path.glob("**/*.pdf"))
        if not pdf_files:
            print(f"❌ Error: No PDF files found in: {input_path}")
            sys.exit(1)
    else:
        print(f"❌ Error: Invalid path: {input_path}")
        sys.exit(1)
    
    # Display configuration
    print(f"🚀 PDF Ingestion Pipeline")
    print(f"   Collection: {Config.QDRANT_COLLECTION_NAME}")
    print(f"   Qdrant URL: {Config.QDRANT_URL}")
    print(f"   Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"   Files to process: {len(pdf_files)}")
    
    # Create pipeline once (reuses embedder and client)
    pipeline = PDFIngestionPipeline()
    
    # Process all PDFs
    successful = 0
    failed = 0
    
    for idx, pdf_path in enumerate(pdf_files, start=1):
        if ingest_single_pdf(pipeline, pdf_path, idx, len(pdf_files)):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"   ✅ Successful: {successful}")
    if failed > 0:
        print(f"   ❌ Failed: {failed}")
    print(f"   📦 Total points in collection: {pipeline.client.count(Config.QDRANT_COLLECTION_NAME).count if pipeline.client else 'N/A'}")
    print(f"{'='*60}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

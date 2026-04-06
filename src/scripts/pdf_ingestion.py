"""PDF ingestion pipeline for Qdrant vector database with optional multimodal support."""

import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from qdrant_client import QdrantClient

from src.config import Config
from src.helpers.logs import get_logger
from src.helpers.qdrant_utils import (
    create_qdrant_client,
    create_qdrant_collection,
    insert_data_into_qdrant,
    get_collection_info,
    create_payload_index,
)
from src.models.embedder import get_embedder

logger = get_logger()


class PDFIngestionPipeline:
    """Pipeline for ingesting PDF documents into Qdrant vector database."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        index_fields: Optional[List[str]] = None,
        extract_images: bool = False,  # Disabled by default for speed
    ):
        """
        Initialize the PDF ingestion pipeline.

        Args:
            collection_name: Qdrant collection name (defaults to config)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            index_fields: Payload fields to create indexes for
            extract_images: Enable multimodal image extraction (slower, uses Unstructured)
        """
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images
        self.index_fields = index_fields or [
            "meta_data.company_name",
            "meta_data.quarter",
            "meta_data.year",
            "meta_data.document_type",
            "meta_data.content_type",  # "text" or "image"
        ]

        self.client: Optional[QdrantClient] = None

        logger.info(
            f"Initialized PDFIngestionPipeline for collection: {self.collection_name}"
        )

    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extract structured metadata from filename.

        Args:
            filename: PDF filename in one of these formats:
                - Earnings calls: 'NVDA-Q1-2026-Earnings-Call-...'
                - Presentations: 'NVDA-F1Q26-Quarterly-Presentation-...'

        Returns:
            Dictionary with quarter, year, and document_type
        """
        parts = filename.split("-")
        metadata: Dict[str, str] = {}

        # Check if it's a presentation format (e.g., F1Q26)
        fiscal_quarter_pattern = None
        for part in parts:
            if part.startswith("F") and "Q" in part and len(part) >= 4:
                fiscal_quarter_pattern = part
                break

        if fiscal_quarter_pattern:
            # Presentation format: NVDA-F1Q26-Quarterly-Presentation
            metadata["document_type"] = "presentation"
            try:
                # Extract quarter number (e.g., "1" from "F1Q26")
                q_index = fiscal_quarter_pattern.index("Q")
                quarter_num = fiscal_quarter_pattern[1:q_index]
                metadata["quarter"] = f"Q{quarter_num}"
                
                # Extract year (e.g., "26" from "F1Q26" -> "2026")
                year_short = fiscal_quarter_pattern[q_index + 1 :]
                metadata["year"] = f"20{year_short}"
            except (ValueError, IndexError):
                pass
        else:
            # Earnings call format: NVDA-Q1-2026-Earnings-Call
            metadata["document_type"] = "earnings_call"
            for part in parts:
                if part.startswith("Q") and len(part) == 2:
                    metadata["quarter"] = part
                elif part.isdigit() and len(part) == 4:
                    metadata["year"] = part

        return metadata

    def _load_pdf_with_images(self, pdf_path: Path) -> tuple[List[Any], List[Image.Image]]:
        """
        Load PDF and extract both text elements and images using Unstructured.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (text_elements, images)
        """
        logger.info(f"Loading PDF with Unstructured: {pdf_path}")
        
        # Use Unstructured with extract_images_in_pdf to get images
        loader = UnstructuredPDFLoader(
            str(pdf_path),
            mode="elements",  # Get individual elements (text, tables, images)
            extract_images_in_pdf=True,  # Extract images
            extract_image_block_types=["Image", "Table"],  # Extract images and tables as images
        )
        
        elements = loader.load()
        
        # Separate text elements and extract images
        text_elements = []
        images = []
        
        for element in elements:
            # Check if element has image data
            if hasattr(element, 'metadata') and 'image_base64' in element.metadata:
                # Decode base64 image
                try:
                    image_data = base64.b64decode(element.metadata['image_base64'])
                    image = Image.open(io.BytesIO(image_data))
                    images.append(image)
                    logger.debug(f"Extracted image: {image.size}")
                except Exception as e:
                    logger.warning(f"Failed to decode image: {e}")
            else:
                # Regular text element
                text_elements.append(element)
        
        logger.info(f"Loaded {len(text_elements)} text elements and {len(images)} images")
        return text_elements, images

    def _load_pdf_fast(self, pdf_path: Path) -> List[Any]:
        """
        Fast PDF loading using PyPDFLoader (text only, no images).

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of LangChain documents
        """
        logger.info(f"Loading PDF (fast mode): {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")
        return documents

    def _chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain documents

        Returns:
            List of document chunks
        """
        logger.info(
            f"Chunking documents (size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _format_chunks_for_qdrant(
        self, chunks: List[Any], pdf_path: Path, content_type: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Format document chunks for Qdrant insertion.

        Args:
            chunks: List of document chunks
            pdf_path: Path to source PDF
            content_type: Type of content ("text" or "image")

        Returns:
            List of formatted documents ready for Qdrant
        """
        logger.info(f"Formatting {len(chunks)} {content_type} chunks for Qdrant")
        file_metadata = self._extract_metadata_from_filename(pdf_path.stem)
        
        # Extract company name from parent directory structure
        # e.g., data/nvidia/earnings_call/file.pdf -> company_name = "nvidia"
        company_folder = pdf_path.parent.parent.name if pdf_path.parent.parent.name != "data" else pdf_path.parent.name
        
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted_chunks.append(
                {
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        **file_metadata,
                        "company_name": company_folder,
                        "filename": pdf_path.stem,
                        "content_type": content_type,
                        "chunk_id": i,
                    },
                    "name": f"{pdf_path.stem} - {content_type.capitalize()} {i}",
                }
            )

        logger.info(f"Formatted {len(formatted_chunks)} {content_type} documents")
        return formatted_chunks

    def _format_images_for_qdrant(
        self, images: List[Image.Image], pdf_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Format extracted images for Qdrant insertion with multimodal embeddings.

        Args:
            images: List of PIL Image objects
            pdf_path: Path to source PDF

        Returns:
            List of formatted image documents ready for Qdrant
        """
        logger.info(f"Formatting {len(images)} images for Qdrant")
        file_metadata = self._extract_metadata_from_filename(pdf_path.stem)
        
        # Extract company name from parent directory structure
        company_folder = pdf_path.parent.parent.name if pdf_path.parent.parent.name != "data" else pdf_path.parent.name
        
        formatted_images = []
        for i, image in enumerate(images):
            # Store image as base64 for potential retrieval
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            formatted_images.append(
                {
                    "image": image,  # PIL Image for embedding
                    "text": f"Image {i} from {pdf_path.stem}",  # Placeholder text
                    "metadata": {
                        **file_metadata,
                        "company_name": company_folder,
                        "filename": pdf_path.stem,
                        "content_type": "image",
                        "chunk_id": i,
                        "image_size": f"{image.width}x{image.height}",
                        "image_base64": img_base64,  # Store encoded image
                    },
                    "name": f"{pdf_path.stem} - Image {i}",
                }
            )

        logger.info(f"Formatted {len(formatted_images)} image documents")
        return formatted_images

    def _setup_qdrant_collection(self) -> None:
        """Setup Qdrant collection with embedder and indexes."""
        logger.info(f"Setting up Qdrant collection: {self.collection_name}")

        # Connect to Qdrant
        self.client = create_qdrant_client()
        logger.info("Connected to Qdrant")

        # Create collection with proper dimensions from embedder
        embedder_dimensions = get_embedder().dimensions
        create_qdrant_collection(self.client, embedder_dimensions, self.collection_name)
        logger.info(f"Collection ready with {embedder_dimensions} dimensions")

        # Create payload indexes
        if self.index_fields:
            logger.info("Creating payload indexes")
            for field in self.index_fields:
                try:
                    create_payload_index(self.client, field, self.collection_name)
                    logger.info(f"Created index for: {field}")
                except Exception as e:
                    logger.warning(f"Failed to create index for {field}: {e}")

    def ingest(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline with smart mode selection.
        
        Automatically uses:
        - Fast mode (PyPDF) for earnings calls (text only)
        - Multimodal mode (Unstructured) for presentations (if extract_images=True)

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting ingestion pipeline for: {pdf_path}")
        
        # Detect document type from filename
        file_metadata = self._extract_metadata_from_filename(pdf_path.stem)
        is_presentation = file_metadata.get("document_type") == "presentation"
        
        # Decide whether to extract images
        should_extract_images = self.extract_images and is_presentation
        
        if should_extract_images:
            logger.info("Using multimodal mode for presentation")
            # Step 1: Load PDF with text and images
            text_elements, images = self._load_pdf_with_images(pdf_path)
            # Step 2: Chunk text documents
            chunks = self._chunk_documents(text_elements)
        else:
            logger.info("Using fast mode (text only)")
            # Step 1: Load PDF fast (text only)
            documents = self._load_pdf_fast(pdf_path)
            # Step 2: Chunk documents
            chunks = self._chunk_documents(documents)
            images = []

        # Step 3: Format text chunks for Qdrant
        formatted_text_chunks = self._format_chunks_for_qdrant(chunks, pdf_path, content_type="text")

        # Step 4: Format images for Qdrant (if any)
        formatted_image_chunks = []
        if images:
            formatted_image_chunks = self._format_images_for_qdrant(images, pdf_path)

        # Step 5: Setup Qdrant (initializes embedder and creates collection)
        self._setup_qdrant_collection()
        assert self.client is not None, "Qdrant client not initialized"

        # Step 6: Insert text data
        total_chunks = len(formatted_text_chunks) + len(formatted_image_chunks)
        logger.info(f"Inserting {len(formatted_text_chunks)} text chunks into Qdrant")
        if formatted_text_chunks:
            insert_data_into_qdrant(formatted_text_chunks, self.client, self.collection_name)

        # Step 7: Insert image data (if any)
        if formatted_image_chunks:
            logger.info(f"Inserting {len(formatted_image_chunks)} image chunks into Qdrant")
            insert_data_into_qdrant(formatted_image_chunks, self.client, self.collection_name)
        
        logger.info("Insertion complete")

        # Step 8: Get final stats
        info = get_collection_info(self.client, self.collection_name)
        logger.info(f"Ingestion complete: {info['points_count']} points in collection")

        return {
            "pdf_path": str(pdf_path),
            "collection_name": self.collection_name,
            "text_chunks_ingested": len(formatted_text_chunks),
            "image_chunks_ingested": len(formatted_image_chunks),
            "total_chunks_ingested": total_chunks,
            "chunks_ingested": total_chunks,  # Backward compatibility
            "collection_info": info,
        }

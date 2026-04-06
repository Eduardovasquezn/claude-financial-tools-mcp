# Financial Agent - Common Tasks

# Show all available commands
default:
    @just --list

# === PDF Ingestion ===

# Ingest a single PDF file
ingest-file FILE:
    uv run python scripts/ingest_pdf.py {{FILE}}

# Ingest all PDFs from a folder
ingest-folder FOLDER:
    uv run python scripts/ingest_pdf.py {{FOLDER}}

# Ingest all NVIDIA documents (earnings calls + presentations)
ingest-nvidia:
    uv run python scripts/ingest_pdf.py data/nvidia/

# Ingest only NVIDIA earnings calls
ingest-nvidia-earnings:
    uv run python scripts/ingest_pdf.py data/nvidia/earnings_call/

# Ingest only NVIDIA presentations
ingest-nvidia-presentations:
    uv run python scripts/ingest_pdf.py data/nvidia/presentations/

# === Collection Management ===

# Delete the Qdrant collection
delete-collection:
    uv run python scripts/delete_collection.py
 
# List available Qdrant tools
list-tools:
    uv run python examples/list_qdrant_tools.py

# Test Qdrant tools interactively
test-qdrant:
    uv run python examples/test_qdrant_tools.py
 
# Run linter checks
lint:
    uv run ruff check src/

# Format code
format:
    uv run ruff format src/

# Type check with mypy
typecheck:
    uv run mypy src/

# Run all checks (lint + format + typecheck)
check: lint format typecheck

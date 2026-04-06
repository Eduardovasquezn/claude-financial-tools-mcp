"""LangChain tools for Qdrant operations - wrapping existing helper functions."""

from typing import Optional, Dict, Any, List, cast
from langchain_core.tools import tool

from src.config import Config
from src.helpers.logs import get_logger
from src.helpers.qdrant_utils import (
    create_qdrant_client,
    search_in_qdrant,
    get_collection_info,
)
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = get_logger()


@tool
def search_documents_semantic(
    query: str,
    company_name: Optional[str] = None,
    quarter: Optional[str] = None,
    year: Optional[str] = None,
    document_type: Optional[str] = None,
    collection_name: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Semantic search for documents using vector similarity with optional metadata filtering.

    Use this when you need to find documents based on meaning/concepts.
    Can optionally filter to specific company/quarter/year/document_type for more precise results.

    Args:
        query: Natural language search query
        company_name: Optional company name to filter by (e.g., "nvidia", "apple")
        quarter: Optional quarter to filter by (e.g., "Q1", "Q2", "Q3", "Q4")
        year: Optional year to filter by (e.g., "2026", "2025")
        document_type: Optional document type ("earnings_call" or "presentation")
        collection_name: Collection to search (defaults to config)
        limit: Maximum results

    Returns:
        List of relevant documents with content and metadata

    Examples:
        search_documents_semantic("What is NVIDIA's revenue growth strategy?")
        search_documents_semantic("revenue guidance", company_name="nvidia", quarter="Q3", year="2025")
        search_documents_semantic("data center growth", company_name="nvidia", document_type="presentation")
    """
    try:
        collection_name = collection_name or Config.QDRANT_COLLECTION_NAME

        client = create_qdrant_client()

        # Build metadata filters if provided
        metadata_filters = {}
        if company_name:
            metadata_filters["meta_data.company_name"] = company_name.lower()
        if quarter:
            metadata_filters["meta_data.quarter"] = quarter
        if year:
            metadata_filters["meta_data.year"] = str(year)
        if document_type:
            metadata_filters["meta_data.document_type"] = document_type

        results = search_in_qdrant(
            qdrant_client=client,
            input_question=query,
            k=limit,
            collection_name=collection_name,
            metadata_filters=metadata_filters if metadata_filters else None,
        )

        formatted_results = []
        for point in results.points:
            metadata = point.payload.get("meta_data", {})
            formatted_results.append(
                {
                    "content": point.payload.get("content", ""),
                    "company_name": metadata.get("company_name", ""),
                    "quarter": metadata.get("quarter", ""),
                    "year": metadata.get("year", ""),
                    "document_type": metadata.get("document_type", ""),
                    "filename": metadata.get("filename", ""),
                    "score": point.score,
                }
            )

        filter_info = f" with filters: {metadata_filters}" if metadata_filters else ""
        logger.info(
            f"Semantic search found {len(formatted_results)} documents{filter_info}"
        )
        return formatted_results

    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return [{"error": str(e)}]


@tool
def search_by_metadata(
    company_name: Optional[str] = None,
    quarter: Optional[str] = None,
    year: Optional[str] = None,
    document_type: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search documents by metadata filters only (no semantic search).

    Use this when you need ALL documents matching specific criteria,
    like getting all Q1 2026 documents regardless of content.

    Args:
        company_name: Company name (e.g., "nvidia", "apple")
        quarter: Quarter (e.g., "Q1", "Q2", "Q3", "Q4")
        year: Year (e.g., "2026")
        document_type: Document type ("earnings_call" or "presentation")
        limit: Maximum results to return

    Returns:
        List of documents matching the metadata filters

    Example:
        search_by_metadata(company_name="nvidia", quarter="Q1", year="2026")
        search_by_metadata(company_name="nvidia", document_type="presentation")
    """
    try:
        if not any([company_name, quarter, year, document_type]):
            return [
                {"error": "At least one filter (company_name, quarter, year, or document_type) is required"}
            ]

        client = create_qdrant_client()

        # Build filter conditions
        conditions = []
        if company_name:
            conditions.append(
                FieldCondition(
                    key="meta_data.company_name", match=MatchValue(value=company_name.lower())
                )
            )
        if quarter:
            conditions.append(
                FieldCondition(key="meta_data.quarter", match=MatchValue(value=quarter))
            )
        if year:
            conditions.append(
                FieldCondition(key="meta_data.year", match=MatchValue(value=str(year)))
            )
        if document_type:
            conditions.append(
                FieldCondition(key="meta_data.document_type", match=MatchValue(value=document_type))
            )

        # Scroll through collection with filters (no vector search)
        results = client.scroll(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(must=cast(Any, conditions)),
            limit=limit,
        )

        formatted_results = []
        for point in results[0]:  # scroll returns (points, next_page_offset)
            payload = cast(Dict[str, Any], point.payload or {})
            metadata = cast(Dict[str, Any], payload.get("meta_data") or {})
            formatted_results.append(
                {
                    "content": payload.get("content", ""),
                    "company_name": metadata.get("company_name", ""),
                    "quarter": metadata.get("quarter", ""),
                    "year": metadata.get("year", ""),
                    "document_type": metadata.get("document_type", ""),
                    "filename": metadata.get("filename", ""),
                    "chunk_id": metadata.get("chunk_id", 0),
                }
            )

        logger.info(f"Metadata search found {len(formatted_results)} documents")
        return formatted_results

    except Exception as e:
        logger.error(f"Error in metadata search: {e}")
        return [{"error": str(e)}]


@tool
def compare_quarters(
    company_name: str,
    quarters: List[str],
    year: str,
    document_type: Optional[str] = None,
    query: Optional[str] = None,
    limit_per_quarter: int = 5,
) -> Dict[str, Any]:
    """
    Compare information across multiple quarters for a specific company.

    Use this when you need to compare Q1 vs Q2, or any quarters.
    Returns documents grouped by quarter for easy comparison.

    Args:
        company_name: Company name (e.g., "nvidia")
        quarters: List of quarters to compare (e.g., ["Q1", "Q2"])
        year: Year (e.g., "2026")
        document_type: Optional document type to filter ("earnings_call" or "presentation")
        query: Optional semantic search query to filter relevant content
        limit_per_quarter: Max documents per quarter

    Returns:
        Dictionary with results grouped by quarter

    Example:
        compare_quarters(
            company_name="nvidia",
            quarters=["Q1", "Q2"],
            year="2026",
            document_type="earnings_call",
            query="revenue and guidance"
        )
    """
    try:
        client = create_qdrant_client()

        results_by_quarter = {}

        for quarter in quarters:
            # Build metadata filter
            filters = {
                "meta_data.company_name": company_name.lower(),
                "meta_data.quarter": quarter,
                "meta_data.year": str(year),
            }
            if document_type:
                filters["meta_data.document_type"] = document_type

            if query:
                # Semantic search with metadata filter
                results = search_in_qdrant(
                    qdrant_client=client,
                    input_question=query,
                    collection_name=Config.QDRANT_COLLECTION_NAME,
                    k=limit_per_quarter,
                    metadata_filters=filters,
                )

                quarter_docs = []
                for point in results.points:
                    metadata = point.payload.get("meta_data", {})
                    quarter_docs.append(
                        {
                            "content": point.payload.get("content", ""),
                            "score": point.score,
                            "company_name": metadata.get("company_name", ""),
                            "quarter": metadata.get("quarter", ""),
                            "year": metadata.get("year", ""),
                            "document_type": metadata.get("document_type", ""),
                            "filename": metadata.get("filename", ""),
                            "chunk_id": metadata.get("chunk_id", 0),
                        }
                    )
            else:
                # Pure metadata search
                conditions = [
                    FieldCondition(
                        key="meta_data.company_name", match=MatchValue(value=company_name.lower())
                    ),
                    FieldCondition(
                        key="meta_data.quarter", match=MatchValue(value=quarter)
                    ),
                    FieldCondition(
                        key="meta_data.year", match=MatchValue(value=str(year))
                    ),
                ]
                if document_type:
                    conditions.append(
                        FieldCondition(
                            key="meta_data.document_type", match=MatchValue(value=document_type)
                        )
                    )

                results = client.scroll(
                    collection_name=Config.QDRANT_COLLECTION_NAME,
                    scroll_filter=Filter(must=cast(Any, conditions)),
                    limit=limit_per_quarter,
                )

                quarter_docs = []
                for point in results[0]:
                    payload = cast(Dict[str, Any], point.payload or {})
                    meta = cast(Dict[str, Any], payload.get("meta_data") or {})
                    quarter_docs.append(
                        {
                            "content": payload.get("content", ""),
                            "company_name": meta.get("company_name", ""),
                            "quarter": meta.get("quarter", ""),
                            "year": meta.get("year", ""),
                            "document_type": meta.get("document_type", ""),
                            "filename": meta.get("filename", ""),
                            "chunk_id": meta.get("chunk_id", 0),
                        }
                    )

            results_by_quarter[quarter] = quarter_docs

        logger.info(f"Comparison retrieved data for {len(quarters)} quarters")
        return {
            "company_name": company_name,
            "year": year,
            "quarters": results_by_quarter,
            "total_documents": sum(len(docs) for docs in results_by_quarter.values()),
        }

    except Exception as e:
        logger.error(f"Error comparing quarters: {e}")
        return {"error": str(e)}


@tool
def get_available_metadata(
    company_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get all available metadata values in the collection.

    Use this to discover what companies, quarters, and years are available
    before making queries.

    Args:
        company_name: Optional company name to get metadata for specific company

    Returns:
        Dictionary with available companies, quarters, and years

    Example:
        get_available_metadata()  # All metadata
        get_available_metadata(company_name="nvidia")  # nvidia-specific
    """
    try:
        client = create_qdrant_client()

        # Scroll through all documents to collect unique metadata values
        scroll_filter = None
        if company_name:
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="meta_data.company_name", match=MatchValue(value=company_name.lower())
                    )
                ]
            )

        results = client.scroll(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=1000,  # Adjust based on your collection size
        )

        companies = set()
        quarters = set()
        years = set()
        document_types = set()

        for point in results[0]:
            payload = cast(Dict[str, Any], point.payload or {})
            meta = cast(Dict[str, Any], payload.get("meta_data") or {})
            if meta.get("company_name"):
                companies.add(meta["company_name"])
            if meta.get("quarter"):
                quarters.add(meta["quarter"])
            if meta.get("year"):
                years.add(meta["year"])
            if meta.get("document_type"):
                document_types.add(meta["document_type"])

        return {
            "companies": sorted(list(companies)),
            "quarters": sorted(list(quarters)),
            "years": sorted(list(years)),
            "document_types": sorted(list(document_types)),
            "total_documents": len(results[0]),
        }

    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return {"error": str(e)}


@tool
def get_qdrant_collection_stats(
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get statistics about a Qdrant collection.

    Use this to check how many documents are in the database or verify collection status.

    Args:
        collection_name: Name of the collection (defaults to config)

    Returns:
        Dictionary with collection statistics

    Example:
        get_qdrant_collection_stats()
    """
    try:
        collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        client = create_qdrant_client()

        info = get_collection_info(client, collection_name)
        logger.info(f"Retrieved stats for collection: {collection_name}")
        return info

    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {"error": str(e)}


# Export all tools as a list for easy integration
QDRANT_TOOLS = [
    search_documents_semantic,
    search_by_metadata,
    compare_quarters,
    get_available_metadata,
    get_qdrant_collection_stats,
]


def get_qdrant_tools() -> List:
    """
    Get all Qdrant tools for use with LangChain agents.

    Returns:
        List of LangChain tools

    Tools included:
    - search_documents_semantic: Vector similarity search
    - search_by_metadata: Filter by ticker/quarter/year
    - compare_quarters: Compare data across multiple quarters
    - get_available_metadata: Discover available metadata values
    - get_qdrant_collection_stats: Collection statistics

    Example:
        ```python
        from src.tools.qdrant_tools import get_qdrant_tools

        tools = get_qdrant_tools()
        agent = create_agent(llm, tools)
        ```
    """
    return QDRANT_TOOLS

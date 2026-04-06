"""Qdrant vector database utilities for storing and retrieving embeddings."""

import uuid
from typing import Any, Dict, List, Optional, cast

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.config import Config
from src.helpers.logs import get_logger
from src.models.embedder import get_embedder

logger = get_logger()


def create_qdrant_client() -> QdrantClient:
    """
    Creates a connection to the Qdrant client using Config values.

    Returns:
        QdrantClient: An instance of the QdrantClient if the connection is successful.
    """
    try:
        logger.info("Starting Qdrant connection")
        client = QdrantClient(
            url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY or None
        )
        logger.info("Successfully connected to Qdrant")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        raise


def create_qdrant_collection(
    qdrant_client: QdrantClient,
    vector_dimensions: int,
    collection_name: str = Config.QDRANT_COLLECTION_NAME,
) -> None:
    """
    Creates a Qdrant collection if it does not already exist.

    Args:
        qdrant_client: The Qdrant client instance to interact with
        vector_dimensions: The dimensionality of vectors to store
        collection_name: The name of the collection to create (defaults to Config.QDRANT_COLLECTION_NAME)

    Raises:
        ValueError: If an error occurs while creating the collection
    """
    try:
        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dimensions, distance=Distance.COSINE
                ),
            )
            logger.info(
                f"Collection '{collection_name}' has been successfully created."
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {str(e)}")
        raise


def prepare_points(data: List[Dict[str, Any]]) -> List[PointStruct]:
    """
    Prepares points for upsert into Qdrant with batched text embeddings and image support.

    Args:
        data: List of dictionaries containing:
            - 'text' (required for text items)
            - 'image' (required for image items, PIL Image object)
            - 'metadata' (optional)
            - 'name' (optional)

    Returns:
        List of PointStruct objects for insertion into Qdrant
    """
    try:
        logger.info("Starting to prepare points for Qdrant...")
        embedder = get_embedder()
        points = []
        
        # Separate text and image items for batch processing
        text_items = []
        image_items = []
        
        for item in data:
            if "image" in item:
                image_items.append(item)
            else:
                text_content = item.get("text", item.get("content", ""))
                if not text_content:
                    logger.warning(
                        f"Skipping item with no text content: {item.get('name', 'unknown')}"
                    )
                    continue
                text_items.append(item)
        
        # Batch embed all text items - chunk into reasonable batch sizes
        if text_items:
            batch_size = 100  # Process 100 texts per API call to avoid limits
            logger.info(f"Batch embedding {len(text_items)} text items in batches of {batch_size}...")
            
            for i in range(0, len(text_items), batch_size):
                batch = text_items[i:i + batch_size]
                texts = [item.get("text", item.get("content", "")) for item in batch]
                
                logger.debug(f"Embedding batch {i//batch_size + 1}/{(len(text_items) + batch_size - 1)//batch_size}")
                vectors = embedder.embed(texts)  # Single API call per batch
                
                # Create points from batched embeddings
                for item, vector in zip(batch, vectors):
                    text_content = item.get("text", item.get("content", ""))
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "meta_data": item.get("metadata", {}),
                            "content": text_content,
                            "name": item.get("name", ""),
                            "usage": item.get("usage"),
                        },
                    )
                    points.append(point)
        
        # Process images individually (can't batch different content types)
        if image_items:
            logger.info(f"Embedding {len(image_items)} images...")
            for item in image_items:
                image = item["image"]
                try:
                    vector = embedder.embed_image(image)
                    text_content = item.get("text", f"Image from {item.get('name', 'unknown')}")
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "meta_data": item.get("metadata", {}),
                            "content": text_content,
                            "name": item.get("name", ""),
                            "usage": item.get("usage"),
                        },
                    )
                    points.append(point)
                except Exception as e:
                    logger.warning(f"Failed to embed image: {e}. Skipping.")
                    continue
            
        logger.info(f"Finished preparing {len(points)} points for Qdrant...")
        return points
    except Exception as e:
        logger.error(f"Error preparing points: {str(e)}")
        raise


def insert_data_into_qdrant(
    data: List[Dict[str, Any]],
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """
    Embeds text and inserts metadata into Qdrant.

    Args:
        data: List of dictionaries containing 'text' and 'metadata'
        qdrant_client: The Qdrant client instance to use for insertion
        collection_name: The name of the Qdrant collection where the data will be stored

    Returns:
        None
    """
    try:
        logger.info("Starting prepare_points...")
        points = prepare_points(data)
        logger.info(f"Prepared {len(points)} points for insertion into Qdrant.")

        qdrant_client.upsert(collection_name=collection_name, points=points)
        logger.info(
            f"Successfully inserted {len(points)} points into '{collection_name}' collection."
        )
    except Exception as e:
        logger.error(
            f"Error inserting data into Qdrant collection '{collection_name}': {str(e)}"
        )
        raise


def create_payload_index(
    qdrant_client: QdrantClient,
    field_name: str,
    collection_name: str,
    field_schema: PayloadSchemaType = PayloadSchemaType.KEYWORD,
) -> None:
    """
    Creates a payload index for a given collection and field.

    Args:
        qdrant_client: The client connection object to interact with the database
        field_name: The name of the field to index
        collection_name: The name of the collection
        field_schema: The schema type for the field
    """
    try:
        logger.info(
            f"Creating payload index for collection: {collection_name}, field: {field_name}"
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        logger.info(f"Successfully created payload index for field: {field_name}")
    except Exception as e:
        logger.error(
            f"Error creating payload index for collection: {collection_name}, "
            f"field: {field_name}: {e}",
            exc_info=True,
        )
        raise


def search_in_qdrant(
    qdrant_client: QdrantClient,
    input_question: str,
    k: int = 5,
    collection_name: str = Config.QDRANT_COLLECTION_NAME,
    metadata_filters: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None,
) -> Any:
    """
    Searches for similar vectors in Qdrant, with optional metadata filtering.

    Args:
        qdrant_client: The Qdrant client instance to use for searching
        input_question: The text query to search for
        k: The maximum number of results to return
        collection_name: The name of the Qdrant collection to search in (defaults to Config.QDRANT_COLLECTION_NAME)
        metadata_filters: Key-value pairs to filter results based on metadata
        score_threshold: Minimum similarity score to return

    Returns:
        Search results from Qdrant
    """
    try:
        query_vector = get_embedder().embed(input_question)
        query_filter = None

        # If metadata_filters is provided, construct the Qdrant filter
        if metadata_filters:
            filter_conditions: List[FieldCondition] = []
            for key, value in metadata_filters.items():
                # Extract the first value if the input is a list
                single_value = value[0] if isinstance(value, list) and value else value

                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=single_value))
                )

            query_filter = Filter(must=cast(Any, filter_conditions))

        # Perform the query with or without filters
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        logger.info(
            f"Found {len(results.points)} results for query: '{input_question[:50]}...'"
        )
        return results
    except Exception as e:
        logger.error(
            f"Error searching in Qdrant collection '{collection_name}': {str(e)}"
        )
        raise


def get_collection_info(
    qdrant_client: QdrantClient, collection_name: str
) -> Dict[str, Any]:
    """
    Get information about a Qdrant collection.

    Args:
        qdrant_client: The Qdrant client instance
        collection_name: The name of the collection

    Returns:
        Dictionary with collection information
    """
    try:
        info = qdrant_client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "status": info.status,
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise

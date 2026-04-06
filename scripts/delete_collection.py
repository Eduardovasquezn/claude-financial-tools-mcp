"""Delete Qdrant collection to recreate with correct configuration."""

from src.config import Config
from src.helpers.qdrant_utils import create_qdrant_client
from src.helpers.logs import get_logger

logger = get_logger()


def main():
    collection_name = Config.QDRANT_COLLECTION_NAME
    
    print(f"🗑️  Deleting collection: {collection_name}")
    
    # Connect to Qdrant
    client = create_qdrant_client()
    
    # Check if collection exists
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"✓ Collection '{collection_name}' deleted successfully!")
    else:
        print(f"⚠️  Collection '{collection_name}' does not exist.")


if __name__ == "__main__":
    main()

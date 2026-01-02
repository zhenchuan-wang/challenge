from typing import List, Dict, Any
import os
import chromadb
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import json

logger = logging.getLogger(__name__)


class SupportVectorStore:
    """
    A class to manage the vector store for support tickets using ChromaDB.

    This class handles the creation, storage, and retrieval of vector embeddings
    for technical, product, and customer support tickets in separate collections.

    IMPORTANT:
    - Empty queries (null or whitespace-only) must be rejected with an empty result list
    - Queries shorter than 10 characters must be rejected with an empty result list
    - All metadata must be properly processed for ChromaDB compatibility
    - Embedding model to be used should be OpenAI text-embedding-ada-002
    """

    def __init__(self, vecstore_path):
        """Initialize the vector store with ChromaDB client and OpenAI embeddings."""
        self.vecstore_path = vecstore_path
        os.makedirs(vecstore_path, exist_ok=True)
        settings = chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=vecstore_path,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.collections = {}

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB by converting lists to strings and ensuring valid types.

        ChromaDB requires all metadata values to be primitive types (str, int, float, bool).
        Lists must be converted to comma-separated strings, and None values must be handled appropriately.

        Args:
            metadata (Dict[str, Any]): Original metadata dictionary

        Returns:
            Dict[str, Any]: Processed metadata with ChromaDB-compatible types
        """
        processed = {}
        for key, value in metadata.items():
            if value is None:
                processed[key] = ""
            elif isinstance(value, list):
                processed[key] = ",".join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)):
                processed[key] = value
            else:
                # Convert complex objects to string
                processed[key] = str(value)
        return processed

    def _process_metadata_for_return(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata when retrieving from ChromaDB, converting string-lists back to actual lists.

        This function reverses the transformations done in _prepare_metadata() to ensure
        that metadata is returned in the expected format.

        Args:
            metadata (Dict[str, Any]): Metadata from ChromaDB

        Returns:
            Dict[str, Any]: Processed metadata with proper types
        """
        processed = {}
        for key, value in metadata.items():
            if key == "tags" and isinstance(value, str) and value:
                # Convert comma-separated tags back to list
                processed[key] = [
                    tag.strip() for tag in value.split(",") if tag.strip()
                ]
            elif key in ["priority"] and value != "":
                # Try to convert back to int if it's a number
                try:
                    processed[key] = int(value)
                except ValueError:
                    processed[key] = value
            elif key in ["is_resolved"] and isinstance(value, str):
                # Convert string boolean back
                processed[key] = value.lower() == "true"
            else:
                processed[key] = value
        return processed

    def create_vector_store(self, documents_by_type: Dict[str, List[Document]]) -> None:
        """
        Create vector store collections from documents, organized by support type.

        Args:
            documents_by_type (Dict[str, List[Document]]): Dictionary of documents organized by support type
        """
        for support_type, documents in documents_by_type.items():
            # Create or get collection
            collection = self.client.get_or_create_collection(
                name=support_type,
                embedding_function=None,  # We'll handle embeddings manually
            )
            self.collections[support_type] = collection

            # Prepare data for batch addition
            ids = []
            contents = []
            metadatas = []

            for doc in documents:
                # Use ticket_id as unique ID, or generate one
                doc_id = doc.metadata.get("ticket_id", f"{support_type}_{len(ids)}")
                ids.append(doc_id)
                contents.append(doc.page_content)
                metadatas.append(self._prepare_metadata(doc.metadata))

            if contents:
                # Generate embeddings
                embeddings = self.embeddings.embed_documents(contents)

                # Add to collection
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=contents,
                )

        # Persist the changes
        try:
            self.client.persist()
        except Exception:
            pass  # Some versions may not have persist method

    @classmethod
    def load_local(cls, directory: str) -> "SupportVectorStore":
        """
        Load a vector store from local storage.

        Args:
            directory (str): Directory path containing the vector store

        Returns:
            SupportVectorStore: Loaded vector store instance
        """
        instance = cls(directory)
        # Load existing collections into instance.collections
        try:
            collection_names = instance.client.list_collections()
            for collection_name in collection_names:
                collection = instance.client.get_collection(collection_name.name)
                instance.collections[collection_name.name] = collection
        except Exception as e:
            logger.warning(f"Error loading collections: {e}")
        return instance

    def query_similar(
        self, query: str, support_type: str = None, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.

        IMPORTANT:
        - Empty queries (null or whitespace-only) MUST return an empty list
        - When the query is null or whitespace-only, log a warning but DO NOT raise an exception
        - Non-existent support types MUST return an empty list with an appropriate warning

        Args:
            query (str): Query text to find similar documents
            support_type (str, optional): Specific support type to query. If None, queries all types
            k (int): Number of similar documents to return per collection

        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata, each containing:
            - 'content': Document content
            - 'metadata': Document metadata
            - 'similarity': Similarity score (1 - distance)
        """
        # Validate query
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        if len(query.strip()) < 10:
            logger.warning("Query too short (less than 10 characters)")
            return []

        # Determine which collections to query
        if support_type:
            if support_type not in self.collections:
                logger.warning(
                    f"Support type '{support_type}' not found in vector store"
                )
                return []
            collections_to_query = {support_type: self.collections[support_type]}
        else:
            collections_to_query = self.collections

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        results = []
        for coll_name, collection in collections_to_query.items():
            try:
                # Query the collection
                response = collection.query(
                    query_embeddings=[query_embedding], n_results=k
                )

                # Process results
                for i in range(len(response["ids"][0])):
                    result = {
                        "content": response["documents"][0][i],
                        "metadata": self._process_metadata_for_return(
                            response["metadatas"][0][i]
                        ),
                        "similarity": (
                            1.0 - response["distances"][0][i]
                            if response["distances"]
                            else 0.0
                        ),
                    }
                    results.append(result)

            except Exception as e:
                logger.warning(f"Error querying collection '{coll_name}': {e}")

        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Return top k results
        return results[:k]

    def get_support_types(self) -> List[str]:
        """
        Get list of available support types in the vector store.

        Returns:
            List[str]: List of support type names
        """
        return list(self.collections.keys())

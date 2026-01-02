from typing import List, Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
import logging
import jq
from uuid import uuid4
from functools import partial
from collections import defaultdict

logger = logging.getLogger(__name__)


class SupportDocumentLoader:
    """
    A class to load and process support tickets from JSON and XML files using LangChain loaders.

    This loader uses LangChain's JSONLoader and custom XML loading to process support tickets and
    converts them into a standardized document format for the RAG system.

    IMPORTANT:
    - Even when using LangChain loaders, you MUST use the custom get_json_content and
      get_json_metadata functions to ensure consistent document formatting
    - Ensure all ticket IDs are unique across the entire dataset
    - The format of ticket IDs must follow the pattern: "{support_type}_{original_id}" for JSON
      and "{support_type}_xml_{original_id}" for XML files
    """

    def __init__(self, data_path: str):
        """
        Initialize the document loader with the path to data files.

        Args:
            data_path (str): Directory path containing support ticket files

        Raises:
            FileNotFoundError: If the specified data path does not exist
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists() or not self.data_path.is_dir():
            raise FileNotFoundError(
                f"Data path {data_path} does not exist or is not a directory"
            )

    def get_json_content(self, data: Dict[str, Any]) -> str:
        """
        Format JSON data into a standardized content string.

        This function MUST produce content in the exact format shown below to ensure
        consistent document formatting across the system.

        Args:
            data (Dict[str, Any]): Raw JSON data

        Returns:
            str: Formatted content string with the exact format:

            Subject: {}
            Description: {}
            Resolution: {}
            Type: {}
            Queue: {}
            Priority: {}
        """
        return (
            f"Subject: {data.get('subject', '')}\n"
            f"Description: {data.get('body', '')}\n"
            f"Resolution: {data.get('answer', '')}\n"
            f"Type: {data.get('type', '')}\n"
            f"Queue: {data.get('queue', '')}\n"
            f"Priority: {data.get('priority', '')}"
        )

    def get_json_metadata(
        self, record: Dict[str, Any], support_type: str = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from JSON data.

        This function MUST produce metadata with all the required fields shown below.
        The 'ticket_id' MUST follow the format "{support_type}_{original_id}" to ensure
        proper document identification.

        Args:
            record (Dict[str, Any]): Raw JSON record
            support_type (str, optional): Type of support (technical, product, customer)

        Returns:
            Dict[str, Any]: Extracted metadata with the exact format:
            {
                'ticket_id': "{support_type}_{original_id}",  # Unique ID
                'original_ticket_id': str,    # Original ticket ID from JSON ("Ticket ID" field)
                'support_type': str,          # Type of support (technical, product, customer)
                'type': str,                  # Type field from original data
                'queue': str,                 # Queue information
                'priority': str,              # Priority level
                'language': str,              # Content language
                'tags': List[str],            # List of tags from tag_1 through tag_8
                'source': 'json',             # Source format identifier
                'subject': str,               # Subject field for content formatting
                'body': str,                  # Body field for content formatting
                'answer': str                 # Answer field for content formatting
            }

        Raises:
            ValueError: If support_type is not provided
        """
        if support_type is None:
            raise ValueError("support_type must be provided")

        tags = [
            record.get(f"tag_{i}")
            for i in range(1, 9)
            if record.get(f"tag_{i}") and str(record.get(f"tag_{i}")).lower() != "nan"
        ]

        return {
            "ticket_id": f"{support_type}_{record['Ticket ID']}",
            "original_ticket_id": str(record["Ticket ID"]),
            "support_type": support_type,
            "type": record.get("type", ""),
            "queue": record.get("queue", ""),
            "priority": record.get("priority", ""),
            "language": record.get("language", ""),
            "tags": tags,
            "source": "json",
            "subject": record.get("subject", ""),
            "body": record.get("body", ""),
            "answer": record.get("answer", ""),
        }

    def load_xml_tickets(self, file_path: Path, support_type: str) -> List[Document]:
        """
        Load tickets from an XML file.

        XML tickets MUST be processed to follow the same content and metadata format
        as JSON tickets, with the only difference being the 'ticket_id' format and
        'source' field.

        Args:
            file_path (Path): Path to the XML file
            support_type (str): Type of support (technical, product, customer)

        Returns:
            List[Document]: List of Document objects with the following format:

            Content format:
            Subject: {}
            Description: {}
            Resolution: {}
            Type: {}
            Queue: {}
            Priority: {}

            Metadata format:
            {
                'ticket_id': "{support_type}_xml_{original_id}",  # Unique ID
                'original_ticket_id': str,    # Original ticket ID from XML
                'support_type': str,          # Type of support
                'type': str,                  # Type field
                'queue': str,                 # Queue information
                'priority': str,              # Priority level
                'language': str,              # Content language
                'tags': List[str],            # List of tags
                'source': 'xml'               # Source format identifier
            }
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        documents = []

        for ticket in root.findall("Ticket"):
            data = {child.tag: child.text for child in ticket}
            content = self.get_json_content(data)

            tags = [
                data.get(f"tag_{i}")
                for i in range(1, 9)
                if data.get(f"tag_{i}") and str(data.get(f"tag_{i}")).lower() != "nan"
            ]

            metadata = {
                "ticket_id": f"{support_type}_xml_{data['TicketID']}",
                "original_ticket_id": str(data["TicketID"]),
                "support_type": support_type,
                "type": data.get("type", ""),
                "queue": data.get("queue", ""),
                "priority": data.get("priority", ""),
                "language": data.get("language", ""),
                "tags": tags,
                "source": "xml",
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def _normalize_support_type(self, support_type_raw: str) -> str:
        """
        Normalize support type from file name to standardized key.

        Args:
            support_type_raw (str): Raw support type from file name

        Returns:
            str: Normalized support type
        """
        if "Technical" in support_type_raw:
            return "technical"
        elif "Product" in support_type_raw:
            return "product"
        elif "Customer" in support_type_raw:
            return "customer"
        else:
            return support_type_raw.lower().replace(" ", "")

    def load_tickets(self) -> Dict[str, List[Document]]:
        """
        Load all support tickets using LangChain loaders, organized by support type.

        IMPORTANT:
        - When using JSONLoader, you MUST create a custom function that properly passes
          the support_type parameter to get_json_metadata
        - Validate that all ticket IDs are unique across the entire dataset

        Returns:
            Dict[str, List[Document]]: Dictionary with support types as keys and lists of Documents as values

        Raises:
            ValueError: If duplicate ticket IDs are found
        """
        documents = defaultdict(list)

        # Load JSON files
        for json_file in self.data_path.glob("*_tickets.json"):
            support_type_raw = json_file.stem.replace("_tickets", "")
            support_type = self._normalize_support_type(support_type_raw)

            try:
                import json

                with open(json_file, "r") as f:
                    data = json.load(f)
                for record in data:
                    content = self.get_json_content(record)
                    metadata = self.get_json_metadata(record, support_type)
                    doc = Document(page_content=content, metadata=metadata)
                    documents[support_type].append(doc)
            except Exception:
                # If JSON loading fails, ensure the key exists with empty list
                documents[
                    support_type
                ]  # This creates the key with empty list due to defaultdict

        # Load XML files
        for xml_file in self.data_path.glob("*_tickets.xml"):
            support_type_raw = xml_file.stem.replace("_tickets", "")
            support_type = self._normalize_support_type(support_type_raw)

            try:
                docs = self.load_xml_tickets(xml_file, support_type)
                documents[support_type].extend(docs)
            except Exception:
                # If XML loading fails, skip this file
                pass

        # Validate unique ticket IDs
        all_ticket_ids = [
            doc.metadata["ticket_id"]
            for docs_list in documents.values()
            for doc in docs_list
        ]
        if len(all_ticket_ids) != len(set(all_ticket_ids)):
            raise ValueError("Duplicate ticket IDs found")

        return dict(documents)

    def create_documents(self) -> Dict[str, List[Document]]:
        """
        Load and process all support tickets into LangChain Document objects.

        Returns:
            Dict[str, List[Document]]: Dictionary with support types as keys and lists of Document objects as values
        """
        return self.load_tickets()

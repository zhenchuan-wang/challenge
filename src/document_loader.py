from typing import List, Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
import logging
import jq
from uuid import uuid4
from functools import partial

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
        self.data_path = data_path

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
        str_subject = data.get("subject")
        str_description = data.get("description")
        str_resolution = data.get("resolution")
        str_type = data.get("type")
        str_queue = data.get("queue")
        str_priority = data.get("priority")
        result = {}
        result["Subject"] = str_subject
        result["Description"] = str_description
        result["Resolution"] = str_resolution
        result["Type"] = str_type
        result["Queue"] = str_queue
        result["Priority"] = str_priority
        return str(result)

    def get_json_metadata(self, record: Dict[str, Any], support_type: str = None) -> Dict[str, Any]:
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
            raise ValueError
        metadata = {}
        metadata["ticket_id"] = '_'.join([support_type, record.get(original_id)])
        metadata["original_ticket_id"] = record.get("Ticket ID")
        metadata["support_type"] = support_type
        metadata["type"] = record.get("type")
        metadata["queue"] = record.get("queue")
        metadata["priority"] = record.get("priority")
        metadata["language"] = record.get("language")
        metadata["tags"] = record.get("tags")
        metadata["source"] = record.get("source")
        metadata["subject"] = record.get("source")
        metadata["body"] = record.get("source")
        metadata["answer"] = record.get("answer")
        return metadata
        


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
        if support_type is None:
            raise ValueError
        metadata = {}
        metadata["ticket_id"] = '_xml_'.join([support_type, record.get(original_id)])
        metadata["original_ticket_id"] = record.get("Ticket ID")
        metadata["support_type"] = support_type
        metadata["type"] = record.get("type")
        metadata["queue"] = record.get("queue")
        metadata["priority"] = record.get("priority")
        metadata["language"] = record.get("language")
        metadata["tags"] = record.get("tags")
        metadata["source"] = record.get("source")
        return metadata

        

        
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
        raise NotImplementedError("This function is not yet implemented.")



    def create_documents(self) -> Dict[str, List[Document]]:
        """
        Load and process all support tickets into LangChain Document objects.
        
        Returns:
            Dict[str, List[Document]]: Dictionary with support types as keys and lists of Document objects as values
        """

        raise NotImplementedError("This function is not yet implemented.")
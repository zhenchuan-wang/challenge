from typing import List, Dict, Any
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import asyncio
import logging

from .vector_store import SupportVectorStore
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_api = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)


class SupportRAGChain:
    """
    A class implementing the Retrieval-Augmented Generation (RAG) chain for support tickets.
    
    This class combines vector similarity search with LLM-based generation to provide
    relevant and contextual responses to support queries.
    
    IMPORTANT:
    - Empty queries MUST be rejected with the EXACT error message: "Query cannot be empty"
    - Queries shorter than 10 characters MUST be rejected with the EXACT error message: 
      "Query too short. Please provide more details."
    - Context preparation MUST follow the exact format specified in _prepare_context
    """
    
    def __init__(self, vector_store: SupportVectorStore):
        """
        Initialize the RAG chain with a vector store and LLM.
        Make sure the llm should be openAI gpt-4o
        
        Args:
            vector_store (SupportVectorStore): Vector store containing support tickets
        """
        raise NotImplementedError("This function is not yet implemented.")

        

    def get_relevant_documents(
        self, 
        query: str, 
        support_type: str = None, 
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant support tickets for a given query.
        
        IMPORTANT:
        - Empty queries or queries shorter than 10 characters MUST be rejected with ValueError
        - The exact error message should be: "Query too short. Please provide more details."
        
        Args:
            query (str): User's support query
            support_type (str, optional): Specific support type to search for
            k (int): Number of documents to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents with metadata
            
        Raises:
            ValueError: If query is empty or too short (less than 10 characters)
        """
        raise NotImplementedError("This function is not yet implemented.")



    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Prepare retrieved documents into a formatted context string.
        
        IMPORTANT:
        - The context MUST be formatted with the EXACT format shown below
        - Each document must include: Support Type, Tags, and Content
        - When no documents are found, return "No relevant support tickets found."
        
        Args:
            documents (List[Dict[str, Any]]): Retrieved similar documents
            
        Returns:
            str: Formatted context string with the exact format:
            
            Ticket {i}:
            Support Type: {doc['metadata'].get('support_type', 'Unknown')}
            Tags: {', '.join(doc['metadata'].get('tags', []))}
            Content: {doc['content']}
        """
        raise NotImplementedError("This function is not yet implemented.")

    async def query(
        self, 
        query: str, 
        support_type: str = None
    ) -> str:
        """
        Generate a response to a support query using RAG.
        
        IMPORTANT:
        - Empty queries MUST be rejected with the EXACT error message: "Query cannot be empty"
        - Queries with only whitespace MUST be rejected with the EXACT error message: "Query cannot be empty"
        - Queries shorter than 10 characters MUST be rejected with the EXACT error message:
          "Query too short. Please provide more details."
        
        Args:
            query (str): User's support query
            support_type (str, optional): Specific support type to search for
            
        Returns:
            str: Generated response based on relevant support tickets
            
        Raises:
            ValueError: With message "Query cannot be empty" if query is empty or whitespace only
            ValueError: With message "Query too short. Please provide more details." if query is shorter than 10 chars
            Exception: If there's an error generating the response
        """
        raise NotImplementedError("This function is not yet implemented.")
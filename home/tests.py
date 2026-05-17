from unittest.mock import patch
from django.test import TestCase
from langchain_core.documents import Document

from .rag_service import (
    load_website,
    split_documents,
    create_vector_store,
    ask_website,
)


class RagTests(TestCase):

    @patch("rag_app.rag.WebBaseLoader")
    def test_load_website(self, mock_loader):

        mock_loader.return_value.load.return_value = [
            Document(page_content="Test content")
        ]

        docs = load_website("https://example.com")

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Test content")

    def test_split_documents(self):

        docs = [
            Document(page_content="Hello world " * 500)
        ]

        chunks = split_documents(docs)

        self.assertTrue(len(chunks) > 1)

    def test_create_vector_store(self):

        docs = [
            Document(page_content="AI is awesome")
        ]

        chunks = split_documents(docs)

        vector_store = create_vector_store(chunks)

        results = vector_store.similarity_search(
            "AI",
            k=1
        )

        self.assertEqual(len(results), 1)

    @patch("rag_app.rag.WebBaseLoader")
    @patch("rag_app.rag.ChatGoogleGenerativeAI")
    def test_ask_website(
        self,
        mock_llm,
        mock_loader
    ):

        mock_loader.return_value.load.return_value = [
            Document(
                page_content="Paris is the capital of France."
            )
        ]

        mock_llm.return_value.invoke.return_value.content = (
            "Paris is the capital of France."
        )

        result = ask_website(
            "https://example.com",
            "What is the capital of France?"
        )

        self.assertIn("answer", result)
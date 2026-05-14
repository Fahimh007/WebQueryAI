import os
from unittest.mock import patch

from django.test import TestCase
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from . import rag_service


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class OpenRouterConfigTests(TestCase):
    @patch("home.rag_service.load_dotenv", return_value=False)
    def test_load_openrouter_config_uses_canonical_openrouter_keys(self, _mock_load_dotenv):
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "OPENROUTER_MODEL": "openai/gpt-4o-mini",
                "OPENROUTER_EMBEDDING_MODEL": "openai/text-embedding-3-small",
            },
            clear=True,
        ):
            config = rag_service._load_openrouter_config()

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.chat_model, "openai/gpt-4o-mini")
        self.assertEqual(config.embedding_model, "openai/text-embedding-3-small")

    @patch("home.rag_service.load_dotenv", return_value=False)
    def test_load_openrouter_config_falls_back_to_legacy_model_name(self, _mock_load_dotenv):
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "MODEL_NAME": "openai/gpt-4o-mini",
                "OPENROUTER_EMBEDDING_MODEL": "openai/text-embedding-3-small",
            },
            clear=True,
        ):
            config = rag_service._load_openrouter_config()

        self.assertEqual(config.chat_model, "openai/gpt-4o-mini")

    @patch("home.rag_service.load_dotenv", return_value=False)
    def test_load_openrouter_config_reports_missing_values_clearly(self, _mock_load_dotenv):
        with patch.dict(
            os.environ,
            {"OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1"},
            clear=True,
        ):
            with self.assertRaisesMessage(
                ValueError,
                "Missing OpenRouter configuration in webinfo/.env: OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_EMBEDDING_MODEL.",
            ):
                rag_service._load_openrouter_config()

    @patch("home.rag_service.OpenAIEmbeddings")
    def test_create_embeddings_uses_explicit_openrouter_settings(self, mock_embeddings):
        config = rag_service.OpenRouterConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            chat_model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
        )

        rag_service._create_embeddings(config)

        mock_embeddings.assert_called_once_with(
            model="openai/text-embedding-3-small",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )

    @patch("home.rag_service.ChatOpenAI")
    def test_create_llm_uses_explicit_openrouter_settings(self, mock_chat_openai):
        config = rag_service.OpenRouterConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            chat_model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
        )

        rag_service._create_llm(config)

        mock_chat_openai.assert_called_once_with(
            model="openai/gpt-4o-mini",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )


class HomeViewTests(TestCase):
    @patch("home.views.get_rag_answer", side_effect=ValueError("Missing OpenRouter configuration in webinfo/.env: OPENROUTER_API_KEY."))
    def test_home_renders_config_errors(self, _mock_get_rag_answer):
        response = self.client.post(
            "/",
            {
                "urlInput": "https://example.com",
                "userQuery": "What is this page about?",
            },
        )

        self.assertContains(response, "Missing OpenRouter configuration in webinfo/.env: OPENROUTER_API_KEY.")


class RagServiceTests(TestCase):
    def test_create_retriever_returns_matching_documents(self):
        docs = [Document(page_content="WebRAG helps answer questions about websites.")]

        retriever = rag_service._create_retriever(docs, FakeEmbeddings())

        results = retriever.invoke("What does WebRAG do?")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "WebRAG helps answer questions about websites.")

    @patch("home.rag_service.WebBaseLoader")
    @patch("home.rag_service._create_embeddings", return_value=FakeEmbeddings())
    @patch("home.rag_service._create_llm", return_value=RunnableLambda(lambda _: "stub answer"))
    @patch(
        "home.rag_service._load_openrouter_config",
        return_value=rag_service.OpenRouterConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            chat_model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
        ),
    )
    def test_get_rag_answer_completes_without_chroma(
        self,
        _mock_config,
        _mock_llm,
        _mock_embeddings,
        mock_loader,
    ):
        mock_loader.return_value.load.return_value = [
            Document(page_content="WebRAG extracts content from a URL and answers questions.")
        ]

        answer = rag_service.get_rag_answer("https://example.com", "What does this app do?")

        self.assertEqual(answer, "stub answer")

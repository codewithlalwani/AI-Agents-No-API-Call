import unittest
from unittest.mock import Mock, patch

import numpy as np
from streamlit.testing.v1 import AppTest

from day3 import ai_web_scrapper2 as app


class ScraperTests(unittest.TestCase):
    def setUp(self):
        self.store = {"index": None, "chunks": [], "urls": set()}
        self.storage_patch = patch.object(app, "initialize_storage", return_value=self.store)
        self.storage_patch.start()
        self.addCleanup(self.storage_patch.stop)
        self.embeddings = Mock()
        self.embeddings.embed_documents.side_effect = lambda texts: [[1., 0.] for _ in texts]
        self.embeddings.embed_query.return_value = [1., 0.]
        self.embedding_patch = patch.object(app, "get_embeddings", return_value=self.embeddings)
        self.embedding_patch.start()
        self.addCleanup(self.embedding_patch.stop)

    def test_chunk_mapping_multiple_pages_and_duplicate(self):
        app.store_in_faiss("First page sentence. " * 100, "https://example.com/one")
        app.store_in_faiss("Second page.", "https://example.com/two")
        self.assertEqual(self.store["index"].ntotal, len(self.store["chunks"]))
        self.assertGreater(self.store["index"].ntotal, 2)
        self.assertTrue(all(len(c["text"]) <= 500 for c in self.store["chunks"]))
        count = self.store["index"].ntotal
        app.store_in_faiss("Repeated page.", "https://example.com/two")
        self.assertEqual(self.store["index"].ntotal, count)

    def test_question_and_source_reach_llm_with_small_index(self):
        app.store_in_faiss("The capital is Paris.", "https://example.com")
        llm = Mock()
        llm.invoke.return_value = "Paris"
        with patch.object(app, "get_llm", return_value=llm):
            self.assertEqual(app.retrieve_and_answer("What is the capital?"), "Paris")
        prompt = llm.invoke.call_args.args[0]
        for text in ("What is the capital?", "The capital is Paris.", "https://example.com"):
            self.assertIn(text, prompt)

    def test_empty_store_and_empty_inputs(self):
        self.assertIn("Scrape and store", app.retrieve_and_answer("Question"))
        for callback, args in ((app.retrieve_and_answer, (" ",)), (app.store_in_faiss, (" ", "https://example.com")), (app.scrape_website, ("bad-url",))):
            with self.assertRaises(ValueError):
                callback(*args)
        self.embeddings.embed_documents.assert_not_called()

    def test_failed_embedding_does_not_commit_data(self):
        self.embeddings.embed_documents.return_value = [[np.nan, 0.]]
        self.embeddings.embed_documents.side_effect = None
        with self.assertRaises(ValueError):
            app.store_in_faiss("Text", "https://example.com")
        self.assertIsNone(self.store["index"])
        self.assertEqual(self.store["chunks"], [])

    def test_scraping_and_http_errors(self):
        response = Mock()
        response.headers = {"Content-Type": "text/html"}
        response.content = b"<body><h1>Title</h1><p>Hello</p><script>bad()</script></body>"
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(app.requests, "get", return_value=response) as get:
            self.assertEqual(app.scrape_website("https://example.com"), "Title\nHello")
            self.assertEqual(get.call_args.kwargs["timeout"], 30)
            response.raise_for_status.side_effect = app.requests.HTTPError("404")
            with self.assertRaises(app.requests.HTTPError):
                app.scrape_website("https://example.com")


class StreamlitTests(unittest.TestCase):
    def test_startup_validation_persistence_and_clear(self):
        at = AppTest.from_file("day3/ai_web_scrapper2.py").run(timeout=30)
        self.assertFalse(at.exception)
        at.text_input[0].set_value("bad-url")
        at.button[1].click().run()
        self.assertFalse(at.exception)
        self.assertIn("valid website URL", at.error[0].value)
        at.session_state["scraper_store"] = {"index": None, "chunks": [], "urls": {"https://example.com"}}
        at.run()
        self.assertIn("https://example.com", at.session_state["scraper_store"]["urls"])
        at.button[0].click().run()
        self.assertEqual(at.session_state["scraper_store"]["urls"], set())
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()

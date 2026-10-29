
import unittest

from src.qa_assistant import DocumentQA, FALLBACK


class TestDocumentQA(unittest.TestCase):
    def setUp(self):
        self.qa = DocumentQA({
            "internship.txt": (
                "Project 4 is a Document Question-Answering Assistant. "
                "The assistant uses Python and TF-IDF retrieval "
                "with cosine similarity. "
                "Every answer should include a supporting source passage."
            )
        })

    def test_project_question_returns_relevant_answer(self):
        result = self.qa.answer("What is Project 4?")

        self.assertIn("Document Question-Answering Assistant",
                      result["answer"])
        self.assertTrue(result["sources"])

    def test_tools_question_returns_relevant_answer(self):
        result = self.qa.answer("Which tools does the assistant use?")

        self.assertIn("TF-IDF", result["answer"])
        self.assertTrue(result["sources"])

    def test_unrelated_question_returns_fallback(self):
        result = self.qa.answer("What is the capital of Mars?")

        self.assertEqual(result["answer"], FALLBACK)
        self.assertEqual(result["sources"], [])

    def test_empty_question_returns_fallback(self):
        result = self.qa.answer("")

        self.assertEqual(result["answer"], FALLBACK)
        self.assertEqual(result["sources"], [])

    def test_source_filename_is_preserved(self):
        result = self.qa.answer("What is Project 4?")

        self.assertEqual(
            result["sources"][0]["source"],
            "internship.txt",
        )


if __name__ == "__main__":
    unittest.main()

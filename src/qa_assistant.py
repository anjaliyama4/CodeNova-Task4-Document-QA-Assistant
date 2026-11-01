
from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
FALLBACK = "Information not found in provided document"



_embedding_model = None


def chunk_text(text, max_words=60, similarity_threshold=0.30):
    """Group adjacent sentences using semantic similarity and paragraph boundaries."""
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    passages = []
    paragraphs = re.split(r"\n\s*\n", text.strip())

    for paragraph in paragraphs:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip())
            if sentence.strip()
        ]

        if not sentences:
            continue

        embeddings = _embedding_model.encode(sentences)
        current_chunk = [sentences[0]]
        current_words = len(sentences[0].split())

        for index in range(1, len(sentences)):
            sentence = sentences[index]
            sentence_words = len(sentence.split())

            similarity = cosine_similarity(
                [embeddings[index - 1]],
                [embeddings[index]]
            )[0][0]

            if (
                similarity >= similarity_threshold
                and current_words + sentence_words <= max_words
            ):
                current_chunk.append(sentence)
                current_words += sentence_words
            else:
                passages.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_words = sentence_words

        if current_chunk:
            passages.append(" ".join(current_chunk))

    return passages



class DocumentQA:
    def __init__(self, documents):
        self.passages = []

        for filename, text in documents.items():
            for number, passage in enumerate(chunk_text(text), start=1):
                self.passages.append({
                    "source": filename,
                    "passage_number": number,
                    "text": passage
                })

        self.vectorizer = None
        self.document_matrix = None

        if self.passages:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
            self.document_matrix = self.vectorizer.fit_transform(
                [item["text"] for item in self.passages]
            )

    @classmethod
    def from_folder(cls, folder="data"):
        folder_path = Path(folder)

        if not folder_path.exists():
            raise FileNotFoundError(
                f"Document folder not found: {folder_path.resolve()}"
            )

        documents = {
            file.name: file.read_text(encoding="utf-8")
            for file in sorted(folder_path.glob("*.txt"))
            if file.is_file()
        }

        if not documents:
            raise FileNotFoundError(
                f"No .txt documents found in {folder_path.resolve()}"
            )

        return cls(documents)

    def answer(self, question):
        if not question or not question.strip() or not self.passages:
            return {"answer": FALLBACK, "sources": []}

        query_matrix = self.vectorizer.transform([question])

        if query_matrix.nnz == 0:
            return {"answer": FALLBACK, "sources": []}

        scores = cosine_similarity(
            query_matrix, self.document_matrix
        ).ravel()

        best_index = int(scores.argmax())
        best_score = float(scores[best_index])

        # Reject questions that do not match the documents sufficiently.
        if best_score < 0.10:
            return {"answer": FALLBACK, "sources": []}

        passage = self.passages[best_index]

        return {
            "answer": passage["text"],
            "sources": [{
                "source": passage["source"],
                "passage_number": passage["passage_number"],
                "text": passage["text"]
            }]
        }


def main():
    try:
        assistant = DocumentQA.from_folder("data")
    except (FileNotFoundError, OSError) as error:
        print(f"Error: {error}")
        return

    print("Document Question-Answering Assistant (TF-IDF)")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        result = assistant.answer(question)
        print(f"\nAnswer: {result['answer']}")

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(
                    f"- {source['source']}, "
                    f"passage {source['passage_number']}"
                )
        else:
            print("\nSources: None — no supporting passage was found.")

        print()


if __name__ == "__main__":
    main()

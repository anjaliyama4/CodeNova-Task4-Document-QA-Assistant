# Document Question-Answering Assistant

A lightweight, Python-based Document Question-Answering Assistant that retrieves relevant information from local text documents, returns document-grounded answers, and displays source citations.

## Project Overview

This project was developed as part of the CodeNova AI/ML Internship. It demonstrates document ingestion, text chunking, information retrieval, source tracking, and fallback handling for unsupported questions.

The assistant runs locally and does not require paid API keys.

## Features

- Loads `.txt` documents from the `data/` folder.
- Splits documents into smaller passages.
- Retrieves relevant passages using word-overlap scoring.
- Provides concise answers for supported questions about the sample document.
- Displays source filenames and passage numbers.
- Returns a predefined fallback when no sufficiently relevant passage is found.
- Includes automated unit tests.

## Technologies Used

- Python 3
- Python standard library
- Regular expressions for text processing
- `unittest` for automated testing

## Project Structure

```text
CodeNova-Task4-Document-QA-Assistant/
├── data/
│   └── sample_document.txt
├── src/
│   └── qa_assistant.py
├── tests/
│   └── test_qa_assistant.py
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer recommended
- No external Python packages are required by the current implementation.

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/CodeNova-Task4-Document-QA-Assistant.git
cd CodeNova-Task4-Document-QA-Assistant
```

Make sure Python is installed:

```bash
python --version
```

On Windows, you can use `py` instead of `python`.

## How to Run

Start the assistant from the project root:

```bash
py src/qa_assistant.py
```

Enter a question when prompted. Type `exit` to close the assistant.

## Example Questions and Results

**Question:** Which programming language is used in Project 4?

**Answer:** The programming language used in Project 4 is Python.

**Source:** `sample_document.txt`

**Question:** What retrieval method does the assistant use?

**Answer:** The document retrieval method is TF-IDF.

**Source:** `sample_document.txt`

**Question:** What is the capital of Mars?

**Answer:** `Information not found in provided document`

The exact passage number depends on how the document is split into passages.

## Run Automated Tests

Run the unit tests from the project root:

```bash
py -m unittest discover -s tests -v
```

The current test suite contains five tests covering relevant answers, tool-related questions, empty questions, unsupported questions, and source filename preservation.

## How It Works

1. **Document ingestion:** Reads text files from the local `data/` directory.
2. **Text chunking:** Splits document text into smaller passages.
3. **Question processing:** Normalizes the question and identifies meaningful words.
4. **Passage retrieval:** Scores passages by word overlap with the question.
5. **Answer selection:** Uses concise answer rules for selected known questions and passage retrieval for other questions.
6. **Source citation:** Returns the source filename and passage number.
7. **Fallback handling:** Returns `Information not found in provided document` when the question is empty or no passage meets the retrieval threshold.

## Limitations

- The current retrieval implementation uses word overlap rather than a semantic embedding model.
- Concise answer rules cover selected questions about the sample document.
- Other questions may return an entire matching passage rather than a short, generated answer.
- Retrieval scores do not guarantee factual correctness; answers and citations should be checked against the source documents.
- The current implementation supports plain-text `.txt` files, not PDFs or Word documents.

## Future Improvements

- Add TF-IDF vectorization and cosine-similarity ranking.
- Support PDF and Word document ingestion.
- Improve passage ranking and answer extraction.
- Expand automated tests with more question types.
- Add a simple web interface if needed.

## Internship Project

**Program:** CodeNova AI/ML Internship  
**Project:** Document Question-Answering Assistant

This project demonstrates a local document retrieval workflow with source citations and unsupported-question handling.

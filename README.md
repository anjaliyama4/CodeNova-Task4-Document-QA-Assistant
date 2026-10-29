# Document Question-Answering Assistant (TF-IDF)

A lightweight Python-based Document Question-Answering Assistant that retrieves relevant information from local text documents using **TF-IDF vectorization and cosine similarity**. It returns document-grounded answers with source citations and handles unsupported questions safely.

## Project Overview

This project was developed as part of the CodeNova AI/ML Internship. It demonstrates document ingestion, passage chunking, TF-IDF-based information retrieval, source tracking, and fallback handling for unsupported questions.

The assistant runs locally using free, open-source tools and does not require paid API keys.

## Features

- Loads local `.txt` documents from the `data/` folder.
- Splits documents into smaller passages.
- Uses **TF-IDF vectorization** to represent text.
- Ranks passages using **cosine similarity**.
- Returns relevant text from the retrieved passage.
- Displays source filenames and passage numbers.
- Returns `Information not found in provided document` when no sufficiently relevant passage is found.
- Includes automated unit tests.

## Technologies Used

- Python 3
- scikit-learn (`TfidfVectorizer` and cosine similarity)
- Python standard library
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
- scikit-learn

Install the required dependency from the project root:

```bash
py -m pip install -r requirements.txt
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/anjaliyama4/CodeNova-Task4-Document-QA-Assistant.git
cd CodeNova-Task4-Document-QA-Assistant
```

Check that Python is installed:

```bash
py --version
```

## How to Run

Start the assistant from the project root:

```bash
py src/qa_assistant.py
```

Enter a question when prompted. Type `exit` to close the assistant.

## Example Questions and Results

**Question:** Which programming language is used in Project 4?

**Answer:** The programming language used in Project 4 is Python.

**Source:** `sample_document.txt`, passage 2

**Question:** What retrieval method does the assistant use?

**Answer:** The document retrieval method is TF-IDF.

**Source:** `sample_document.txt`, passage 3

**Question:** What is the capital of Mars?

**Answer:** `Information not found in provided document`

The passage numbers correspond to the passages created from the sample document.

## Run Automated Tests

Run the unit tests from the project root:

```bash
py -m unittest discover -s tests -v
```

The test suite covers relevant answers, tool-related questions, empty questions, unsupported questions, and source filename preservation.

## How It Works

1. **Document ingestion:** Reads text documents from the local `data/` directory.
2. **Text chunking:** Splits document content into passages.
3. **Question processing:** Converts the question into a TF-IDF vector.
4. **Passage retrieval:** Uses cosine similarity to rank document passages by relevance.
5. **Answer selection:** Returns text from the most relevant passage rather than inventing unsupported facts.
6. **Source citation:** Identifies the source filename and passage number.
7. **Fallback handling:** Returns `Information not found in provided document` when the question is empty or no sufficiently relevant passage is found.

## Limitations

- The current implementation supports plain-text `.txt` files only.
- It returns extracted passage text rather than generating answers with a large language model.
- TF-IDF measures lexical similarity and may not recognize questions expressed using very different wording.
- A similarity score alone does not guarantee factual correctness; answers should be checked against the source document.

## Future Improvements

- Support PDF and Word document ingestion.
- Improve passage ranking and answer extraction.
- Expand automated tests with additional question types.
- Add an optional web interface.

## Internship Project

**Program:** CodeNova AI/ML Internship  
**Project:** Document Question-Answering Assistant

This project demonstrates a local document retrieval workflow using TF-IDF, cosine similarity, source citations, and unsupported-question handling.

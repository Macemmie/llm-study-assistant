# Research and Study Assistant

A teaching application for document questions, model comparison and responsible
use of large language models.

**[Open the application](https://test-app-study-assistant.streamlit.app/)**

## Student activity

Type your answers in the PDF's shaded response spaces and save your copy.

Follow [the one-page worksheet](Student_Guide.pdf), using
[Practice Notes](samples/Practice_Notes.pdf). You can also upload
[Evaluation Notes](samples/Evaluation_Notes.pdf) as a second PDF.

1. Upload one or two PDFs together and ask a question.
2. Check an answer against the supporting passages.
3. Compare GPT-OSS 120B, GPT-OSS 20B and Qwen 3.8 27B using the same PDFs and question.
4. Test a question that the notes cannot answer.

## The application

Choose GPT-OSS 120B, GPT-OSS 20B or Qwen 3.8 27B. Upload one or two PDFs
with selectable text and ask about their contents. The latest answer shows
the model used and supporting source passages.

Python extracts the text and splits it into passages. TF-IDF and cosine
similarity find up to three matching passages. The app sends the passages
and question to the selected Groq-hosted model. Streamlit runs the app and
provides its interface. Uploading a PDF does not retrain the model.

## Responsible use

Use public or non-sensitive documents you are permitted to upload. Questions
and selected passages are sent to Groq. Check claims against the displayed
evidence: citations do not guarantee correctness, and retrieval can miss
relevant information. Scanned PDFs need OCR, which this app does not include.
The app accepts at most two PDFs, each up to 10 MB and 100 pages.

## Running your own copy

Use Python 3.14 and uv. Run `uv sync`, then
`uv run streamlit run streamlit_app.py`. Configure `GROQ_API_KEY` privately
in Streamlit Secrets or an environment variable. Never commit a real key.
Model access is subject to the configured account's limits.

The project retains the Apache-2.0 license from the Streamlit template.

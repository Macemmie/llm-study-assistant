# Research and Study Assistant
## Student worksheet

Open the app: https://test-app-study-assistant.streamlit.app/

You need a browser. Two practice PDFs are supplied.
Type your answers in the shaded spaces in the PDF and save your copy.
Use public or non-sensitive PDFs. Your question and selected passages are sent to Groq.

### 1. Upload your notes

- Open the app and choose **GPT-OSS 120B** in the sidebar.
- Upload **one or two PDFs**: `Practice_Notes.pdf`, with `Evaluation_Notes.pdf`
  as an optional second file. To select both together, click **Upload**, hold
  **Ctrl** and click both files, then click **Open**.
- Wait for **PDF text ready**. With two PDFs, the app searches across both.

### 2. Ask and check

Ask **What is RAG?** and click **Ask**.
Open **View supporting passages**. Choose one claim from the answer and find
the passage that supports it. A citation alone does not prove a claim is correct.

Claim checked:



Source filename/page:



**What is happening?** The app finds matching passages in the PDF and sends
them with your question to the LLM. The LLM uses them to write an answer.
This process is called retrieval-augmented generation (RAG).

### 3. Compare all three models

Keep the same PDF(s) and question. Select **GPT-OSS 20B** and click **Ask**.
Repeat with **Qwen 3.8 27B**. Record each result before switching models.
Changing the dropdown alone does not generate a new answer.

| Model | Is the answer supported by the passages? |
|---|---|
| GPT-OSS 120B | |
| GPT-OSS 20B | |
| Qwen 3.8 27B | |

Which answer is clearest, and why? 

### 4. Test a question the notes cannot answer

Ask **What accuracy percentage did GPT-OSS 120B achieve on these notes?**

The practice notes contain no such result. Does the model acknowledge missing
information, or invent an answer?

What happened? 

### Finish

In two sentences, explain how the PDF(s) help the LLM and what you would check
before using its answer in your studies.

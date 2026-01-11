# 🎓 Agentic Academic Grader: RAG & Human-in-the-Loop

An intelligent automated grading system built with **LangGraph**, **LangChain**, and **OpenAI/LM Studio**. This project moves beyond simple prompting by using an agentic state machine to handle grading, context retrieval, and human verification.

## 🚀 Key Features

* **Stateful Orchestration**: Managed via `LangGraph`, allowing for complex conditional paths and state persistence.
* **Hybrid Grading Logic**: Uses an initial AI evaluation, falling back to **RAG (Retrieval-Augmented Generation)** if the context is unclear.
* **Human-in-the-Loop (HITL)**: Automatically pauses execution for manual review when AI confidence is low or scores are ambiguous.
* **Robust Data Extraction**: Custom regex-based parsers for PDF document loading and cleaning JSON responses from local LLMs.
* **Local LLM Support**: Optimized for local inference using **LM Studio** (Qwen 2.5 14B).

## 🛠️ Architecture & Workflow

The system follows a directed graph logic to ensure high-quality grading:

1. **Initial Grader**: Evaluates the student's answer based on general knowledge.
2. **Router Logic**: Analyzes the score and confidence.
* **High Confidence/High Score**: Direct to Completion.
* **Low Score**: Routes to **RAG Fallback** to check against a reference textbook.
* **Low Confidence**: Routes to **Human Review**.


3. **RAG Fallback**: Performs a keyword-based search on a reference PDF to provide the LLM with factual context before re-grading.
4. **Human Review**: Uses LangGraph `interrupts` to pause the thread, allowing a human to override or confirm the grade.

## 💻 Tech Stack

* **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph)
* **LLM Framework:** [LangChain](https://github.com/langchain-ai/langchain)
* **Model:** Qwen 2.5 14B (via LM Studio)
* **Data Handling:** Pandas, PyPDFLoader
* **State Management:** TypedDict & MemorySaver

## 📋 Prerequisites

* **Python 3.10+**
* **LM Studio** running a local server at `http:localhost/v1` (or update `LM_STUDIO_URL` in the code).
* Input Files:
* `thebook.pdf`: The reference textbook.
* `student_notebook.pdf`: The student's questions and answers.



## ⚙️ Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/your-username/agentic-grader.git
cd agentic-grader

```


2. **Install dependencies**
```bash
pip install langchain langchain-openai langgraph pandas pypdf

```


3. **Run the Grader**
```bash
python main.py

```



## 📊 Output

The system generates a `final_grades.csv` containing:

* The original question.
* The final score (AI-generated or Human-adjusted).
* Qualitative feedback and verification stamps.

---

### Future Enhancements

* [ ] Integration with Vector Databases (ChromaDB/Pinecone) for semantic RAG.
* [ ] Multi-modal support for grading diagrams and handwritten notes.
* [ ] Web UI for the Human-in-the-Loop interface.

**Would you like me to help you write a technical blog post or a LinkedIn snippet to showcase this project?**

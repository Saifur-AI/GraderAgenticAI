import json
import re
import pandas as pd
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# =========================
# 1. STATE DEFINITION
# =========================
class GradingState(TypedDict):
    question: str
    answer: str
    grade_result: Optional[dict]


# =========================
# 2. CONFIG & LLM
# =========================
LM_STUDIO_URL = "http://10.120.0.46:5000/v1"
API_KEY = "lm-studio"

llm = ChatOpenAI(
    base_url=LM_STUDIO_URL,
    api_key=API_KEY,
    model="qwen2.5-14b-instruct",
    temperature=0.1,  # Keep temperature low for better JSON stability
    max_tokens=1500
)


# =========================
# 3. ROBUST UTILITIES
# =========================
def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)


def clean_json_response(content: str):
    """
    Finds the JSON block inside a string even if the AI
    adds prose before or after the code block.
    """
    # 1. Try to find content between curly braces
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = content

    # 2. Strip markdown backticks
    json_str = re.sub(r'```json|```', '', json_str).strip()

    try:
        data = json.loads(json_str)
        return {
            "total": int(data.get("total", 0)),
            "summary_feedback": data.get("summary_feedback", "N/A"),
            "confidence": int(data.get("confidence", 0))
        }
    except Exception as e:
        # Fallback if the AI broke the JSON format
        return {
            "total": 0,
            "summary_feedback": f"Parsing Error. AI said: {content[:100]}...",
            "confidence": 0
        }


def extract_qa_pairs(text):
    lines = text.splitlines()
    qa_pairs = []
    current_question = None
    current_answer = []
    question_patterns = [r'^##\s*Question', r'^Question\s*\d+', r'^Q\d+[\.:)]', r'^\d+\.\s+.*\?']

    def is_question(line):
        return any(re.match(p, line, re.IGNORECASE) for p in question_patterns)

    for line in lines:
        line = line.rstrip()
        if not line: continue
        if is_question(line):
            if current_question:
                qa_pairs.append({"question": current_question, "answer": "\n".join(current_answer).strip()})
            current_question = line
            current_answer = []
            continue
        if current_question:
            current_answer.append(line)

    if current_question:
        qa_pairs.append({"question": current_question, "answer": "\n".join(current_answer).strip()})
    return qa_pairs


def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - 100)]


def rag_search(question, reference_chunks):
    """Simple keyword search. Returns the single most relevant chunk."""
    q_words = set(question.lower().split())
    scored = []
    for c in reference_chunks:
        score = len(q_words & set(c.lower().split()))
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 2 else ""


# =========================
# 4. NODES & ROUTING
# =========================
def initial_grader_node(state: GradingState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert academic professor. Evaluate the student answer against the question. Be fair. If the answer is correct, give high points. Output ONLY a JSON object."),
        ("human",
         "QUESTION:\n{question}\n\nSTUDENT ANSWER:\n{answer}\n\nRequired JSON format:\n{{\"total\": 0-100, \"summary_feedback\": \"your text\", \"confidence\": 0-100}}")
    ])
    output = (prompt | llm).invoke({"question": state["question"], "answer": state["answer"]})
    return {"grade_result": clean_json_response(output.content)}


def rag_fallback_node(state: GradingState, reference_chunks):
    context = rag_search(state["question"], reference_chunks)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader with access to a textbook snippet. Use the context ONLY if relevant. If context is unrelated, grade based on general facts. Output ONLY JSON."),
        ("human", "CONTEXT FROM TEXTBOOK:\n{context}\n\nQUESTION:\n{question}\n\nSTUDENT ANSWER:\n{answer}")
    ])
    output = (prompt | llm).invoke({"context": context, "question": state["question"], "answer": state["answer"]})
    return {"grade_result": clean_json_response(output.content)}


def router_logic(state: GradingState):
    res = state["grade_result"]
    # If the score is high and the AI is very sure, we are done
    if res.get("total", 0) >= 85 and res.get("confidence", 0) >= 85:
        return "end"
    # If the score is low, try looking at the textbook (RAG)
    if res.get("total", 0) < 60:
        return "fallback"
    # Otherwise (moderate score or low confidence), ask the human
    return "human"


# =========================
# 5. BUILD GRAPH
# =========================
def build_agentic_graph(reference_chunks):
    memory = MemorySaver()
    workflow = StateGraph(GradingState)

    workflow.add_node("initial_grader", initial_grader_node)
    workflow.add_node("rag_fallback", lambda state: rag_fallback_node(state, reference_chunks))
    workflow.add_node("human_review", lambda state: state)

    workflow.add_edge(START, "initial_grader")
    workflow.add_conditional_edges("initial_grader", router_logic, {
        "fallback": "rag_fallback",
        "human": "human_review",
        "end": END
    })
    workflow.add_edge("rag_fallback", "human_review")
    workflow.add_edge("human_review", END)

    return workflow.compile(checkpointer=memory, interrupt_before=["human_review"])


# =========================
# 6. MAIN EXECUTION
# =========================
if __name__ == "__main__":
    # Ensure these files exist in your folder
    ref_chunks = chunk_text(load_pdf("thebook.pdf"))
    qa_pairs = extract_qa_pairs(load_pdf("student_notebook.pdf"))

    app = build_agentic_graph(ref_chunks)
    final_results = []

    for i, qa in enumerate(qa_pairs):
        config = {"configurable": {"thread_id": f"q_run_{i}"}}

        # Start Graph
        app.invoke({"question": qa["question"], "answer": qa["answer"]}, config)

        # Display Progress
        print("\n" + "=" * 60)
        print(f"QUESTION {i + 1}: {qa['question']}")
        print("-" * 30)

        snapshot = app.get_state(config)
        ai_res = snapshot.values.get("grade_result")

        # Handle Human Intervention
        if snapshot.next:
            print(f"🤖 AI SUGGESTION (Confidence: {ai_res.get('confidence')}%):")
            print(f"   Score: {ai_res.get('total')} | Feedback: {ai_res.get('summary_feedback')}")

            choice = input("\n[✋ REVIEW] Press Enter to accept or type a NEW SCORE: ").strip()
            if choice:
                new_grade = ai_res.copy()
                new_grade["total"] = int(choice)
                new_grade["summary_feedback"] += " (Human Verified)"
                app.update_state(config, {"grade_result": new_grade})

            app.invoke(None, config)  # Finalize the thread
            final_state = app.get_state(config).values
        else:
            print(f"✅ AI AUTO-GRADE (Confidence: {ai_res.get('confidence')}%):")
            print(f"   Score: {ai_res.get('total')} | Feedback: {ai_res.get('summary_feedback')}")
            final_state = snapshot.values

        final_results.append({
            "Question": final_state["question"],
            "Score": final_state["grade_result"].get("total"),
            "Feedback": final_state["grade_result"].get("summary_feedback")
        })

    # Save to CSV
    pd.DataFrame(final_results).to_csv("final_grades.csv", index=False)
    print("\n" + "=" * 60)
    print("✅ PROCESS COMPLETE. Results saved to final_grades.csv")
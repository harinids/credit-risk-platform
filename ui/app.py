"""
Streamlit UI for the Credit Risk Intelligence Platform.

Talks to the FastAPI backend (src/api/main.py) via HTTP. API_BASE_URL
points at the "api" service name inside Docker's network (see
docker-compose.yml), or localhost for local dev outside Docker.

Status: Chatbot tab is fully wired to the real /chat endpoint.
Other tabs are placeholders pending Phase 3/4 (predict/explain/rules/eda
endpoints still 501 stubs in main.py).
"""

import os
import uuid

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

st.set_page_config(page_title="Credit Risk Intelligence Platform", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("AI-Powered Credit Risk Intelligence Platform")

tab_eda, tab_predict, tab_explain, tab_rules, tab_chat = st.tabs(
    ["EDA", "Prediction", "Explainability", "Rules", "Chatbot"]
)

with tab_eda:
    st.header("Exploratory Data Analysis")
    st.info("EDA summary endpoint not yet wired (Phase 1 stub in the API). "
            "See notebooks/eda_outputs/ for generated charts in the meantime.")

with tab_predict:
    st.header("Risk Prediction")
    st.info("Prediction endpoint not yet wired (Phase 3 stub in the API).")

with tab_explain:
    st.header("Explainability")
    st.info("Explainability endpoint not yet wired (Phase 4 stub in the API).")

with tab_rules:
    st.header("Business Rules")
    st.info("Rules endpoint not yet wired (Phase 4 stub in the API).")

with tab_chat:
    st.header("Talk to Your Data")

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("sql"):
                with st.expander("View generated SQL"):
                    st.code(turn["sql"], language="sql")

    question = st.chat_input("Ask a question about the loan data...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={"question": question, "session_id": st.session_state.session_id},
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.write(result["answer"])
                    if result.get("sql"):
                        with st.expander("View generated SQL"):
                            st.code(result["sql"], language="sql")
                    st.session_state.chat_history.append(result)
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach the API: {exc}")

st.sidebar.header("System Status")
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
    st.sidebar.success(f"API: {health['status']}")
    st.sidebar.text(f"DB: {health['db']}")
    st.sidebar.text(f"LLM: {health['llm_provider']}")
except requests.exceptions.RequestException:
    st.sidebar.error("API unreachable")

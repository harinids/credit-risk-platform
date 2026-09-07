"""
Streamlit UI for the Credit Risk Intelligence Platform.

All 5 tabs wired to real API endpoints (see src/api/main.py):
  EDA            -> GET /eda/summary
  Prediction     -> GET /predict/sample-ids, GET /predict/{id}
  Explainability -> GET /explain/{id}
  Rules          -> GET /rules
  Chatbot        -> POST /chat, DELETE /chat/{session_id}

API_BASE_URL defaults to the Docker-network service name ("http://api:8000"),
matching docker-compose.yml. For local testing outside Docker, set
API_BASE_URL=http://localhost:8000 before running streamlit.
"""

import os
import uuid

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
REQUEST_TIMEOUT = 30

st.set_page_config(page_title="Credit Risk Intelligence Platform", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("AI-Powered Credit Risk Intelligence Platform")


def api_get(path, params=None):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


tab_eda, tab_predict, tab_explain, tab_rules, tab_chat = st.tabs(
    ["EDA", "Prediction", "Explainability", "Rules", "Chatbot"]
)

with tab_eda:
    st.header("Exploratory Data Analysis")
    if st.button("Load EDA Summary", key="load_eda"):
        with st.spinner("Running EDA..."):
            data, err = api_get("/eda/summary")
        if err:
            st.error(f"Could not load EDA summary: {err}")
        else:
            st.session_state["eda_data"] = data

    data = st.session_state.get("eda_data")
    if data:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{data['rows']:,}")
        c2.metric("Columns", data["columns"])
        c3.metric("Class Imbalance Ratio", f"{data['class_imbalance_ratio']}:1")

        st.subheader("Target Distribution")
        target_df = pd.DataFrame({
            "Outcome": ["Repaid", "Defaulted"],
            "Percent": [data["target_distribution"]["repaid_pct"], data["target_distribution"]["defaulted_pct"]],
        }).set_index("Outcome")
        st.bar_chart(target_df)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Feature Categories")
            cat_df = pd.DataFrame(list(data["feature_categories"].items()), columns=["Category", "Count"]).set_index("Category")
            st.bar_chart(cat_df)
        with col_b:
            st.subheader("Top Missing-Value Columns (%)")
            missing_df = pd.DataFrame(list(data["top_missing_columns"].items()), columns=["Column", "PctMissing"]).set_index("Column")
            st.bar_chart(missing_df)

        st.subheader("Business Insights")
        for insight in data["business_insights"]:
            st.markdown(f"- {insight}")

        st.subheader("Data Quality Flags")
        for flag in data["data_quality_flags"]:
            st.warning(flag)
    else:
        st.info("Click 'Load EDA Summary' to run the analysis (may take a minute or two on first load).")

with tab_predict:
    st.header("Risk Prediction")
    st.caption("Looks up an existing applicant by ID and returns a risk score and band.")

    if st.button("Load sample applicant IDs", key="load_ids_predict"):
        with st.spinner("Fetching sample IDs..."):
            data, err = api_get("/predict/sample-ids", params={"limit": 20})
        if err:
            st.error(f"Could not fetch sample IDs: {err}")
        else:
            st.session_state["sample_ids"] = data["sample_ids"]

    sample_ids = st.session_state.get("sample_ids", [])
    if sample_ids:
        selected_id = st.selectbox("Select an applicant ID", sample_ids, key="predict_id")
        if st.button("Predict Risk", key="predict_btn"):
            with st.spinner("Scoring applicant..."):
                data, err = api_get(f"/predict/{selected_id}")
            if err:
                st.error(f"Prediction failed: {err}")
            else:
                band = data["risk_band"]
                color = {"Low": "green", "Medium": "orange", "High": "red"}.get(band, "gray")
                st.markdown(f"### Risk Band: :{color}[{band}]")
                c1, c2, c3 = st.columns(3)
                c1.metric("Default Probability", f"{data['default_probability']:.1%}")
                c2.metric("Applicant ID", data["sk_id_curr"])
                c3.metric("Actual Outcome (historical)", data["actual_outcome"])
    else:
        st.info("Click 'Load sample applicant IDs' to get started.")

with tab_explain:
    st.header("Explainability")
    st.caption("SHAP-based explanation showing which features drove the risk score for a specific applicant.")

    if st.button("Load sample applicant IDs", key="load_ids_explain"):
        with st.spinner("Fetching sample IDs..."):
            data, err = api_get("/predict/sample-ids", params={"limit": 20})
        if err:
            st.error(f"Could not fetch sample IDs: {err}")
        else:
            st.session_state["sample_ids_explain"] = data["sample_ids"]

    sample_ids_explain = st.session_state.get("sample_ids_explain", [])
    if sample_ids_explain:
        selected_id_e = st.selectbox("Select an applicant ID", sample_ids_explain, key="explain_id")
        if st.button("Explain Prediction", key="explain_btn"):
            with st.spinner("Computing SHAP explanation..."):
                data, err = api_get(f"/explain/{selected_id_e}")
            if err:
                st.error(f"Explanation failed: {err}")
            else:
                band = data["risk_band"]
                color = {"Low": "green", "Medium": "orange", "High": "red"}.get(band, "gray")
                st.markdown(f"### Risk Band: :{color}[{band}]  (probability: {data['default_probability']:.1%})")
                st.write(data["explanation"])

                st.subheader("Top Contributing Factors")
                factors_df = pd.DataFrame(data["top_factors"])
                factors_df = factors_df.rename(columns={
                    "feature": "Feature", "value": "Value", "impact": "SHAP Impact", "direction": "Direction"
                })
                st.dataframe(factors_df, use_container_width=True)
    else:
        st.info("Click 'Load sample applicant IDs' to get started.")

with tab_rules:
    st.header("Business Rules")
    st.caption("Human-readable rules distilled from the ML model via a surrogate decision tree.")

    if st.button("Load Rules", key="load_rules"):
        with st.spinner("Loading derived rules..."):
            data, err = api_get("/rules")
        if err:
            st.error(f"Could not load rules: {err}")
        else:
            st.session_state["rules_data"] = data

    rules_data = st.session_state.get("rules_data")
    if rules_data:
        st.subheader("Depth Sensitivity (fidelity vs. rule count)")
        sensitivity = rules_data.get("depth_sensitivity", [])
        if sensitivity:
            sens_df = pd.DataFrame(sensitivity)
            st.dataframe(sens_df, use_container_width=True)

        st.subheader("Derived Rules")
        st.text(rules_data["rules_text"])
    else:
        st.info("Click 'Load Rules' to view the derived business rules.")

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
                        timeout=REQUEST_TIMEOUT,
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

    if st.button("Clear conversation", key="clear_chat"):
        try:
            requests.delete(f"{API_BASE_URL}/chat/{st.session_state.session_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass
        st.session_state.chat_history = []
        st.rerun()

st.sidebar.header("System Status")
health, herr = api_get("/health")
if health:
    st.sidebar.success(f"API: {health['status']}")
    st.sidebar.text(f"DB: {health['db']}")
    st.sidebar.text(f"LLM: {health['llm_provider']}")
else:
    st.sidebar.error(f"API unreachable: {herr}")

import streamlit as st
from pathlib import Path
import os

from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy import MetaData, Table, Column, String, Integer, Float, create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Redshift's catalog (forked from PG 8.0.2) lacks pg_class.relpersistence,
# which SQLAlchemy 2.x's temp-table reflection query relies on.
# Redshift has no session-local temp table concept this check assumes,
# so it's safe to report none and skip the broken query entirely.
PGDialect.get_temp_table_names = lambda self, connection, schema=None, **kw: []

file_path_icon = Path(__file__).parent.parent / "Logos32.32px.png"
st.set_page_config(page_title="Business Chatbot", page_icon=file_path_icon, layout="wide")
file_path_logo = Path(__file__).parent.parent / "poweredbymatedata.png"
st.image(file_path_logo, width=400)


@st.cache_resource(show_spinner="Connecting to Redshift and preparing the agent...")
def get_sql_agent(model_name: str, temperature: float):
    redshift_host = os.getenv("REDSHIFT_HOST")
    redshift_port = int(os.getenv("REDSHIFT_PORT", 5439))
    redshift_db = os.getenv("REDSHIFT_DB", "dev")
    redshift_user = os.getenv("REDSHIFT_USER")
    redshift_pass = os.getenv("REDSHIFT_PASS")

    missing = [
        name
        for name, val in [
            ("REDSHIFT_HOST", redshift_host),
            ("REDSHIFT_USER", redshift_user),
            ("REDSHIFT_PASS", redshift_pass),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    redshift_uri = (
        f"redshift+psycopg2://{redshift_user}:{redshift_pass}"
        f"@{redshift_host}:{redshift_port}/{redshift_db}"
    )

    engine = create_engine(redshift_uri)
    metadata = MetaData()

    # Define table columns explicitly (no autoload_with=engine)
    Table(
        "coffee_sales",
        metadata,
        Column("transaction_id", Integer, primary_key=True),
        Column("product_category", String),
        Column("amount", Float),
        schema="public",
    )

    # Pass custom SQL DDL description to bypass reflection entirely
    db = SQLDatabase(
        engine=engine,
        metadata=metadata,
        include_tables=["coffee_sales"],
        sample_rows_in_table_info=0,
        custom_table_info={
            "coffee_sales": """
        CREATE TABLE public.coffee_sales (
            sale_date DATE,
            sale_datetime TIMESTAMP,
            hour_of_day SMALLINT,
            cash_type VARCHAR(10),
            card_number VARCHAR(50),
            money NUMERIC(10, 2),
            coffee_name VARCHAR(100),
            time_of_day VARCHAR(20),
            weekday VARCHAR(10),
            month_name VARCHAR(15),
            weekday_sort SMALLINT,
            month_sort SMALLINT
        );
        """
        },
    )

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        verbose=True,
    )
    return agent_executor


with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox(
        "Model",
        options=[
            "gemini-3.1-flash-lite"
        ],
        index=0,
    )

    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.0, step=0.1,
        help="Kept low by default for more deterministic SQL generation.",
    )

    show_reasoning = st.checkbox("Show agent reasoning", value=False)

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you with your coffee sales data?"}
    ]

# Render historical chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            agent_executor = get_sql_agent(model, temperature)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({"input": prompt})
                answer = response["output"]
            except Exception as e:
                answer = f"Sorry, I ran into an error answering that: {e}"

        if show_reasoning and isinstance(response, dict) and "intermediate_steps" in response:
            with st.expander("🔍 Agent reasoning"):
                for step in response["intermediate_steps"]:
                    st.write(step)

        st.write(answer[0]['text'])

    # Save complete response to session state
    st.session_state.messages.append({"role": "assistant", "content": answer[0]['text']})
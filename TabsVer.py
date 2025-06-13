# requirements:
# pip install streamlit mysql-connector-python python-dotenv langchain openai pandas

import os
import mysql.connector
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import re
import uuid
import json
import pandas as pd

load_dotenv()

MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI DB Analyst", layout="wide")
st.title("🧠 Conversational AI Data Analyst")

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "db_setup_done" not in st.session_state:
    st.session_state.db_setup_done = False

st.sidebar.title("💬 Chats")

delete_disabled = st.session_state.current_chat_id is None or st.session_state.current_chat_id == "➕ New Chat"
if st.sidebar.button("🗑️ Delete Current Chat", disabled=delete_disabled):
    chat_id = st.session_state.current_chat_id
    if chat_id and chat_id in st.session_state.conversations:
        try:
            conn = st.session_state.conversations[chat_id]["conn"]
            db_name = st.session_state.conversations[chat_id]["db_name"]
            cursor = conn.cursor()
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
            conn.close()
        except:
            pass
        del st.session_state.conversations[chat_id]
        st.session_state.current_chat_id = None
        st.rerun()

chat_ids = list(st.session_state.conversations.keys())
selected_chat = st.sidebar.radio(
    "Select a chat:",
    options=chat_ids + ["➕ New Chat"],
    index=chat_ids.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in chat_ids else len(chat_ids),
)

if selected_chat == "➕ New Chat":
    uploaded_file = st.file_uploader("Upload your MySQL .sql file", type=["sql"])

    if uploaded_file and not st.session_state.db_setup_done:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        db_name = f"temp_db_{uuid.uuid4().hex[:8]}"
        conn = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD)
        cursor = conn.cursor()

        try:
            cursor.execute(f"CREATE DATABASE {db_name}")
            conn.database = db_name
            cursor.execute("SET foreign_key_checks = 0;")
            for command in re.split(r';\s*\n', content):
                if command.strip():
                    try:
                        cursor.execute(command)
                    except:
                        continue
            cursor.execute("SET foreign_key_checks = 1;")
            conn.commit()

            file_base_name = os.path.splitext(uploaded_file.name)[0]
            chat_title = file_base_name.strip() + " Chat"

            st.session_state.conversations[chat_title] = {
                "db_name": db_name,
                "conn": conn,
                "schema": "",
                "memory": ConversationBufferMemory(return_messages=True),
                "chat_history": []
            }
            st.session_state.current_chat_id = chat_title
            st.session_state.db_setup_done = True
            st.rerun()

        except Exception as e:
            st.error(f"Failed to set up DB: {e}")
else:
    st.session_state.current_chat_id = selected_chat
    st.session_state.db_setup_done = False

if st.session_state.current_chat_id:
    convo = st.session_state.conversations[st.session_state.current_chat_id]
    conn = convo["conn"]

    if not convo["schema"]:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        schema = ""
        for (table,) in tables:
            if table == "query_history":
                continue
            schema += f"\nTable: {table}\n"
            cursor.execute(f"DESCRIBE {table};")
            columns = cursor.fetchall()
            for col in columns:
                schema += f" - {col[0]} ({col[1]})\n"
        convo["schema"] = schema.strip()

    with st.sidebar.expander("📚 Database Schema", expanded=False):
        st.text(convo["schema"])

    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful data analyst AI. Use the following database schema to answer questions by generating correct SQL queries.\n\nSchema:\n{convo['schema']}\n\nOnly use the columns and tables shown in the schema above. Do not guess or make up columns."""),
        ("user", "{input}"),
    ])
    chain = prompt | llm

    def run_sql_query(conn, query):
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            return {"columns": cols, "rows": rows}
        except Exception as e:
            return {"error": str(e)}

    user_input = st.chat_input("Ask a question about your data")
    if user_input:
        with st.spinner("Thinking..."):
            history_text = "\n".join([f"User: {u}\nAI: {a}" for u, a in convo["chat_history"]])
            context_prompt = f"""
                Previous Conversation:
                {history_text}

                New Question: {user_input}

                Provide the SQL and answer.
                """
            llm_response = chain.invoke({"input": context_prompt}).content.strip()
            sql_match = re.search(r"```sql\n(.*?)```", llm_response, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()
                result = run_sql_query(conn, sql_query)
                preview_rows = result["rows"][:] if "rows" in result else []
                if "error" not in result:
                    summary_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant who explains database results in plain English. Your job is to convert the DB result into a simple, understandable and short summary. Don't give any headings or titles, just a concise summary."),
                        ("user", f"""SQL Query:\n{sql_query}\n\nColumns: {result['columns']}\nFirst Rows: {preview_rows}""")
                    ])
                    summary_chain = summary_prompt | llm
                    summary = summary_chain.invoke({}).content.strip()
                else:
                    summary = "Unable to summarize due to SQL error."
            else:
                sql_query = "Not found"
                result = {"error": "Could not find SQL in response."}
                summary = "No summary available due to SQL extraction failure."

            convo["chat_history"].append((user_input, summary))

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                st.markdown(llm_response)
                st.code(sql_query, language="sql")
                if "error" in result:
                    st.error("SQL Error: " + result["error"])
                else:
                    st.info("SQL query result:")
                    st.dataframe(result["rows"], use_container_width=True)
                    st.success("**Plain English Summary:** " + summary)

                    # --- Chart Decision Logic ---
                    chart_prompt = ChatPromptTemplate.from_messages([
                        ("system", 
                        """You are a data visualization assistant. Your job is to determine whether a bar or line chart can be generated from the given SQL result.

                    Strict rules:
                    1. Only return a chart if there are at least 2 columns AND at least 3 rows.
                    2. Only use "bar" if the X-axis is a category (like name, product, department, month, etc.).
                    3. Use "line" only if the X-axis is a numeric or date/time column that increases (like year, date, time, or index).
                    4. Never use pie charts or other types.
                    5. If chart is not meaningful or structure doesn't fit, return: {{"chart_type": "none"}}.

                    Always respond in one of the following formats:
                    - {{"chart_type": "bar", "x_axis": "month", "y_axis": "sales"}}
                    - {{"chart_type": "line", "x_axis": "date", "y_axis": "visits"}}
                    - {{"chart_type": "none"}}

                    Do not add any extra explanation."""
                        ),
                        ("user", f"""Here is the SQL result:
                    Columns: {result['columns']}
                    Rows: {preview_rows}

                    Return JSON only:""")
                    ])



                    chart_chain = chart_prompt | llm
                    chart_response = chart_chain.invoke({}).content.strip()

                    try:
                        chart_meta = json.loads(chart_response)
                        if chart_meta.get("chart_type") in ["bar", "line"]:
                            df = pd.DataFrame(result["rows"], columns=result["columns"])
                            chart_data = df[[chart_meta["x_axis"], chart_meta["y_axis"]]].dropna()
                            st.info(f"📊 {chart_meta['chart_type'].capitalize()} Chart:")
                            if chart_meta["chart_type"] == "bar":
                                st.bar_chart(chart_data.set_index(chart_meta["x_axis"]))
                            else:
                                st.line_chart(chart_data.set_index(chart_meta["x_axis"]))
                        else:
                            st.warning("📉 Chart cannot be generated for this output.")
                    except Exception:
                        st.warning("📉 Chart could not be created due to parsing error.")

    st.sidebar.subheader("Conversation History")
    for u, a in convo["chat_history"]:
        st.sidebar.markdown(f"**You:** {u}")
        st.sidebar.markdown(f"**AI:** {a}")

    if st.sidebar.button("🗑️ Clear History"):
        convo["chat_history"] = []


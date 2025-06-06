# requirements:
# pip install streamlit mysql-connector-python python-dotenv langchain openai

import os
import mysql.connector
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import re
import uuid

load_dotenv()

# Connect to MySQL server (without DB selected)
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Streamlit app
st.set_page_config(page_title="AI DB Analyst", layout="wide")
st.title("🧠 Conversational AI Data Analyst")

# --- Session State Setup ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "db_setup_done" not in st.session_state:
    st.session_state.db_setup_done = False

# Sidebar tab selection
st.sidebar.title("💬 Chats")

# Delete chat button
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

            # Use file name as chat title
            file_base_name = os.path.splitext(uploaded_file.name)[0]
            chat_title = file_base_name.strip() + " Chat"

            # Create conversation
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

    # --- Schema ---
    if not convo["schema"]:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        schema = ""
        for (table,) in tables:
            if table == "query_history":
                continue  # skip custom history table
            schema += f"\nTable: {table}\n"
            cursor.execute(f"DESCRIBE {table};")
            columns = cursor.fetchall()
            for col in columns:
                schema += f" - {col[0]} ({col[1]})\n"
        convo["schema"] = schema.strip()

    # --- LLM Setup ---
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

    st.sidebar.subheader("Conversation History")
    for u, a in convo["chat_history"]:
        st.sidebar.markdown(f"**You:** {u}")
        st.sidebar.markdown(f"**AI:** {a}")

    if st.sidebar.button("🗑️ Clear History"):
        convo["chat_history"] = []

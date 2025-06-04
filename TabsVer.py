# requirements:
# pip install streamlit mysql-connector-python python-dotenv langchain openai

import os
import mysql.connector
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import re
import tempfile
import uuid

load_dotenv()

# Connect to MySQL server (without DB selected)
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Setup Streamlit ---
st.set_page_config(page_title="AI DB Analyst", layout="wide")

st.title("🧠 Conversational AI Data Analyst")
st.markdown("Ask natural language questions about your uploaded MySQL database.")

# --- Tabbed Conversations ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Default"

# Sidebar: Tab management
st.sidebar.title("🗂️ Conversations")
existing_tabs = list(st.session_state.conversations.keys()) or ["Default"]
st.session_state.active_tab = st.sidebar.selectbox("Select Conversation", existing_tabs)

new_tab_name = st.sidebar.text_input("New Conversation Name")
if st.sidebar.button("➕ Add Conversation") and new_tab_name:
    if new_tab_name not in st.session_state.conversations:
        st.session_state.conversations[new_tab_name] = []
        st.session_state.active_tab = new_tab_name

if st.sidebar.button("🗑️ Delete Active Conversation") and st.session_state.active_tab != "Default":
    del st.session_state.conversations[st.session_state.active_tab]
    st.session_state.active_tab = "Default"

# Upload SQL file
db_name = f"temp_db_{uuid.uuid4().hex[:8]}"
uploaded_file = st.file_uploader("Upload your MySQL .sql file", type=["sql"])

# Function to load SQL into temporary DB
def initialize_temp_db(sql_content):
    conn = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD)
    cursor = conn.cursor()
    try:
        cursor.execute(f"CREATE DATABASE {db_name}")
        conn.database = db_name
        for command in sql_content.decode("utf-8", errors="ignore").split(";"):
            if command.strip():
                try:
                    cursor.execute(command)
                except:
                    continue  # ignore minor SQL errors
        conn.commit()
        return conn
    except Exception as e:
        return str(e)

# Get schema info from DB
def get_db_schema(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        schema_info = ""
        for (table,) in tables:
            schema_info += f"\nTable: {table}\n"
            cursor.execute(f"DESCRIBE {table};")
            columns = cursor.fetchall()
            for col in columns:
                schema_info += f" - {col[0]} ({col[1]})\n"
        return schema_info.strip()
    except Exception as e:
        return f"Error getting schema: {str(e)}"

# Execute query
def run_sql_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        return {"columns": cols, "rows": rows}
    except Exception as e:
        return {"error": str(e)}

if uploaded_file:
    with st.spinner("Setting up your database..."):
        conn = initialize_temp_db(uploaded_file.read())

    if isinstance(conn, str):
        st.error(f"Failed to set up DB: {conn}")
    else:
        schema = get_db_schema(conn)
        memory = ConversationBufferMemory(return_messages=True)
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful data analyst AI. Use the following database schema to answer questions by generating correct SQL queries.\n\nSchema:\n{schema}\n\nOnly use the columns and tables shown in the schema above. Do not guess or make up columns."""),
            ("user", "{input}"),
        ])

        chain = prompt | llm

        user_input = st.chat_input("Ask a question about your data")

        if user_input:
            with st.spinner("Thinking..."):
                chat_history = st.session_state.conversations.get(st.session_state.active_tab, [])
                history_text = "\n".join([f"User: {u}\nAI: {a}" for u, a in chat_history])
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
                    preview_rows = result["rows"][:5] if "rows" in result else []

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

                chat_history.append((user_input, summary))
                st.session_state.conversations[st.session_state.active_tab] = chat_history

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

        st.sidebar.markdown("---")
        st.sidebar.title("💬 Chat History")
        for u, a in st.session_state.conversations.get(st.session_state.active_tab, []):
            st.sidebar.markdown(f"**You:** {u}")
            st.sidebar.markdown(f"**AI:** {a}")

        # Drop temp DB after session ends
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"DROP DATABASE {db_name}")
                conn.close()
            except:
                pass
else:
    st.info("Upload a .sql MySQL dump file to begin.")

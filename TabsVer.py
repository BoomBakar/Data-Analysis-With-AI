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

# --- Constants ---
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Session Init ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # {chat_id: {db_name, conn, memory, history}}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# --- Page Setup ---
st.set_page_config(page_title="AI DB Analyst", layout="wide")
st.title("🧠 Conversational AI Data Analyst")

# --- Upload Flow (New Chat) ---
if st.session_state.active_chat is None:
    uploaded_file = st.file_uploader("Upload your MySQL .sql file to begin", type=["sql"])

    def initialize_temp_db(sql_content):
        chat_id = f"chat_{uuid.uuid4().hex[:8]}"
        db_name = f"tempdb_{uuid.uuid4().hex[:8]}"
        try:
            conn = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE {db_name}")
            conn.database = db_name
            for command in sql_content.decode("utf-8", errors="ignore").split(";"):
                if command.strip():
                    try:
                        cursor.execute(command)
                    except:
                        pass  # skip bad SQL
            conn.commit()
            return chat_id, db_name, conn
        except Exception as e:
            return None, None, str(e)

    if uploaded_file:
        with st.spinner("Setting up your database and chat..."):
            chat_id, db_name, conn = initialize_temp_db(uploaded_file.read())
            if isinstance(conn, str):
                st.error(f"Failed: {conn}")
            else:
                # Get schema
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES;")
                schema = ""
                for (table,) in cursor.fetchall():
                    schema += f"\nTable: {table}\n"
                    cursor.execute(f"DESCRIBE {table};")
                    for col in cursor.fetchall():
                        schema += f" - {col[0]} ({col[1]})\n"

                # Setup memory + llm + prompt
                memory = ConversationBufferMemory(return_messages=True)
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are a helpful data analyst AI. Use the following database schema to answer questions by generating correct SQL queries.\n\nSchema:\n{schema}\n\nOnly use the columns and tables shown in the schema above. Do not guess or make up columns."""),
                    ("user", "{input}"),
                ])
                chain = prompt | llm

                st.session_state.conversations[chat_id] = {
                    "db_name": db_name,
                    "conn": conn,
                    "memory": memory,
                    "llm": llm,
                    "schema": schema,
                    "chain": chain,
                    "history": []
                }
                st.session_state.active_chat = chat_id
                st.rerun()

# --- Active Chat Flow ---
if st.session_state.active_chat:
    chat_id = st.session_state.active_chat
    chat = st.session_state.conversations[chat_id]
    conn = chat["conn"]
    chain = chat["chain"]
    memory = chat["memory"]
    schema = chat["schema"]
    llm = chat["llm"]

    user_input = st.chat_input("Ask a question about your data")
    if user_input:
        with st.spinner("Thinking..."):
            history_text = "\n".join([f"User: {u}\nAI: {a}" for u, a in chat["history"]])
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
                try:
                    cursor = conn.cursor()
                    cursor.execute(sql_query)
                    rows = cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description]
                    preview = rows[:5]
                    summary_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant who explains database results in plain English. Your job is to convert the DB result into a simple, understandable and short summary. Don't give any headings or titles, just a concise summary."),
                        ("user", f"SQL Query:\n{sql_query}\n\nColumns: {cols}\nFirst Rows: {preview}")
                    ])
                    summary = (summary_prompt | llm).invoke({}).content.strip()
                except Exception as e:
                    summary = f"SQL Error: {str(e)}"
                    sql_query = "Query Failed"
                    rows = []
                    cols = []
            else:
                sql_query = "Not found"
                summary = "No SQL found in response"
                rows = []
                cols = []

            chat["history"].append((user_input, summary))

            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                st.markdown(llm_response)
                st.code(sql_query, language="sql")
                if rows:
                    st.dataframe(rows, use_container_width=True)
                    st.success("**Plain English Summary:** " + summary)
                else:
                    st.warning(summary)

# --- Sidebar with Chat Tabs ---
st.sidebar.title("🗂️ Conversations")
for cid, data in st.session_state.conversations.items():
    label = f"{data['db_name']} Chat"
    if st.sidebar.button(label, key=cid):
        st.session_state.active_chat = cid
        st.rerun()

if st.sidebar.button("➕ New Chat"):
    st.session_state.active_chat = None
    st.rerun()

if st.sidebar.button("🗑️ Delete Active Chat"):
    if st.session_state.active_chat:
        chat = st.session_state.conversations.pop(st.session_state.active_chat)
        try:
            cursor = chat["conn"].cursor()
            cursor.execute(f"DROP DATABASE {chat['db_name']}")
            chat["conn"].close()
        except:
            pass
        st.session_state.active_chat = None
        st.rerun()

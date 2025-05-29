
import os
import mysql.connector
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import re

load_dotenv()

# MySQL DB connection config
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# Function to get schema info
def get_db_schema():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
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

        cursor.close()
        conn.close()
        return schema_info.strip()
    except Exception as e:
        return f"Error getting schema: {str(e)}"

# Function to execute query and return results
def run_sql_query(query):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        return {"columns": cols, "rows": rows}
    except Exception as e:
        return {"error": str(e)}

# Get schema once
schema = get_db_schema()

# Set up memory and LLM
memory = ConversationBufferMemory(return_messages=True)
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a helpful data analyst AI. Use the following database schema to answer questions by generating correct SQL queries.

Schema:
{schema}

Only use the columns and tables shown in the schema above. Do not guess or make up columns.
"""),
    ("user", "{input}"),
])

# Chain (latest way)
chain = prompt | llm

# Streamlit UI
st.set_page_config(page_title="AI DB Analyst", layout="wide")
st.title("🧠 Conversational AI Data Analyst")
st.markdown("Ask natural language questions about your database.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about your data")

if user_input:
    with st.spinner("Thinking..."):
        # Combine with schema and context
        history_text = "\n".join([f"User: {u}\nAI: {a}" for u, a in st.session_state.chat_history])
        context_prompt = f"""
Previous Conversation:
{history_text}

New Question: {user_input}

Provide the SQL and answer.
"""
        llm_response = chain.invoke({"input": context_prompt}).content.strip()

        # Try to extract SQL block
        sql_match = re.search(r"```sql\n(.*?)```", llm_response, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
            result = run_sql_query(sql_query)

            preview_rows = result["rows"][:5] if "rows" in result else []

            # --- NEW: Summarize result using another LLM call ---
            if "error" not in result:
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant who explains database results in plain English."
                    "Your job is to convert the DB result into a simple, understandable and short summary."
                    "Don't give any headings or titles, just a concise summary."),
                    ("user", f"""Given the SQL query and the first few result rows, explain the result in simple terms.
                     Your job is to convert the DB result into a simple, understandable and short summary.
                     Do not give any headings or titles, just a concise summary.

                    SQL Query:
                    {sql_query}

                    Columns: {result['columns']}
                    First Rows: {preview_rows}
                    """)
                ])
                summary_chain = summary_prompt | llm
                summary = summary_chain.invoke({}).content.strip()
            else:
                summary = "Unable to summarize due to SQL error."
        else:
            sql_query = "Not found"
            result = {"error": "Could not find SQL in response."}
            summary = "No summary available due to SQL extraction failure."

        # Save to memory
        st.session_state.chat_history.append((user_input, summary))

        # Display
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

# Show history
st.sidebar.title("💬 Chat History")
for u, a in st.session_state.chat_history:
    st.sidebar.markdown(f"**You:** {u}")
    st.sidebar.markdown(f"**AI:** {summary}")
    

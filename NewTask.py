# import os
# import mysql.connector
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableSequence
# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory

# load_dotenv()

# # MySQL DB connection config
# DB_CONFIG = {
#     "host": os.getenv("DB_HOST"),
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASSWORD"),
#     "database": os.getenv("DB_NAME")
# }

# # Function to get schema info
# def get_db_schema():
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)
#         cursor = conn.cursor()
#         cursor.execute("SHOW TABLES;")
#         tables = cursor.fetchall()

#         schema_info = ""
#         for (table,) in tables:
#             schema_info += f"\nTable: {table}\n"
#             cursor.execute(f"DESCRIBE {table};")
#             columns = cursor.fetchall()
#             for col in columns:
#                 schema_info += f" - {col[0]} ({col[1]})\n"

#         cursor.close()
#         conn.close()
#         return schema_info.strip()
#     except Exception as e:
#         return f"Error getting schema: {str(e)}"

# # Function to execute query and return results
# def run_sql_query(query):
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)
#         cursor = conn.cursor()
#         cursor.execute(query)
#         rows = cursor.fetchall()
#         cols = [desc[0] for desc in cursor.description]
#         cursor.close()
#         conn.close()
#         return {"columns": cols, "rows": rows}
#     except Exception as e:
#         return {"error": str(e)}

# # Get schema once
# schema = get_db_schema()

# # Set up memory and LLM
# memory = ConversationBufferMemory(return_messages=True)
# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

# # Prompt template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", f"""You are a helpful data analyst AI. Use the following database schema to answer questions by generating correct SQL queries.

# Schema:
# {schema}

# Only use the columns and tables shown in the schema above. Do not guess or make up columns.
# """),
#     ("user", "{input}"),
# ])

# # Chain (latest way)
# chain = prompt | llm

# # Streamlit UI
# st.set_page_config(page_title="AI DB Analyst", layout="wide")
# st.title("🧠 Conversational AI Data Analyst")
# st.markdown("Ask natural language questions about your database.")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_input = st.chat_input("Ask a question about your data")

# if user_input:
#     with st.spinner("Thinking..."):
#         # Combine with schema and context
#         history_text = "\n".join([f"User: {u}\nAI: {a}" for u, a in st.session_state.chat_history])
#         context_prompt = f"""
# Previous Conversation:
# {history_text}

# New Question: {user_input}

# Provide the SQL and answer.
# """
#         llm_response = chain.invoke({"input": context_prompt}).content.strip()

#         # Try to extract SQL block
#         import re
#         sql_match = re.search(r"```sql\n(.*?)```", llm_response, re.DOTALL)
#         if sql_match:
#             sql_query = sql_match.group(1).strip()
#             result = run_sql_query(sql_query)
#         else:
#             sql_query = "Not found"
#             result = {"error": "Could not find SQL in response."}

#         # Save to memory
#         st.session_state.chat_history.append((user_input, llm_response))

#         # Display
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         with st.chat_message("assistant"):
#             st.markdown(llm_response)
#             st.code(sql_query, language="sql")
#             if "error" in result:
#                 st.error("SQL Error: " + result["error"])
#             else:
#                 st.success("Top rows:")
#                 st.dataframe(result["rows"], use_container_width=True)

# # Show history
# st.sidebar.title("💬 Chat History")
# for u, a in st.session_state.chat_history:
#     st.sidebar.markdown(f"**You:** {u}")
#     st.sidebar.markdown(f"**AI:** {a.splitlines()[0][:100]}...")

# conversational_ai_db.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv

load_dotenv()

db_uri = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
db = SQLDatabase.from_uri(db_uri)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt for SQL generation with schema awareness
sql_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data analyst AI. Use the given database schema to answer user questions by generating correct SQL queries."
    "Only return SQL queries, do not include any explanations or additional text. Don't even write sql at start. Just return the SQL query directly."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Schema: {schema}\nUser Question: {question}")
])

# Prompt for summarizing the final result into clean natural language
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that summarizes query results into short, clear answers."),
    ("human", "Based on the result rows: {rows} for the question: '{question}', give a plain English answer.")
])

# SQL generation chain
sql_chain = RunnableMap({
    "schema": lambda x: schema_str,
    "question": lambda x: x["question"],
    "chat_history": lambda x: memory.chat_memory.messages  # or memory.load_memory_variables({})
}) | sql_prompt | llm

# Summary generation chain
summary_chain = summary_prompt | llm

# Get DB schema string
schema_str = db.get_table_info()

# Streamlit UI
st.set_page_config(page_title="Conversational Data Analyst AI", layout="wide")
st.title("🧠 Conversational AI Analyst")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about your database...")
if user_input:
    # Append user message to memory
    memory.chat_memory.add_user_message(user_input)

    with st.spinner("Thinking..."):
        try:
            # Step 1: Generate SQL from question
            sql_query = sql_chain.invoke({
                "schema": schema_str,
                "question": user_input,
                "chat_history": memory.chat_memory.messages
            }).content

            # Step 2: Execute SQL
            result = db.run(sql_query)
            top_rows = db.run(sql_query + " LIMIT 5")

            # Step 3: Generate final summary
            summary = summary_chain.invoke({
                "rows": top_rows,
                "question": user_input
            }).content

            # Append AI message (only summary shown in chat history)
            memory.chat_memory.add_ai_message(summary)

            # Append to session history
            st.session_state.chat_history.append((user_input, summary))

            # Display top-down
            st.subheader("Answer")
            st.success(summary)

            st.subheader("SQL Query")
            st.code(sql_query, language="sql")

            st.subheader("Top Rows")
            st.code(top_rows, language="text")

        except Exception as e:
            st.error(f"Error: {e}")

# Chat History Sidebar (summary only)
st.sidebar.title("💬 Chat History")
for i, (user, ai) in enumerate(st.session_state.chat_history):
    st.sidebar.markdown(f"**You:** {user}")
    st.sidebar.markdown(f"**AI:** {ai}")

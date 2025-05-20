import mysql.connector
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

GROQ_API_KEY = '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'
#MODEL = 'llama-3.1-8b-instant'
MODEL = 'llama-3.3-70b-versatile'
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

db = mysql.connector.connect(**db_config)
cursor = db.cursor()
def run_query(sql):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()
    return columns, rows

def clean_sql_response(response_text):
    # Remove leading 'sql' or code blocks
    lines = response_text.strip().splitlines()
    clean_lines = [line for line in lines if not line.strip().lower().startswith('sql') and not line.strip().startswith('```')]
    return '\n'.join(clean_lines).strip().strip(';') + ';'  # Ensure semicolon at end

def natural_language_to_sql(question, db_schema_string):
    prompt = f"""
    You are a helpful assistant that converts natural language questions into valid MySQL queries.
    The following is the schema of the database:

    {db_schema_string}

    Now generate an SQL query for the question:
    "{question}"

    Only return the SQL. Do not explain anything or write anything else.
    """

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )

    raw_sql = response.json()['choices'][0]['message']['content']
    return clean_sql_response(raw_sql)


def results_to_english(question, columns, rows):
    # Convert rows to a readable string format
    table_data = [dict(zip(columns, row)) for row in rows]

    prompt = f"""You are a helpful assistant. Based on the question and the table result below, give a short, natural-language answer. Keep it short and precise.

Question: {question}
Table Result: {table_data}
Answer in plain English:"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
    )

    return response.json()['choices'][0]['message']['content'].strip()

def fetch_db_schema(cursor, db_name):
    query = """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position;
    """
    cursor.execute(query, (db_name,))
    rows = cursor.fetchall()

    schema = {}
    for table, column in rows:
        schema.setdefault(table, []).append(column)

    schema_str = ""
    for table, columns in schema.items():
        schema_str += f"Table {table}({', '.join(columns)})\n"

    return schema_str.strip()


st.title("🚗 Car Database Q&A with AI")

db_name = 'cars'  # replace with your DB name
cursor = db.cursor()

# Fetch schema only once per app run
db_schema_string = fetch_db_schema(cursor, db_name)

# Get user input
question = st.text_input("Ask a question about the database:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        try:
            sql = natural_language_to_sql(question, db_schema_string)
            st.code(sql, language="sql")

            columns, results = run_query(sql)
            st.subheader("Query Results:")
            df = pd.DataFrame(results, columns=columns)
            st.dataframe(df)


            summary = results_to_english(question, columns, results)
            st.subheader("🗣️ Plain English Answer:")
            st.write(summary)

        except Exception as e:
            st.error(f"Error: {e}")
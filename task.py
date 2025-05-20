import mysql.connector
import requests
import streamlit as st
import pandas as pd


GROQ_API_KEY = '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'
#MODEL = 'llama-3.1-8b-instant'
MODEL = 'llama-3.3-70b-versatile'
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'XvrLhr123',
    'database': 'cars'
}

# Step 1: Connect to MySQL
def run_query(sql):
    conn = mysql.connector.connect(**MYSQL_CONFIG)
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

def natural_language_to_sql(question):
    prompt = f"""You are a SQL expert. Based on the following MySQL database schema:

Tables:
- manufacturers(id, name, country)
- owners(id, name, email, phone)
- cars(id, model, year, price, manufacturer_id, owner_id)
- service_records(id, car_id, service_date, description, cost)

Translate the following question into a correct SQL query:
Question: {question}
Only output the SQL query, no explanations, no markdown, no 'sql' prefix, just the raw SQL."""

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


# Step 3: Main flow
def ask_question(question):
    sql = natural_language_to_sql(question)
    print("Generated SQL:\n", sql)
    try:
        columns, results = run_query(sql)
        print("\nRaw Results:")
        print(columns)
        for row in results:
            print(row)

        english_summary = results_to_english(question, columns, results)
        print("\n🗣️ Answer in Plain English:")
        print(english_summary)

    except Exception as e:
        print("❌ Error running SQL:", e)

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

# Example use
# ask_question("List all cars with service records in 2021.")
# #ask_question("Who are the owners of the cars manufactured in Germany?")
# #ask_question("Who are the owners of the lamborghini cars?")
# # ask a complex question
# ask_question("What is the average price of cars manufactured by Toyota in 2020?")
# ask_question("What is the total cost of all service records for cars owned by Alice Smith?")
st.title("🚗 Car Database Q&A with AI")

question = st.text_input("Ask a question about the car database:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        try:
            sql = natural_language_to_sql(question)
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
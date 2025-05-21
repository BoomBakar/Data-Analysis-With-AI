import mysql.connector
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import plotly.express as px

load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#MODEL = 'llama-3.1-8b-instant'
#MODEL = 'llama-3.3-70b-versatile'
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
Answer in plain English. Keep it precise and concise:"""

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

def get_viz_plan(question, df):
    sample = df.head(5).to_dict(orient="records")
    print(sample)
    prompt = f"""
    You are a data analyst. Based on this question and data sample, suggest a visualization.
    Question: {question}
    Sample: {sample}
    Example Return JSON: {{ "chart_type": "bar", "x_axis": "brand", "y_axis": "car_count" }}.
    If not suitable for chart, return {{"chart_type": "none"}}.
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
            "temperature": 0.5
        }
    )
    import json
    try:
        return json.loads(response.json()['choices'][0]['message']['content'].strip())
    except:
        return {"chart_type": "none"}

# Render chart
def render_chart(df, plan):
    chart_type = plan.get("chart_type")
    x = plan.get("x_axis")
    y = plan.get("y_axis")
    if chart_type == "bar":
        st.plotly_chart(px.bar(df, x=x, y=y))
    elif chart_type == "line":
        st.plotly_chart(px.line(df, x=x, y=y))
    elif chart_type == "pie":
        st.plotly_chart(px.pie(df, names=x, values=y))
    else:
        st.write("No chart applicable.")
def draw_chart(columns, rows):
    if not rows or not columns:
        st.warning("No data to display.")
        return

    df = pd.DataFrame(rows, columns=columns)

    # Try to detect numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    if len(numeric_cols) == 0:
        st.info("No suitable chart found (no numeric data).")
        return

    # If there's one numeric column and one categorical, show a bar chart
    if len(numeric_cols) == 1 and len(non_numeric_cols) == 1:
        st.bar_chart(df.set_index(non_numeric_cols[0])[numeric_cols[0]])
    # If two numeric columns, use a line chart
    elif len(numeric_cols) >= 2:
        st.line_chart(df[numeric_cols])
    # If more than one categorical, fallback to table or mention unsupported
    else:
        st.info("No suitable chart found for this data.")

st.title("🚗 Car Database Q&A with AI")

db_name = 'cars'  # replace with your DB name
cursor = db.cursor()

# Fetch schema only once per app run
db_schema_string = fetch_db_schema(cursor, db_name)

# Get user input
question = st.text_input("Ask a question about the database:")

# Dropdown for selecting the model
MODEL = st.selectbox(
    "Choose an LLM model:",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile","gemma2-9b-it"],
    index=0  # default selection
)

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

            # draw_chart(columns, results)
            plan = get_viz_plan(question, df)
            if plan.get("chart_type") != "none":
                st.markdown("### 📊 Suggested Visualization")
                render_chart(df, plan)
            else:
                st.info("No suitable chart found for this data.")


        except Exception as e:
            st.error(f"Error: {e}")
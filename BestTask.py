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

def fetch_db_schema(cursor, db_name, exclude_tables=None):
    if exclude_tables is None:
        exclude_tables = []

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
        if table in exclude_tables:
            continue
        schema.setdefault(table, []).append(column)

    schema_str = ""
    for table, columns in schema.items():
        schema_str += f"Table {table}({', '.join(columns)})\n"

    return schema_str.strip()


def get_viz_plan(question, df):
    sample = df.head(5).to_dict(orient="records")
    prompt = f"""
    You are a data analyst. Based on this question and data sample, suggest a visualization, either bar or line chart only.
    Question: {question}
    Sample: {sample}
    Only use the actual column names visible in the sample.
    Return only valid JSON like: {{ "chart_type": "bar", "x_axis": "brand", "y_axis": "price" }}
    Only return JSON, no extra text. If no chart applies, return: {{ "chart_type": "none" }}"""
    
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

    if x not in df.columns or (chart_type != "pie" and y not in df.columns):
        st.warning("Cannot generate chart: specified columns not found in data.")
        return

    if chart_type == "bar":
        st.plotly_chart(px.bar(df, x=x, y=y))
    elif chart_type == "line":
        st.plotly_chart(px.line(df, x=x, y=y))
    
    else:
        st.write("No chart applicable.")

def save_to_history(question, sql_query, answer, chart_plan):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    chart_type = chart_plan.get("chart_type", "none")
    x_axis = chart_plan.get("x_axis", None)
    y_axis = chart_plan.get("y_axis", None)

    query = """
        INSERT INTO query_history (question, sql_query, answer, chart_type, x_axis, y_axis)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (question, sql_query, answer, chart_type, x_axis, y_axis))
    conn.commit()
    cursor.close()
    conn.close()


st.title("🚗 Car Database Q&A with AI")

db_name = os.getenv("DB_NAME")  # replace with your DB name
cursor = db.cursor()

# Fetch schema only once per app run
db_schema_string = fetch_db_schema(cursor, db_name, exclude_tables=["query_history"])


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

            save_to_history(question, sql, summary, plan)
            st.success("Query saved to history.")


        except Exception as e:
            st.error(f"Error: {e}")

with st.sidebar:
    st.header("📚 Query History")
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT id, question, created_at FROM query_history ORDER BY created_at DESC LIMIT 10")
    rows = cursor.fetchall()

    for row in rows:
        qid, qtext, ts = row
        col1, col2 = st.columns([0.85, 0.15])  # Layout: question | 🗑️
        with col1:
            if st.button(f"{ts.strftime('%Y-%m-%d %H:%M')} — {qtext[:30]}...", key=f"q_{qid}"):
                cur2 = conn.cursor()
                cur2.execute("SELECT question, sql_query, answer, chart_type, x_axis, y_axis FROM query_history WHERE id = %s", (qid,))
                qrow = cur2.fetchone()
                if qrow:                      
                    st.subheader("🕘 History Result")
                    st.write("**Question:**", qrow[0])
                    st.write("**Answer:**", qrow[2])
        with col2:
            if st.button("🗑️", key=f"del_{qid}"):
                delete_query = "DELETE FROM query_history WHERE id = %s"
                cursor.execute(delete_query, (qid,))
                conn.commit()
                st.rerun()  # Refresh UI
    if st.button("🧹 Clear All History"):
        cursor = conn.cursor()
        cursor.execute("DELETE FROM query_history")
        conn.commit()
        st.success("All history deleted.")
        st.rerun()

    cursor.close()
    conn.close()
        



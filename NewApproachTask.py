# LangGraph-based Natural Language to SQL Agent with GROQ

import os
import mysql.connector
import streamlit as st
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import json

# load environment variables
load_dotenv()

# -- Configs --
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Or hardcode for local testing
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# -- Init LLM --
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# -- Streamlit UI --
st.title("🧠 LangGraph SQL Agent")
user_query = st.text_input("Ask a question about your eCommerce database:")

# -- DB Schema Helper (Excludes query_history) --
def fetch_schema():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name, column_name FROM information_schema.columns 
        WHERE table_schema = %s AND table_name != 'query_history' 
        ORDER BY table_name, ordinal_position;
    """, (DB_CONFIG["database"],))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    schema = {}
    for table, column in rows:
        schema.setdefault(table, []).append(column)

    return "\n".join([f"Table {t}({', '.join(c)})" for t, c in schema.items()])

schema_str = fetch_schema()

# --- LangGraph Nodes ---
def generate_sql_node(state):
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        You are an expert SQL assistant.
        Only use the following schema:
        {schema_str}
        Return only the SQL query.
        """),
        ("human", "{input}")
    ])
    chain = prompt | llm
    sql = chain.invoke({"input": question}).content.strip()
    return {"sql": sql, "question": question}

def execute_sql_node(state):
    sql = state["sql"]
    if "```" in sql:
        sql = sql.strip("`").replace("sql", "").strip()

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()

    df = pd.DataFrame(rows, columns=columns)
    result_text = f"Columns: {columns}\nRows: {rows[:10]}"

    return {**state, "result": result_text, "df": df}

def summarize_node(state):
    result = state["result"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following SQL result in plain English. Only write a concise and accurate summary."
        "Do not include any SQL or code."),
        ("human", "{input}")
    ])
    chain = prompt | llm
    summary = chain.invoke({"input": result}).content.strip()
    return {**state, "summary": summary}

import json

def suggest_viz_node(state):
    result = state["result"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Based on this result, suggest an appropriate chart type (e.g., bar or line) and which columns to plot.
        Return only valid JSON in the format: 
        {{ "chart_type": "bar", "x_axis": "brand", "y_axis": "price" }}
        If no chart applies, return:
        {{ "chart_type": "none" }}"""),
        ("human", "{input}")
    ])

    chain = prompt | llm
    raw_viz = chain.invoke({"input": result}).content.strip()
    # print("Suggested Visualization:", raw_viz)

    try:
        viz = json.loads(raw_viz)
    except json.JSONDecodeError:
        viz = {"chart_type": "none"}

    return {**state, "viz": viz}

def draw_chart_node(state):
    import pandas as pd
    import plotly.express as px

    viz = state.get("viz")  # already a dict
    result_text = state.get("result")

    try:
        chart_type = viz.get("chart_type", "none")
        x = viz.get("x_axis")
        y = viz.get("y_axis")

        # Extract columns and rows from result_text
        lines = result_text.split("\n")
        columns = eval(lines[0].split(":")[1].strip())
        rows = eval(lines[1].split(":")[1].strip())
        df = pd.DataFrame(rows, columns=columns)

        fig = None
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y)

        return {**state, "fig": fig}

    except Exception as e:
        print("Chart error:", e)
        return {**state, "fig": None}


# --- Build LangGraph ---
graph = StateGraph(dict)  # Use a basic dict schema for state passing
graph.add_node("generate_sql", RunnableLambda(generate_sql_node))
graph.add_node("execute_sql", RunnableLambda(execute_sql_node))
graph.add_node("summarize", RunnableLambda(summarize_node))
graph.add_node("visualize", RunnableLambda(suggest_viz_node))
graph.add_node("draw_chart", RunnableLambda(draw_chart_node))

# Edge chaining
graph.set_entry_point("generate_sql")
graph.add_edge("generate_sql", "execute_sql")
graph.add_edge("execute_sql", "summarize")
graph.add_edge("summarize", "visualize")
graph.add_edge("visualize", "draw_chart")
graph.set_finish_point("draw_chart")

app = graph.compile()

# --- Run on user query ---
if user_query:
    with st.spinner("Processing your question..."):
        output = app.invoke({"question": user_query})

        # 1. Show answer first
        st.text("Answer:")
        st.success(output["summary"])

        # 2. Show SQL query
        st.text("\nGenerated SQL:")
        st.code(output["sql"], language="sql")

        # 3. Show chart (draw_chart_node already runs this, so just make sure it's in output)
        if "fig" in output and output["fig"]:
            st.text("\nChart:")
            st.plotly_chart(output["fig"], use_container_width=True)

        # 4. Show suggested viz JSON
        st.text("\nSuggested Visualization:")
        st.info(output["viz"])

        # 5. Show top rows of result
        st.text("\nTop Rows:")
        st.code(output["result"], language="text")


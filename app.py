import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

# Streamlit app configuration
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# Constants for database options
LOCALDB = "USE_LOCALDB"#the saved database(student.db)
MYSQL = "USE_MYSQL"

# Radio options for database selection
radio_opt = ["Use SQLLite 3 Database - Student.db", "Connect to your MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB you want to chat with", options=radio_opt)

# Database connection details based on selection
if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

# Groq API key input
api_key = st.sidebar.text_input(label="Groq API Key", type="password")

# Validate inputs
if not db_uri:
    st.info("Please enter the database information and URI.")

if not api_key:
    st.info("Please add the Groq API key.")
else:
    # Initialize LLM model only if API key is provided
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    # Function to configure the database
    @st.cache_resource(ttl=7200)  # 2 hours in seconds
    def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
        try:
            if db_uri == LOCALDB:
                dbfilepath = (Path.cwd() / "student.db").absolute()  # Use current working directory
                print(f"Database path: {dbfilepath}")
                creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
                return SQLDatabase(create_engine("sqlite:///", creator=creator))
            elif db_uri == MYSQL:
                if not (mysql_host and mysql_user and mysql_password and mysql_db):
                    st.error("Please provide all MySQL connection details.")
                    st.stop()
                return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
        except Exception as e:
            st.error(f"Error configuring database: {e}")
            st.stop()

    # Configure the database
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    else:
        db = configure_db(db_uri)

    # Initialize SQLDatabaseToolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create SQL agent
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # Initialize session state for messages
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    user_query = st.chat_input(placeholder="Ask anything from the database")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # Assistant response
        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            try:
                response = agent.run(user_query, callbacks=[streamlit_callback])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
            except Exception as e:
                st.error(f"Error processing your query: {e}")
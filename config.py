import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

load_dotenv()

# LLM Configuration
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
model = ChatOpenAI(model=MODEL_NAME, temperature=0)

# Database & Persistence Configuration
DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/thesis_db")

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True}
)

checkpointer = PostgresSaver(pool)
store = PostgresStore(pool)

def init_db():
    """Initializes tables for checkpointing and long-term storage."""
    checkpointer.setup()
    store.setup()
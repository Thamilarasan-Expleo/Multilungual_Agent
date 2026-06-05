import os
import uuid
import logging
 
from dotenv import load_dotenv
import psycopg2
 
# ------------------------------------------------------------------
# Load Environment Variables
# ------------------------------------------------------------------
 
load_dotenv()
 
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
 
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
CACHE_TABLE_NAME = os.getenv("CACHE_TABLE_NAME", "multilingual_cache")
 
# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
 
logger = logging.getLogger(__name__)
 
# ------------------------------------------------------------------
# Database Connection
# ------------------------------------------------------------------
 
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
 
# ------------------------------------------------------------------
# Check Table Exists
# ------------------------------------------------------------------
 
def table_exists(schema_name: str, table_name: str) -> bool:
 
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_name = %s
        )
    """
 
    conn = None
    cur = None
 
    try:
        conn = get_db_connection()
        cur = conn.cursor()
 
        cur.execute(
            query,
            (schema_name, table_name)
        )
 
        return cur.fetchone()[0]
 
    finally:
        if cur:
            cur.close()
 
        if conn:
            conn.close()
 
# ------------------------------------------------------------------
# Create Schema and Table
# ------------------------------------------------------------------
 
def create_cache_table_if_not_exists():
 
    if table_exists(DB_SCHEMA, CACHE_TABLE_NAME):
        logger.info(
            "Table already exists: %s.%s",
            DB_SCHEMA,
            CACHE_TABLE_NAME
        )
        return
 
    create_schema_query = f"""
        CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}";
    """
 
    create_table_query = f"""
        CREATE TABLE "{DB_SCHEMA}"."{CACHE_TABLE_NAME}" (
            uuid UUID PRIMARY KEY,
            original_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
 
    conn = None
    cur = None
 
    try:
        conn = get_db_connection()
        cur = conn.cursor()
 
        cur.execute(create_schema_query)
        cur.execute(create_table_query)
 
        conn.commit()
 
        logger.info(
            "Created table: %s.%s",
            DB_SCHEMA,
            CACHE_TABLE_NAME
        )
 
    except Exception as e:
        if conn:
            conn.rollback()
 
        logger.error(str(e))
        raise
 
    finally:
        if cur:
            cur.close()
 
        if conn:
            conn.close()
 
# ------------------------------------------------------------------
# Bulk Insert Translation Cache
# ------------------------------------------------------------------
 
def insert_translation_cache(records):
 
    if not records:
        logger.info("No records to insert")
        return
 
    create_cache_table_if_not_exists()
 
    insert_query = f"""
        INSERT INTO "{DB_SCHEMA}"."{CACHE_TABLE_NAME}"
        (
            uuid,
            original_text,
            translated_text
        )
        VALUES (%s, %s, %s)
    """
 
    values = [
        (
            str(uuid.uuid4()),
            item["original_text"],
            item["translated_text"]
        )
        for item in records
    ]
 
    conn = None
    cur = None
 
    try:
        conn = get_db_connection()
        cur = conn.cursor()
 
        cur.executemany(insert_query, values)
 
        conn.commit()
 
        logger.info(
            "%s records inserted successfully",
            len(values)
        )
 
    except Exception as e:
 
        if conn:
            conn.rollback()
 
        logger.error(str(e))
        raise
 
    finally:
 
        if cur:
            cur.close()
 
        if conn:
            conn.close()
 
# ------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------
 
if __name__ == "__main__":
 
    translation_data = [
        {
            "original_text": "Der Benutzer muss sich anmelden.",
            "translated_text": "The user must log in."
        },
        {
            "original_text": "Das System muss Berichte generieren.",
            "translated_text": "The system must generate reports."
        },
        {
            "original_text": "Alle Daten müssen verschlüsselt werden.",
            "translated_text": "All data must be encrypted."
        }
    ]
   
 
    insert_translation_cache(translation_data)
 
    print("Completed successfully.")
 
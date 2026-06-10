import json
import logging
import os
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DB_SCHEMA = os.getenv("DB_SCHEMA")
CACHE_TABLE_NAME = os.getenv("CACHE_TABLE_NAME", "multilingual_cache")
SCHEMA_TABLE_NAME = os.getenv("SCHEMA_TABLE_NAME", "translation_schema")

LOG_LEVEL = os.getenv("DB_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SCHEMA: Dict[str, Any] = {
    "schema_version": "1.0",
    "original_text": "",
    "translated_text_en": "",
    "translated_text_en_file": "",
    "segments": [],
}


def get_db_connection():
    logger.info("Opening database connection to %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


@contextmanager
def db_cursor(commit: bool = False):
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def execute_query(
    query: str,
    params: Optional[Sequence[Any]] = None,
    *,
    fetchone: bool = False,
    fetchall: bool = False,
    commit: bool = False,
):
    with db_cursor(commit=commit) as cur:
        cur.execute(query, params)

        if fetchone:
            return cur.fetchone()

        if fetchall:
            return cur.fetchall()

        return None


def table_exists(schema_name: str, table_name: str) -> bool:
    """Check if a table exists in the specified schema."""
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
    """

    result = execute_query(query, (schema_name, table_name), fetchone=True)
    exists = bool(result[0]) if result else False

    logger.info("Table exists check for %s.%s: %s", schema_name, table_name, exists)
    return exists


def create_table_if_not_exists(table_name: str, create_table_query: str):
    """Generic helper to create schema and table if the table does not already exist."""
    if table_exists(DB_SCHEMA, table_name):
        logger.info("Table already exists: %s.%s", DB_SCHEMA, table_name)
        return

    create_schema_query = f'''
        CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}";
    '''

    with db_cursor(commit=True) as cur:
        cur.execute(create_schema_query)
        cur.execute(create_table_query)

    logger.info("Created table: %s.%s", DB_SCHEMA, table_name)


def create_schema_table_if_not_exists():
    """Create the translation_schema table if it does not exist."""
    create_table_query = f"""
        CREATE TABLE "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}" (
            uuid UUID PRIMARY KEY,
            version TEXT NOT NULL,
            schema JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    create_table_if_not_exists(SCHEMA_TABLE_NAME, create_table_query)


def create_cache_table_if_not_exists():
    """Create the translation cache table if it does not exist."""
    create_table_query = f"""
        CREATE TABLE "{DB_SCHEMA}"."{CACHE_TABLE_NAME}" (
            uuid UUID PRIMARY KEY,
            original_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    create_table_if_not_exists(CACHE_TABLE_NAME, create_table_query)


def initialize_database():
    """Create all required tables and seed the default schema if needed."""
    create_schema_table_if_not_exists()
    create_cache_table_if_not_exists()
    ensure_default_schema_exists()


def save_schema(version: str, schema_json: Dict[str, Any]):
    """
    Insert a schema definition into the translation_schema table.

    This behaves like an append-only save, so each call stores a new row.
    """
    create_schema_table_if_not_exists()

    insert_query = f"""
        INSERT INTO "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}"
        (
            uuid,
            version,
            schema
        )
        VALUES (%s, %s, %s)
    """

    execute_query(
        insert_query,
        (str(uuid.uuid4()), version, Json(schema_json)),
        commit=True,
    )

    logger.info("Schema version %s saved successfully", version)


def ensure_default_schema_exists():
    """Ensure the translation_schema table has a default schema record."""
    create_schema_table_if_not_exists()

    check_query = f"""
        SELECT COUNT(*)
        FROM "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}"
        WHERE version = %s
    """

    row = execute_query(check_query, ("1.0",), fetchone=True)
    count = row[0] if row else 0

    if count == 0:
        save_schema("1.0", DEFAULT_SCHEMA)
        logger.info("Default schema inserted into %s.%s", DB_SCHEMA, SCHEMA_TABLE_NAME)


def fetch_schema_by_version(version: str) -> Dict[str, Any]:
    """
    Fetch schema details by version from the translation_schema table.

    Returns the stored schema if found, otherwise a default schema template.
    """
    create_schema_table_if_not_exists()

    select_query = f"""
        SELECT schema
        FROM "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}"
        WHERE version = %s
        ORDER BY created_at DESC
        LIMIT 1
    """

    try:
        row = execute_query(select_query, (version,), fetchone=True)
        if row:
            logger.info("Schema found for version %s", version)
            return row[0]
    except Exception as e:
        logger.error("Schema fetch failed: %s", str(e))

    default_schema = DEFAULT_SCHEMA.copy()
    default_schema["schema_version"] = version
    logger.info("Returning default schema for version %s", version)
    return default_schema


def insert_translation_cache(records: List[Dict[str, str]]):
    """Insert translation cache records into the database."""
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
            item["translated_text"],
        )
        for item in records
    ]

    with db_cursor(commit=True) as cur:
        cur.executemany(insert_query, values)

    logger.info(
        "%s records inserted successfully into %s.%s",
        len(values),
        DB_SCHEMA,
        CACHE_TABLE_NAME,
    )


def fetch_translation_cache(original_texts: Iterable[str]) -> Dict[str, str]:
    """Fetch translation cache records by original texts."""
    original_texts = list(original_texts)

    if not original_texts:
        logger.info("No texts provided for cache lookup")
        return {}

    create_cache_table_if_not_exists()

    select_query = f"""
        SELECT original_text, translated_text
        FROM "{DB_SCHEMA}"."{CACHE_TABLE_NAME}"
        WHERE original_text = ANY(%s)
    """

    rows = execute_query(select_query, (original_texts,), fetchall=True)

    logger.info(
        "Cache lookup returned %s record(s) from %s.%s",
        len(rows),
        DB_SCHEMA,
        CACHE_TABLE_NAME,
    )

    return {row[0]: row[1] for row in rows}


# Backward-compatible aliases
insert_schema = save_schema
update_schema = save_schema


if __name__ == "__main__":
    initialize_database()

    translation_data = [
        {
            "original_text": "Der Benutzer muss sich anmelden.",
            "translated_text": "The user must log in.",
        },
        {
            "original_text": "Das System muss Berichte generieren.",
            "translated_text": "The system must generate reports.",
        },
        {
            "original_text": "Alle Daten müssen verschlüsselt werden.",
            "translated_text": "All data must be encrypted.",
        },
    ]

    insert_translation_cache(translation_data)
    print("Completed successfully.")
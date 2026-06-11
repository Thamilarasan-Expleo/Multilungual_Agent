import logging
import os
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json, RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool

load_dotenv()

# ---------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------
REQUIRED_ENV_VARS = [
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "DB_SCHEMA",
]

missing_env_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_env_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_env_vars)}"
    )

DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = int(os.getenv("DB_PORT", "5432").strip())
DB_NAME = os.getenv("DB_NAME", "").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_SCHEMA = os.getenv("DB_SCHEMA", "").strip()

CACHE_TABLE_NAME = os.getenv("CACHE_TABLE_NAME", "multilingual_cache").strip()
SCHEMA_TABLE_NAME = os.getenv("SCHEMA_TABLE_NAME", "translation_schema").strip()

DB_POOL_MINCONN = int(os.getenv("DB_POOL_MINCONN", "1"))
DB_POOL_MAXCONN = int(os.getenv("DB_POOL_MAXCONN", "10"))

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

_connection_pool: Optional[SimpleConnectionPool] = None


# ---------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------
def _build_connection_pool() -> SimpleConnectionPool:
    """Create the global PostgreSQL connection pool."""
    logger.info(
        "Creating database connection pool for %s:%s/%s",
        DB_HOST,
        DB_PORT,
        DB_NAME,
    )
    return SimpleConnectionPool(
        minconn=DB_POOL_MINCONN,
        maxconn=DB_POOL_MAXCONN,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def get_connection_pool() -> SimpleConnectionPool:
    """Return the singleton connection pool, creating it lazily if needed."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = _build_connection_pool()
    return _connection_pool


def get_db_connection():
    """Backward-compatible helper to get a pooled connection."""
    return get_connection_pool().getconn()


def release_db_connection(conn) -> None:
    """Return a pooled connection back to the pool."""
    if conn is not None:
        get_connection_pool().putconn(conn)


@contextmanager
def db_cursor(commit: bool = False):
    """
    Context manager that yields a database cursor from the connection pool.

    The connection is always returned to the pool.
    """
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
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
            release_db_connection(conn)


def execute_query(
    query: str,
    params: Optional[Sequence[Any]] = None,
    *,
    fetchone: bool = False,
    fetchall: bool = False,
    commit: bool = False,
):
    """Execute a SQL query and optionally fetch one or many rows."""
    with db_cursor(commit=commit) as cur:
        cur.execute(query, params)

        if fetchone:
            return cur.fetchone()

        if fetchall:
            return cur.fetchall()

        return None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def table_exists(schema_name: str, table_name: str) -> bool:
    """Check if a table exists in the specified schema."""
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        ) AS exists
    """
    row = execute_query(query, (schema_name, table_name), fetchone=True)
    exists = bool(row["exists"]) if row else False
    logger.info("Table exists check for %s.%s: %s", schema_name, table_name, exists)
    return exists


def index_exists(schema_name: str, index_name: str) -> bool:
    """Check if an index exists in the specified schema."""
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = %s
              AND indexname = %s
        ) AS exists
    """
    row = execute_query(query, (schema_name, index_name), fetchone=True)
    return bool(row["exists"]) if row else False


def create_schema_if_not_exists() -> None:
    """Create the configured schema if it does not already exist."""
    query = f'CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}";'
    execute_query(query, commit=True)
    logger.info("Ensured schema exists: %s", DB_SCHEMA)


def create_table_if_not_exists(table_name: str, create_table_query: str) -> None:
    """Create a table only if it does not already exist."""
    if table_exists(DB_SCHEMA, table_name):
        logger.info("Table already exists: %s.%s", DB_SCHEMA, table_name)
        return

    create_schema_if_not_exists()
    execute_query(create_table_query, commit=True)
    logger.info("Created table: %s.%s", DB_SCHEMA, table_name)


def create_indexes_if_not_exists() -> None:
    """Create performance indexes used by the application."""
    schema_index_name = f"idx_{SCHEMA_TABLE_NAME}_version"
    cache_index_name = f"idx_{CACHE_TABLE_NAME}_original_text"

    if not index_exists(DB_SCHEMA, schema_index_name):
        create_index_query = f"""
            CREATE INDEX "{schema_index_name}"
            ON "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}" (version);
        """
        execute_query(create_index_query, commit=True)
        logger.info("Created index: %s.%s", DB_SCHEMA, schema_index_name)
    else:
        logger.info("Index already exists: %s.%s", DB_SCHEMA, schema_index_name)

    if not index_exists(DB_SCHEMA, cache_index_name):
        create_index_query = f"""
            CREATE INDEX "{cache_index_name}"
            ON "{DB_SCHEMA}"."{CACHE_TABLE_NAME}" (original_text);
        """
        execute_query(create_index_query, commit=True)
        logger.info("Created index: %s.%s", DB_SCHEMA, cache_index_name)
    else:
        logger.info("Index already exists: %s.%s", DB_SCHEMA, cache_index_name)


# ---------------------------------------------------------------------
# Table creation and initialization
# ---------------------------------------------------------------------
def create_schema_table_if_not_exists() -> None:
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


def create_cache_table_if_not_exists() -> None:
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


def initialize_database() -> None:
    """Create all required tables, indexes, and seed the default schema if needed."""
    create_schema_if_not_exists()
    create_schema_table_if_not_exists()
    create_cache_table_if_not_exists()
    create_indexes_if_not_exists()
    ensure_default_schema_exists()


# ---------------------------------------------------------------------
# Schema operations
# ---------------------------------------------------------------------
def save_schema(version: str, schema_json: Dict[str, Any]):
    """
    Insert a schema definition into the translation_schema table.

    This is append-only and keeps schema history.
    """
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


def ensure_default_schema_exists() -> None:
    """Ensure the translation_schema table has a default schema record."""
    check_query = f"""
        SELECT COUNT(*) AS count
        FROM "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}"
        WHERE version = %s
    """
    row = execute_query(check_query, ("1.0",), fetchone=True)
    count = row["count"] if row else 0

    if count == 0:
        save_schema("1.0", DEFAULT_SCHEMA)
        logger.info("Default schema inserted into %s.%s", DB_SCHEMA, SCHEMA_TABLE_NAME)


def fetch_schema_by_version(version: str) -> Dict[str, Any]:
    """
    Fetch schema details by version from the translation_schema table.

    Returns the latest stored schema if found. If not found, a default schema
    template is returned and also persisted for future lookups.
    """
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
            return row["schema"]
    except Exception as exc:
        logger.exception("Schema fetch failed for version %s: %s", version, exc)

    default_schema = DEFAULT_SCHEMA.copy()
    default_schema["schema_version"] = version

    try:
        save_schema(version, default_schema)
        logger.info("Returning and saving default schema for version %s", version)
    except Exception as exc:
        logger.exception("Failed to save default schema for version %s: %s", version, exc)

    return default_schema


# ---------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------
def insert_translation_cache(records: List[Dict[str, str]]):
    """Insert translation cache records into the database."""
    if not records:
        logger.info("No records to insert")
        return

    insert_query = f"""
        INSERT INTO "{DB_SCHEMA}"."{CACHE_TABLE_NAME}"
        (
            uuid,
            original_text,
            translated_text
        )
        VALUES %s
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
        execute_values(cur, insert_query, values)

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

    select_query = f"""
        SELECT original_text, translated_text
        FROM "{DB_SCHEMA}"."{CACHE_TABLE_NAME}"
        WHERE original_text = ANY(%s)
    """

    rows = execute_query(select_query, (original_texts,), fetchall=True) or []

    logger.info(
        "Cache lookup returned %s record(s) from %s.%s",
        len(rows),
        DB_SCHEMA,
        CACHE_TABLE_NAME,
    )

    return {row["original_text"]: row["translated_text"] for row in rows}


# ---------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------
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

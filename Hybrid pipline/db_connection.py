import os
import uuid
import json
import logging

from dotenv import load_dotenv
import psycopg2

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


def get_db_connection():
    logger.info("Opening database connection to %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


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

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            query,
            (schema_name, table_name)
        )

        exists = cur.fetchone()[0]
        logger.info("Table exists check for %s.%s: %s", schema_name, table_name, exists)
        return exists

    finally:
        if cur:
            cur.close()

        if conn:
            conn.close()


def create_schema_table_if_not_exists():
    """Create the translation_schema table if it does not exist."""
    if table_exists(DB_SCHEMA, SCHEMA_TABLE_NAME):
        logger.info(
            "Table already exists: %s.%s",
            DB_SCHEMA,
            SCHEMA_TABLE_NAME
        )
        return

    create_schema_query = f"""
        CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}";
    """

    create_table_query = f"""
        CREATE TABLE "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}" (
            uuid UUID PRIMARY KEY,
            version TEXT NOT NULL,
            schema JSONB NOT NULL,
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
            SCHEMA_TABLE_NAME
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


def fetch_schema_by_version(version: str) -> dict:
    """Fetch schema details by version from the translation_schema table.

    Args:
        version: The schema version to fetch.

    Returns:
        The schema dictionary if found, otherwise an empty dict.
    """
    create_schema_table_if_not_exists()

    select_query = f"""
        SELECT schema
        FROM "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}"
        WHERE version = %s
        ORDER BY created_at DESC
        LIMIT 1
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(select_query, (version,))

        row = cur.fetchone()
        if row:
            logger.info("Schema found for version %s", version)
            return row[0]
        logger.info("No schema found for version %s", version)
        return {}

    except Exception as e:
        logger.error("Schema fetch failed: %s", str(e))
        raise

    finally:
        if cur:
            cur.close()

        if conn:
            conn.close()


def insert_schema(version: str, schema_json: dict):
    """Insert a schema definition into the translation_schema table.

    Args:
        version: The schema version identifier.
        schema_json: The schema dictionary to store.
    """
    create_schema_table_if_not_exists()

    insert_query = f"""
        INSERT INTO "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}"
        (uuid, version, schema)
        VALUES (%s, %s, %s)
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(insert_query, (str(uuid.uuid4()), version, json.dumps(schema_json)))

        conn.commit()
        logger.info("Schema version %s inserted successfully", version)

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


def update_schema(version: str, schema_json: dict):
    """Update a schema definition in the translation_schema table.

    Inserts a new version record. The table maintains history via created_at timestamp.

    Args:
        version: The schema version identifier to update.
        schema_json: The schema dictionary to store.
    """
    create_schema_table_if_not_exists()

    insert_query = f"""
        INSERT INTO "{DB_SCHEMA}"."{SCHEMA_TABLE_NAME}"
        (uuid, version, schema)
        VALUES (%s, %s, %s)
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(insert_query, (str(uuid.uuid4()), version, json.dumps(schema_json)))

        conn.commit()
        logger.info("Schema version %s updated successfully", version)

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


def create_cache_table_if_not_exists():
    """Create the translation cache table if it does not exist."""
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


def insert_translation_cache(records):
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
            "%s records inserted successfully into %s.%s",
            len(values),
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


def fetch_translation_cache(original_texts):
    """Fetch translation cache records by original texts."""
    if not original_texts:
        logger.info("No texts provided for cache lookup")
        return {}

    create_cache_table_if_not_exists()

    select_query = f"""
        SELECT original_text, translated_text
        FROM "{DB_SCHEMA}"."{CACHE_TABLE_NAME}"
        WHERE original_text = ANY(%s)
    """

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(select_query, (list(original_texts),))

        rows = cur.fetchall()
        logger.info(
            "Cache lookup returned %s record(s) from %s.%s",
            len(rows),
            DB_SCHEMA,
            CACHE_TABLE_NAME
        )
        return {row[0]: row[1] for row in rows}

    except Exception as e:
        logger.error("Cache lookup failed: %s", str(e))
        raise

    finally:
        if cur:
            cur.close()

        if conn:
            conn.close()


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
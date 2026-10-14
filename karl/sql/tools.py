import json
import os
import re
import ssl
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit
from urllib.request import urlopen

from langchain_core.tools import tool
from pydantic import BaseModel

try:
    import pymysql
except ImportError:
    raise ImportError("Please install karl[mysql] to use SQL tools")


DATABASES_ENV_VAR = "KARL_DATABASES"
DEFAULT_MAX_ROWS = 100
HARD_MAX_ROWS = 1_000
MAX_CELL_CHARS = 500
CONNECT_TIMEOUT_SECONDS = 10
QUERY_TIMEOUT_SECONDS = int(os.getenv("KARL_SQL_TIMEOUT_SECONDS", "30"))

_ALLOWED_SCHEMES = {"mysql", "mysql+pymysql", "mariadb"}
_ALLOWED_PREFIXES = ("select", "with", "show", "describe", "desc", "explain")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseConfig:
    name: str
    host: str
    port: int
    user: str
    password: str = field(repr=False, default="")
    database: str = ""
    description: str | None = None

    def __str__(self) -> str:
        """Safe for logs and tool output - never includes the password."""
        return f"{self.name} ({self.user}@{self.host}:{self.port}/{self.database})"


def _parse_database(name: str, raw: str | dict) -> DatabaseConfig:
    description = None

    if isinstance(raw, dict):
        description = raw.get("description")
        raw = raw.get("url") or raw.get("uri") or ""

    url = str(raw).strip()
    if url.startswith("jdbc:"):
        url = url[len("jdbc:") :]

    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Database {name!r} has unsupported scheme {parts.scheme!r}. "
            f"Only MySQL is supported: {sorted(_ALLOWED_SCHEMES)}"
        )

    params = dict(parse_qsl(parts.query))

    return DatabaseConfig(
        name=name,
        host=parts.hostname or "localhost",
        port=parts.port or 3306,
        user=unquote(parts.username or params.get("user", "")),
        password=unquote(parts.password or params.get("password", "")),
        database=unquote(parts.path.lstrip("/")) or params.get("database", ""),
        description=description,
    )


@lru_cache(maxsize=1)
def _databases() -> dict[str, DatabaseConfig]:
    configured = json.loads(os.getenv(DATABASES_ENV_VAR, "{}"))
    return {name: _parse_database(name, raw) for name, raw in configured.items()}


def _config(database: str) -> DatabaseConfig:
    databases = _databases()
    if database not in databases:
        known = ", ".join(sorted(databases)) or "<none configured>"
        raise ValueError(
            f"Unknown database {database!r}. Configured databases: {known}. "
            f"Use list_databases to see what is available."
        )
    return databases[database]


# -----------------------------------------------------------------------------
# Query guards
# -----------------------------------------------------------------------------

_COMMENTS_AND_LITERALS = re.compile(
    r"""
      '(?:\\.|[^'\\])*'      # single quoted string
    | "(?:\\.|[^"\\])*"      # double quoted string
    | `(?:[^`])*`            # backtick identifier
    | /\*.*?\*/              # block comment
    | (?:--|\#)[^\n]*        # line comment
    """,
    re.VERBOSE | re.DOTALL,
)


def _sanitise(sql: str) -> str:
    """Strip literals and comments so keyword checks can't be fooled by data."""
    return _COMMENTS_AND_LITERALS.sub(" ", sql)


def _assert_read_only(sql: str) -> None:
    if not sql.strip():
        raise ValueError("Empty query.")

    sanitised = _sanitise(sql).strip().rstrip(";")

    if ";" in sanitised:
        raise ValueError(
            "Only a single statement may be executed per call. "
            "Split the work into separate query_database calls."
        )

    first_word = (
        sanitised.lstrip("( \t\n").split(None, 1)[0].lower() if sanitised else ""
    )
    if first_word not in _ALLOWED_PREFIXES:
        raise ValueError(
            f"Only read-only statements are permitted "
            f"({', '.join(w.upper() for w in _ALLOWED_PREFIXES)}). Got {first_word.upper()!r}."
        )

    if re.search(r"\binto\s+(outfile|dumpfile)\b", sanitised, re.IGNORECASE):
        raise ValueError("SELECT ... INTO OUTFILE/DUMPFILE is not permitted.")


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------


class QueryResult(BaseModel):
    database: str
    sql: str
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    truncated_rows: bool = False
    truncated_cells: bool = False
    note: str | None = None


def _coerce(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8", errors="replace")

    text = str(value)
    if len(text) > MAX_CELL_CHARS:
        return text[:MAX_CELL_CHARS] + f"...[+{len(text) - MAX_CELL_CHARS} chars]"
    return text


RDS_CA_BUNDLE_URL = "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"


@lru_cache(maxsize=1)
def _rds_ssl_context() -> ssl.SSLContext:
    with urlopen(RDS_CA_BUNDLE_URL, timeout=10) as response:
        ca_pem = response.read().decode("ascii")

    # Trust the AWS bundle, and require a certificate matching the DB hostname.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_verify_locations(cadata=ca_pem)
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    return context


def _connect(config: DatabaseConfig):
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database or None,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        read_timeout=QUERY_TIMEOUT_SECONDS + 5,
        write_timeout=QUERY_TIMEOUT_SECONDS + 5,
        ssl=_rds_ssl_context(),
    )

    try:
        # Explicitly reject a connection that did not negotiate TLS.
        with connection.cursor() as cursor:
            cursor.execute("SHOW SESSION STATUS LIKE 'Ssl_cipher'")
            status = cursor.fetchone()
            if not status or not status[1]:
                raise RuntimeError(f"Database {config.name!r} did not negotiate TLS.")
        return connection
    except Exception:
        connection.close()
        raise


def _execute(
    config: DatabaseConfig,
    sql: str,
    params: tuple | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> QueryResult:
    max_rows = max(1, min(int(max_rows), HARD_MAX_ROWS))

    with closing(_connect(config)) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
            cursor.execute(
                "SET SESSION max_execution_time = %s", (QUERY_TIMEOUT_SECONDS * 1000,)
            )
            cursor.execute(sql, params)

            columns = [d[0] for d in cursor.description or []]
            # Fetch one extra row so we can tell the caller there is more.
            fetched = cursor.fetchmany(max_rows + 1)

    truncated_rows = len(fetched) > max_rows
    fetched = fetched[:max_rows]

    truncated_cells = False
    rows: list[list[Any]] = []
    for raw_row in fetched:
        row = []
        for raw_value in raw_row:
            coerced = _coerce(raw_value)
            if isinstance(coerced, str) and coerced.endswith(" chars]"):
                truncated_cells = True
            row.append(coerced)
        rows.append(row)

    note = None
    if truncated_rows:
        note = (
            f"Showing the first {max_rows} rows only - there are more. "
            f"Narrow the query with a WHERE clause, or aggregate instead of listing rows."
        )

    return QueryResult(
        database=config.name,
        sql=sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated_rows=truncated_rows,
        truncated_cells=truncated_cells,
        note=note,
    )


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


class DatabaseSummary(BaseModel):
    name: str
    host: str
    schema_name: str
    user: str
    description: str | None = None


@tool
def list_databases() -> list[DatabaseSummary]:
    """
    Lists the MySQL databases Karl can query, by environment name (e.g. qa, staging, production).
    All connections are read-only. Call this first to discover the valid `database` argument
    for the other SQL tools, and read the descriptions to pick the right environment.
    Credentials are never returned.
    """
    return [
        DatabaseSummary(
            name=config.name,
            host=f"{config.host}:{config.port}",
            schema_name=config.database,
            user=config.user,
            description=config.description,
        )
        for config in _databases().values()
    ]


@tool
def list_database_tables(database: str, name_like: str = "%") -> QueryResult:
    """
    Lists the tables in a database, with approximate row counts and size, so you can
    find the right table before querying it.
    Set database to an environment name from list_databases.
    Set name_like to a MySQL LIKE pattern to filter table names, e.g. "user%" (default: all tables).
    """
    config = _config(database)
    return _execute(
        config,
        """
        SELECT TABLE_NAME                              AS table_name,
               TABLE_ROWS                              AS approx_rows,
               ROUND(DATA_LENGTH / 1024 / 1024, 1)     AS data_mb,
               TABLE_COMMENT                           AS comment
          FROM information_schema.TABLES
         WHERE TABLE_SCHEMA = %s
           AND TABLE_NAME LIKE %s
         ORDER BY TABLE_NAME
        """,
        (config.database, name_like),
        max_rows=HARD_MAX_ROWS,
    )


@tool
def describe_database_table(database: str, table: str) -> dict[str, QueryResult]:
    """
    Describes a table's columns and indexes. Use this before writing a query so you use
    real column names and can tell which columns are indexed (unindexed filters on large
    tables will be slow or time out).
    Set database to an environment name from list_databases.
    Set table to a table name from list_database_tables.
    """
    config = _config(database)

    columns = _execute(
        config,
        """
        SELECT COLUMN_NAME   AS column_name,
               COLUMN_TYPE   AS type,
               IS_NULLABLE   AS nullable,
               COLUMN_KEY    AS key_type,
               COLUMN_DEFAULT AS default_value,
               COLUMN_COMMENT AS comment
          FROM information_schema.COLUMNS
         WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
         ORDER BY ORDINAL_POSITION
        """,
        (config.database, table),
        max_rows=HARD_MAX_ROWS,
    )

    if not columns.rows:
        raise ValueError(
            f"Table {table!r} not found in {config.database!r} on {database!r}. "
            f"Use list_database_tables to see available tables."
        )

    indexes = _execute(
        config,
        """
        SELECT INDEX_NAME   AS index_name,
               NON_UNIQUE   AS non_unique,
               SEQ_IN_INDEX AS position,
               COLUMN_NAME  AS column_name
          FROM information_schema.STATISTICS
         WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
         ORDER BY INDEX_NAME, SEQ_IN_INDEX
        """,
        (config.database, table),
        max_rows=HARD_MAX_ROWS,
    )

    return {"columns": columns, "indexes": indexes}


@tool
def query_database(
    database: str, sql: str, max_rows: int = DEFAULT_MAX_ROWS
) -> QueryResult:
    """
    Runs a single read-only MySQL query against one of the configured databases and
    returns the result set as `columns` plus `rows` (each row is a list of values in
    column order).

    Set database to an environment name from list_databases (e.g. "qa", "staging", "production").
    Set sql to ONE statement. Only SELECT, WITH, SHOW, DESCRIBE and EXPLAIN are permitted;
    the connection uses a read-only user, so writes and DDL will be rejected.
    Set max_rows to cap how many rows come back (default 100, maximum 1000).

    Guidance:
    - Use describe_database_table first so you reference real column names.
    - Always constrain the query. Prefer aggregates (COUNT, SUM, GROUP BY) over dumping rows.
    - Queries are cancelled after a timeout, so avoid unindexed scans of large tables.
    - If `truncated_rows` is true you have NOT seen the whole result set; refine the query
      rather than assuming the rows you got are complete.
    """
    config = _config(database)
    _assert_read_only(sql)
    return _execute(config, sql.strip().rstrip(";"), max_rows=max_rows)


async def get_database_tools():
    return [
        list_databases,
        list_database_tables,
        query_database,
        describe_database_table,
    ]

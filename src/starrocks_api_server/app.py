"""FastAPI application exposing the StarRocks MCP capabilities over HTTP."""

from __future__ import annotations

import ast
import base64
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from mcp_server_starrocks.db_client import (
    DBClient,
    PerfAnalysisInput,
    ResultSet,
    get_db_client,
    reset_db_connections,
)
from mcp_server_starrocks.db_summary_manager import (
    DatabaseSummaryManager,
    get_db_summary_manager,
)

API_TOKEN_ENV = "STARROCKS_API_BEARER_TOKEN"
DEFAULT_TABLE_OVERVIEW_LIMIT = int(os.getenv("STARROCKS_OVERVIEW_LIMIT", "20000"))

bearer_scheme = HTTPBearer(auto_error=False)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., description="SQL statement to execute")
    database: Optional[str] = Field(
        default=None,
        description="Optional database context used when executing the SQL statement.",
    )


class AnalyzeQueryRequest(BaseModel):
    uuid: Optional[str] = Field(
        default=None, description="Existing query identifier used for ANALYZE PROFILE."
    )
    sql: Optional[str] = Field(
        default=None, description="SQL text to analyse via EXPLAIN ANALYZE."
    )
    database: Optional[str] = Field(
        default=None, description="Optional database context used for the analysis query."
    )


class PlotQueryRequest(QueryRequest):
    plotly_expr: str = Field(
        ...,
        description=(
            "Plotly expression evaluated against a pandas DataFrame named ``df`` generated "
            "from the SQL query results."
        ),
    )
    response_format: Optional[str] = Field(
        default=None,
        description="Optional override for the response payload. Valid values: 'plotly_json', 'image_base64'.",
    )


class TableOverviewResponse(BaseModel):
    success: bool
    table: Dict[str, str]
    row_count: Optional[int]
    columns: List[Dict[str, Any]]
    sample_rows: List[Dict[str, Any]]
    errors: List[str]
    cache: Dict[str, Any]


@dataclass
class TableOverviewCacheEntry:
    payload: Dict[str, Any]
    timestamp: float


def validate_plotly_expr(expr: str) -> None:
    """Validate that an expression is a single Plotly Express call."""

    try:
        tree = ast.parse(expr)
    except SyntaxError as exc:  # pragma: no cover - FastAPI surfaces the exception
        raise SyntaxError(f"Invalid Python syntax in expression: {exc}") from exc

    if len(tree.body) != 1:
        raise ValueError("Expression must be a single statement or expression.")

    node = tree.body[0]
    if not isinstance(node, ast.Expr):
        raise ValueError("Expression must be a single expression, not a statement.")

    call = node.value
    if not isinstance(call, ast.Call):
        raise ValueError("Expression must be a function call.")

    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
        raise ValueError("Function call must access an attribute on a simple name (e.g. px.scatter).")

    if call.func.value.id != "px":
        raise ValueError("Function call must target the 'px' object.")

    for index, arg_node in enumerate(call.args):
        for sub_node in ast.walk(arg_node):
            if isinstance(sub_node, ast.Call):
                raise ValueError(
                    f"Positional argument at index {index} contains a disallowed nested function call."
                )

    for keyword in call.keywords:
        for sub_node in ast.walk(keyword.value):
            if isinstance(sub_node, ast.Call):
                keyword_name = keyword.arg if keyword.arg else "<unknown>"
                raise ValueError(
                    f"Keyword argument '{keyword_name}' contains a disallowed nested function call."
                )


def create_app() -> FastAPI:
    """Create and configure a FastAPI application instance."""

    application = FastAPI(
        title="StarRocks MCP API Server",
        description=(
            "REST API that mirrors the functionality of the StarRocks MCP server, "
            "enabling integrations such as GPT Actions via the generated OpenAPI schema."
        ),
        version="0.1.0",
    )

    application.state.table_overview_cache: Dict[Tuple[str, str], TableOverviewCacheEntry] = {}

    @application.on_event("shutdown")
    def _close_connections() -> None:
        reset_db_connections()

    application.include_router(router)
    return application


def get_db_client_dependency() -> DBClient:
    return get_db_client()


def get_db_summary_manager_dependency(
    db_client: DBClient = Depends(get_db_client_dependency),
) -> DatabaseSummaryManager:
    return get_db_summary_manager(db_client)


def require_authentication(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> None:
    expected_token = os.getenv(API_TOKEN_ENV)
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API token is not configured on the server.",
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )

    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        )


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def escape_proc_path(path: str) -> str:
    return path.replace("'", "''")


def normalise_cell(value: Any) -> Any:
    if isinstance(value, Decimal):
        # Cast to float for JSON compatibility.
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def rows_to_lists(rows: Optional[Iterable[Iterable[Any]]]) -> List[List[Any]]:
    if not rows:
        return []
    return [list(map(normalise_cell, row)) for row in rows]


def rows_to_dicts(result: ResultSet) -> List[Dict[str, Any]]:
    if not result.column_names or not result.rows:
        return []
    column_names = list(result.column_names)
    dictionaries: List[Dict[str, Any]] = []
    for row in result.rows:
        row_dict = {column: normalise_cell(value) for column, value in zip(column_names, row)}
        dictionaries.append(row_dict)
    return dictionaries


def build_query_response(result: ResultSet) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": result.success,
        "execution_time": result.execution_time,
    }
    if result.column_names is not None:
        payload["columns"] = list(result.column_names)
        payload["rows"] = rows_to_lists(result.rows)
    if result.rows_affected is not None:
        payload["rows_affected"] = result.rows_affected
    if result.error_message:
        payload["error_message"] = result.error_message
    if result.success:
        payload["preview"] = result.to_string(limit=10000)
    return payload


def resolve_plot_response_format(requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    env_value = os.getenv("STARROCKS_PLOTLY_JSON")
    if env_value and env_value.lower() not in {"0", "false", "no"}:
        return "plotly_json"
    return None


def compute_table_overview(
    db_client: DBClient,
    database: str,
    table: str,
    limit: int = DEFAULT_TABLE_OVERVIEW_LIMIT,
) -> Dict[str, Any]:
    errors: List[str] = []
    success = True
    row_count: Optional[int] = None
    columns: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    table_identifier = quote_identifier(table)

    count_result = db_client.execute(
        f"SELECT COUNT(*) FROM {table_identifier}", db=database
    )
    if count_result.success and count_result.rows:
        row_count = int(count_result.rows[0][0])
    else:
        success = False
        errors.append(count_result.error_message or "Failed to retrieve row count.")

    describe_result = db_client.execute(f"DESCRIBE {table_identifier}", db=database)
    if describe_result.success:
        columns = rows_to_dicts(describe_result)
    else:
        success = False
        errors.append(describe_result.error_message or "Failed to retrieve column metadata.")

    sample_result = db_client.execute(
        f"SELECT * FROM {table_identifier} LIMIT 3", db=database
    )
    if sample_result.success:
        sample_rows = rows_to_dicts(sample_result)
    else:
        # Sample rows are useful but optional; do not flip overall success for this alone.
        errors.append(sample_result.error_message or "Failed to fetch sample rows.")

    overview_payload = {
        "success": success,
        "table": {"database": database, "name": table},
        "row_count": row_count,
        "columns": columns,
        "sample_rows": sample_rows,
        "errors": errors,
        "cache": {"hit": False, "last_updated": time.time()},
    }
    return overview_payload


@router.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/databases", dependencies=[Depends(require_authentication)])
def list_databases(
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    result = db_client.execute("SHOW DATABASES")
    if not result.success:
        return {
            "success": False,
            "error_message": result.error_message,
        }
    databases = [row[0] for row in rows_to_lists(result.rows)]
    return {
        "success": True,
        "databases": databases,
        "execution_time": result.execution_time,
    }


@router.get(
    "/databases/{database}/tables",
    dependencies=[Depends(require_authentication)],
)
def list_tables(
    database: str,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    result = db_client.execute(
        f"SHOW TABLES FROM {quote_identifier(database)}", db=database
    )
    if not result.success:
        return {
            "success": False,
            "error_message": result.error_message,
        }
    tables = [row[0] for row in rows_to_lists(result.rows)]
    return {
        "success": True,
        "database": database,
        "tables": tables,
        "execution_time": result.execution_time,
    }


@router.get(
    "/databases/{database}/tables/{table}/schema",
    dependencies=[Depends(require_authentication)],
)
def get_table_schema(
    database: str,
    table: str,
    format: Optional[str] = Query(
        default="text", description="Return format. Only 'text' is currently supported."
    ),
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    result = db_client.execute(
        f"SHOW CREATE TABLE {quote_identifier(database)}.{quote_identifier(table)}",
        db=database,
    )
    if not result.success:
        return {
            "success": False,
            "error_message": result.error_message,
        }
    schema_sql = result.to_string()
    return {
        "success": True,
        "database": database,
        "table": table,
        "schema_sql": schema_sql,
        "format": format,
    }


@router.get(
    "/proc/{path:path}",
    dependencies=[Depends(require_authentication)],
)
def get_proc_information(
    path: str,
    limit: Optional[int] = Query(
        default=None,
        description="Optional character limit applied to the textual response.",
    ),
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    query_path = escape_proc_path(path)
    result = db_client.execute(f"show proc '{query_path}'")
    if not result.success:
        return {
            "success": False,
            "error_message": result.error_message,
        }
    return {
        "success": True,
        "path": path,
        "content": result.to_string(limit=limit or DEFAULT_TABLE_OVERVIEW_LIMIT),
    }


@router.post(
    "/queries/read",
    dependencies=[Depends(require_authentication)],
)
def execute_read_query(
    payload: QueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    result = db_client.execute(payload.query, db=payload.database)
    return build_query_response(result)


@router.post(
    "/queries/write",
    dependencies=[Depends(require_authentication)],
)
def execute_write_query(
    payload: QueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    result = db_client.execute(payload.query, db=payload.database)
    return build_query_response(result)


@router.post(
    "/queries/analyze",
    dependencies=[Depends(require_authentication)],
)
def analyze_query(
    payload: AnalyzeQueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    if not payload.uuid and not payload.sql:
        return {
            "success": False,
            "error_message": "Either 'uuid' or 'sql' must be provided.",
        }

    if payload.uuid:
        result = db_client.execute(
            f"ANALYZE PROFILE FROM '{payload.uuid}'", db=payload.database
        )
    else:
        result = db_client.execute(
            f"EXPLAIN ANALYZE {payload.sql}", db=payload.database
        )

    if not result.success:
        return {
            "success": False,
            "error_message": result.error_message,
        }

    return {
        "success": True,
        "analysis": result.to_string(),
    }


@router.post(
    "/queries/perf-collect",
    dependencies=[Depends(require_authentication)],
)
def collect_perf_data(
    payload: QueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    result: PerfAnalysisInput = db_client.collect_perf_analysis_input(
        payload.query, db=payload.database
    )
    error_message = result.get("error_message")
    return {
        "success": not error_message,
        "result": result,
    }


@router.post(
    "/queries/plot",
    dependencies=[Depends(require_authentication)],
)
def query_and_plot(
    payload: PlotQueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    try:
        result = db_client.execute(
            payload.query, db=payload.database, return_format="pandas"
        )
        if not result.success:
            return {
                "success": False,
                "error_message": result.error_message,
            }
        if result.pandas is None:
            return {
                "success": False,
                "error_message": "Query did not return tabular data.",
            }

        df: pd.DataFrame = result.pandas
        if df.empty:
            return {
                "success": False,
                "error_message": "Query returned no data to plot.",
            }

        validate_plotly_expr(payload.plotly_expr)
        local_vars = {"df": df}
        figure = eval(payload.plotly_expr, {"px": px}, local_vars)

        response_format = resolve_plot_response_format(payload.response_format)
        preview = df.head(20).to_dict(orient="records")

        if response_format == "plotly_json":
            plot_json = figure.to_plotly_json()
            structured_content = build_query_response(result)
            structured_content["data"] = plot_json["data"]
            structured_content["layout"] = plot_json["layout"]
            structured_content["dataframe_preview"] = preview
            return {
                "success": True,
                "chart": {
                    "format": "plotly_json",
                    "data": plot_json["data"],
                    "layout": plot_json["layout"],
                },
                "query_result": structured_content,
            }

        img_bytes = figure.to_image(format="png", width=960, height=720)
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        structured_content = build_query_response(result)
        structured_content["dataframe_preview"] = preview
        return {
            "success": True,
            "chart": {
                "format": "image_base64",
                "image": image_base64,
            },
            "query_result": structured_content,
        }
    except Exception as exc:  # pragma: no cover - dependent on plotly/kaleido runtime
        return {
            "success": False,
            "error_message": str(exc),
        }


@router.get(
    "/databases/{database}/tables/{table}/overview",
    response_model=TableOverviewResponse,
    dependencies=[Depends(require_authentication)],
)
def table_overview(
    database: str,
    table: str,
    request: Request,
    refresh: bool = Query(
        default=False,
        description="Set to true to bypass the cached response and fetch fresh data.",
    ),
    db_client: DBClient = Depends(get_db_client_dependency),
) -> Dict[str, Any]:
    cache_key = (database, table)
    cache_store = request.app.state.table_overview_cache
    if not refresh and cache_key in cache_store:
        payload = deepcopy(cache_store[cache_key].payload)
        payload["cache"]["hit"] = True
        return payload

    payload = compute_table_overview(db_client, database, table)
    cache_store[cache_key] = TableOverviewCacheEntry(payload=payload, timestamp=time.time())
    return payload


@router.get(
    "/databases/{database}/summary",
    dependencies=[Depends(require_authentication)],
)
def database_summary(
    database: str,
    limit: int = Query(
        default=10000,
        description="Character limit applied to the textual summary output.",
    ),
    refresh: bool = Query(
        default=False,
        description="Set to true to force refresh of the cached summary.",
    ),
    summary_manager: DatabaseSummaryManager = Depends(
        get_db_summary_manager_dependency
    ),
) -> Dict[str, Any]:
    summary = summary_manager.get_database_summary(database, limit=limit, refresh=refresh)
    summary_text = summary or ""
    success = not summary_text.lower().startswith("error:")
    return {
        "success": success,
        "database": database,
        "summary": summary,
        "limit": limit,
        "refresh": refresh,
    }


app = create_app()


__all__ = [
    "app",
    "create_app",
    "get_db_client_dependency",
    "get_db_summary_manager_dependency",
    "router",
    "require_authentication",
]

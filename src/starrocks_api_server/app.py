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
PUBLIC_SERVER_URL = "https://starrocks-mcp.devicesformula.com"
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


class HealthcheckResponse(BaseModel):
    status: str = Field(..., description="Health status of the API server.")


class BaseSuccessResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded.")
    error_message: Optional[str] = Field(
        default=None, description="Error details when success is False."
    )


class ListDatabasesResponse(BaseSuccessResponse):
    databases: List[str] = Field(
        default_factory=list,
        description="Names of databases visible to the authenticated user.",
    )
    execution_time: Optional[float] = Field(
        default=None, description="Execution time reported by StarRocks."
    )


class ListTablesResponse(BaseSuccessResponse):
    database: Optional[str] = Field(
        default=None, description="Database associated with the listed tables."
    )
    tables: List[str] = Field(
        default_factory=list,
        description="Tables contained within the requested database.",
    )
    execution_time: Optional[float] = Field(
        default=None, description="Execution time reported by StarRocks."
    )


class TableSchemaResponse(BaseSuccessResponse):
    database: Optional[str] = Field(
        default=None, description="Database containing the requested table."
    )
    table: Optional[str] = Field(
        default=None, description="Table whose schema was requested."
    )
    schema_sql: Optional[str] = Field(
        default=None, description="CREATE TABLE statement for the table."
    )
    format: Optional[str] = Field(
        default=None, description="Schema output format."
    )


class ProcResponse(BaseSuccessResponse):
    path: Optional[str] = Field(
        default=None, description="`SHOW PROC` path that was queried."
    )
    content: Optional[str] = Field(
        default=None, description="Textual response returned by StarRocks."
    )


class QueryResultResponse(BaseSuccessResponse):
    execution_time: Optional[float] = Field(
        default=None, description="Execution time reported by StarRocks."
    )
    columns: Optional[List[str]] = Field(
        default=None, description="Column names returned by the query."
    )
    rows: Optional[List[List[Any]]] = Field(
        default=None, description="Tabular rows returned by the query."
    )
    rows_affected: Optional[int] = Field(
        default=None, description="Number of rows impacted by a write query."
    )
    preview: Optional[str] = Field(
        default=None,
        description="Textual preview of the query results.",
    )
    dataframe_preview: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Preview of the query results when rendered as records.",
    )
    data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Plotly trace data generated for the query (when requested).",
    )
    layout: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Plotly layout metadata generated for the query (when requested).",
    )


class AnalyzeQueryResponse(BaseSuccessResponse):
    analysis: Optional[str] = Field(
        default=None, description="Output from ANALYZE PROFILE or EXPLAIN ANALYZE."
    )


class PerfCollectResponse(BaseSuccessResponse):
    result: Optional[PerfAnalysisInput] = Field(
        default=None,
        description="Profiling payload generated for the supplied query.",
    )


class PlotChartPayload(BaseModel):
    format: str = Field(..., description="Format of the chart payload.")
    data: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Plotly JSON data series."
    )
    layout: Optional[Dict[str, Any]] = Field(
        default=None, description="Plotly JSON layout metadata."
    )
    image: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG image of the rendered chart.",
    )


class PlotQueryResponse(BaseSuccessResponse):
    chart: Optional[PlotChartPayload] = Field(
        default=None, description="Rendered chart output."
    )
    query_result: Optional[QueryResultResponse] = Field(
        default=None,
        description="Structured representation of the executed query result.",
    )


class DatabaseSummaryResponse(BaseSuccessResponse):
    database: str = Field(..., description="Database whose summary was requested.")
    summary: Optional[str] = Field(
        default=None,
        description="Generated textual summary of the database.",
    )
    limit: int = Field(..., description="Character limit applied to the summary.")
    refresh: bool = Field(..., description="Whether the summary bypassed the cache.")


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


def resolve_public_server_url() -> str:
    """Return the externally accessible base URL advertised via OpenAPI."""

    return PUBLIC_SERVER_URL


def create_app() -> FastAPI:
    """Create and configure a FastAPI application instance."""

    public_url = resolve_public_server_url()
    application = FastAPI(
        title="StarRocks MCP API Server",
        description=(
            "REST API that mirrors the functionality of the StarRocks MCP server, "
            "enabling integrations such as GPT Actions via the generated OpenAPI schema."
        ),
        version="0.1.0",
        servers=[{"url": public_url}],
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


def build_query_response(result: ResultSet) -> QueryResultResponse:
    rows: Optional[List[List[Any]]] = None
    columns: Optional[List[str]] = None
    if result.column_names is not None:
        columns = list(result.column_names)
        rows = rows_to_lists(result.rows)

    preview: Optional[str] = None
    if result.success:
        preview = result.to_string(limit=10000)

    return QueryResultResponse(
        success=result.success,
        execution_time=result.execution_time,
        columns=columns,
        rows=rows,
        rows_affected=result.rows_affected,
        error_message=result.error_message,
        preview=preview,
    )


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


@router.get("/api/health", response_model=HealthcheckResponse)
def healthcheck() -> HealthcheckResponse:
    return HealthcheckResponse(status="ok")


@router.get(
    "/api/databases",
    response_model=ListDatabasesResponse,
    dependencies=[Depends(require_authentication)],
)
def list_databases(
    db_client: DBClient = Depends(get_db_client_dependency),
) -> ListDatabasesResponse:
    result = db_client.execute("SHOW DATABASES")
    if not result.success:
        return ListDatabasesResponse(
            success=False,
            databases=[],
            execution_time=result.execution_time,
            error_message=result.error_message,
        )
    databases = [row[0] for row in rows_to_lists(result.rows)]
    return ListDatabasesResponse(
        success=True,
        databases=databases,
        execution_time=result.execution_time,
    )


@router.get(
    "/api/databases/tables",
    response_model=ListTablesResponse,
    dependencies=[Depends(require_authentication)],
)
def list_tables(
    database: str = Query(..., description="Database whose tables are requested."),
    db_client: DBClient = Depends(get_db_client_dependency),
) -> ListTablesResponse:
    result = db_client.execute(
        f"SHOW TABLES FROM {quote_identifier(database)}", db=database
    )
    if not result.success:
        return ListTablesResponse(
            success=False,
            database=database,
            tables=[],
            execution_time=result.execution_time,
            error_message=result.error_message,
        )
    tables = [row[0] for row in rows_to_lists(result.rows)]
    return ListTablesResponse(
        success=True,
        database=database,
        tables=tables,
        execution_time=result.execution_time,
    )


@router.get(
    "/api/tables/schema",
    response_model=TableSchemaResponse,
    dependencies=[Depends(require_authentication)],
)
def get_table_schema(
    database: str = Query(..., description="Database containing the requested table."),
    table: str = Query(..., description="Table whose schema is requested."),
    format: Optional[str] = Query(
        default="text", description="Return format. Only 'text' is currently supported."
    ),
    db_client: DBClient = Depends(get_db_client_dependency),
) -> TableSchemaResponse:
    result = db_client.execute(
        f"SHOW CREATE TABLE {quote_identifier(database)}.{quote_identifier(table)}",
        db=database,
    )
    if not result.success:
        return TableSchemaResponse(
            success=False,
            database=database,
            table=table,
            format=format,
            error_message=result.error_message,
        )
    schema_sql = result.to_string()
    return TableSchemaResponse(
        success=True,
        database=database,
        table=table,
        schema_sql=schema_sql,
        format=format,
    )


@router.get(
    "/api/proc",
    response_model=ProcResponse,
    dependencies=[Depends(require_authentication)],
)
def get_proc_information(
    path: str = Query(..., description="SHOW PROC path to query."),
    limit: Optional[int] = Query(
        default=None,
        description="Optional character limit applied to the textual response.",
    ),
    db_client: DBClient = Depends(get_db_client_dependency),
) -> ProcResponse:
    query_path = escape_proc_path(path)
    result = db_client.execute(f"show proc '{query_path}'")
    if not result.success:
        return ProcResponse(
            success=False,
            path=path,
            content=None,
            error_message=result.error_message,
        )
    return ProcResponse(
        success=True,
        path=path,
        content=result.to_string(limit=limit or DEFAULT_TABLE_OVERVIEW_LIMIT),
    )


@router.post(
    "/api/queries/read",
    response_model=QueryResultResponse,
    dependencies=[Depends(require_authentication)],
)
def execute_read_query(
    payload: QueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> QueryResultResponse:
    result = db_client.execute(payload.query, db=payload.database)
    return build_query_response(result)


@router.post(
    "/api/queries/write",
    response_model=QueryResultResponse,
    dependencies=[Depends(require_authentication)],
)
def execute_write_query(
    payload: QueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> QueryResultResponse:
    result = db_client.execute(payload.query, db=payload.database)
    return build_query_response(result)


@router.post(
    "/api/queries/analyze",
    response_model=AnalyzeQueryResponse,
    dependencies=[Depends(require_authentication)],
)
def analyze_query(
    payload: AnalyzeQueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> AnalyzeQueryResponse:
    if not payload.uuid and not payload.sql:
        return AnalyzeQueryResponse(
            success=False,
            analysis=None,
            error_message="Either 'uuid' or 'sql' must be provided.",
        )

    if payload.uuid:
        result = db_client.execute(
            f"ANALYZE PROFILE FROM '{payload.uuid}'", db=payload.database
        )
    else:
        result = db_client.execute(
            f"EXPLAIN ANALYZE {payload.sql}", db=payload.database
        )

    if not result.success:
        return AnalyzeQueryResponse(
            success=False,
            analysis=None,
            error_message=result.error_message,
        )

    return AnalyzeQueryResponse(
        success=True,
        analysis=result.to_string(),
    )


@router.post(
    "/api/queries/perf-collect",
    response_model=PerfCollectResponse,
    dependencies=[Depends(require_authentication)],
)
def collect_perf_data(
    payload: QueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> PerfCollectResponse:
    result: PerfAnalysisInput = db_client.collect_perf_analysis_input(
        payload.query, db=payload.database
    )
    error_message = result.get("error_message")
    return PerfCollectResponse(
        success=not error_message,
        result=result,
        error_message=error_message,
    )


@router.post(
    "/api/queries/plot",
    response_model=PlotQueryResponse,
    dependencies=[Depends(require_authentication)],
)
def query_and_plot(
    payload: PlotQueryRequest,
    db_client: DBClient = Depends(get_db_client_dependency),
) -> PlotQueryResponse:
    try:
        result = db_client.execute(
            payload.query, db=payload.database, return_format="pandas"
        )
        if not result.success:
            return PlotQueryResponse(
                success=False,
                chart=None,
                query_result=build_query_response(result),
                error_message=result.error_message,
            )
        if result.pandas is None:
            return PlotQueryResponse(
                success=False,
                chart=None,
                query_result=build_query_response(result),
                error_message="Query did not return tabular data.",
            )

        df: pd.DataFrame = result.pandas
        if df.empty:
            return PlotQueryResponse(
                success=False,
                chart=None,
                query_result=build_query_response(result),
                error_message="Query returned no data to plot.",
            )

        validate_plotly_expr(payload.plotly_expr)
        local_vars = {"df": df}
        figure = eval(payload.plotly_expr, {"px": px}, local_vars)

        response_format = resolve_plot_response_format(payload.response_format)
        preview = df.head(20).to_dict(orient="records")

        if response_format == "plotly_json":
            plot_json = figure.to_plotly_json()
            structured_content = build_query_response(result)
            structured_content.data = plot_json["data"]
            structured_content.layout = plot_json["layout"]
            structured_content.dataframe_preview = preview
            return PlotQueryResponse(
                success=True,
                chart=PlotChartPayload(
                    format="plotly_json",
                    data=plot_json["data"],
                    layout=plot_json["layout"],
                ),
                query_result=structured_content,
            )

        img_bytes = figure.to_image(format="png", width=960, height=720)
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        structured_content = build_query_response(result)
        structured_content.dataframe_preview = preview
        return PlotQueryResponse(
            success=True,
            chart=PlotChartPayload(
                format="image_base64",
                image=image_base64,
            ),
            query_result=structured_content,
        )
    except Exception as exc:  # pragma: no cover - dependent on plotly/kaleido runtime
        return PlotQueryResponse(success=False, chart=None, error_message=str(exc))


@router.get(
    "/api/tables/overview",
    response_model=TableOverviewResponse,
    dependencies=[Depends(require_authentication)],
)
def table_overview(
    request: Request,
    database: str = Query(..., description="Database containing the table to inspect."),
    table: str = Query(..., description="Table to inspect."),
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
    "/api/databases/summary",
    response_model=DatabaseSummaryResponse,
    dependencies=[Depends(require_authentication)],
)
def database_summary(
    database: str = Query(..., description="Database to summarise."),
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
) -> DatabaseSummaryResponse:
    summary = summary_manager.get_database_summary(database, limit=limit, refresh=refresh)
    summary_text = summary or ""
    success = not summary_text.lower().startswith("error:")
    return DatabaseSummaryResponse(
        success=success,
        database=database,
        summary=summary,
        limit=limit,
        refresh=refresh,
        error_message=None if success else summary_text,
    )


app = create_app()


__all__ = [
    "app",
    "create_app",
    "get_db_client_dependency",
    "get_db_summary_manager_dependency",
    "router",
    "require_authentication",
]

# Copyright 2021-present StarRocks, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import ast
import asyncio
import base64
import hashlib
import json
import math
import sys
import os
import traceback
from fastmcp import FastMCP
from fastmcp.server.low_level import LowLevelServer
from fastmcp.utilities.types import Image
from fastmcp.tools.tool import ToolResult
import mcp.types as mcp_types
from mcp.types import TextContent, ImageContent
from fastmcp.exceptions import ToolError
from typing import Annotated, Any, Dict, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, RootModel
import plotly.express as px
import plotly.graph_objs
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from .db_client import get_db_client, reset_db_connections, ResultSet, PerfAnalysisInput
from .db_summary_manager import get_db_summary_manager

# Configure logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"), 
          format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")

mcp = FastMCP('mcp-server-starrocks')

# a hint for soft limit, not enforced
overview_length_limit = int(os.getenv('STARROCKS_OVERVIEW_LIMIT', str(20000)))
# Global cache for table overviews: {(db_name, table_name): overview_string}
global_table_overview_cache = {}

# Get database client instance
db_client = get_db_client()
# Get database summary manager instance
db_summary_manager = get_db_summary_manager(db_client)
# Description suffix for tools, if default db is set
description_suffix = f". db session already in default db `{db_client.default_database}`" if db_client.default_database else ""

SR_PROC_DESC = '''
Internal information exposed by StarRocks similar to linux /proc, following are some common paths:

'/frontends'	Shows the information of FE nodes.
'/backends'	Shows the information of BE nodes if this SR is non cloud native deployment.
'/compute_nodes'	Shows the information of CN nodes if this SR is cloud native deployment.
'/dbs'	Shows the information of databases.
'/dbs/<DB_ID>'	Shows the information of a database by database ID.
'/dbs/<DB_ID>/<TABLE_ID>'	Shows the information of tables by database ID.
'/dbs/<DB_ID>/<TABLE_ID>/partitions'	Shows the information of partitions by database ID and table ID.
'/transactions'	Shows the information of transactions by database.
'/transactions/<DB_ID>' Show the information of transactions by database ID.
'/transactions/<DB_ID>/running' Show the information of running transactions by database ID.
'/transactions/<DB_ID>/finished' Show the information of finished transactions by database ID.
'/jobs'	Shows the information of jobs.
'/statistic'	Shows the statistics of each database.
'/tasks'	Shows the total number of all generic tasks and the failed tasks.
'/cluster_balance'	Shows the load balance information.
'/routine_loads'	Shows the information of Routine Load.
'/colocation_group'	Shows the information of Colocate Join groups.
'/catalog'	Shows the information of catalogs.
'''



# ---------------------------------------------------------------------------
# MCP actions/search support
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_LIMIT = int(os.getenv('STARROCKS_SEARCH_LIMIT', '10'))
MAX_SEARCH_LIMIT = int(os.getenv('STARROCKS_SEARCH_MAX_LIMIT', '50'))


class ActionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    title: Optional[str] = None
    inputSchema: Dict[str, Any] | None = None
    annotations: Dict[str, Any] | None = None

    model_config = ConfigDict(extra='allow')


class ListActionsResult(mcp_types.PaginatedResult):
    actions: List[ActionDefinition]

    model_config = ConfigDict(extra='allow')


class ListActionsRequest(mcp_types.PaginatedRequest[Literal['actions/list']]):
    method: Literal['actions/list'] = 'actions/list'


class SearchRequestParams(mcp_types.PaginatedRequestParams):
    query: str
    limit: Optional[int] = Field(default=None, ge=1, le=MAX_SEARCH_LIMIT)
    cursor: Optional[str] = None

    model_config = ConfigDict(extra='allow')


class SearchRequest(mcp_types.PaginatedRequest[Literal['actions/search']]):
    method: Literal['actions/search'] = 'actions/search'
    params: SearchRequestParams


class SearchResultItem(BaseModel):
    id: str
    title: str
    content: str
    score: Optional[float] = None
    metadata: Dict[str, Any] | None = None

    model_config = ConfigDict(extra='allow')


class SearchResult(mcp_types.PaginatedResult):
    results: List[SearchResultItem]

    model_config = ConfigDict(extra='allow')


class _ExtendedClientRequest(
    RootModel[
        mcp_types.PingRequest
        | mcp_types.InitializeRequest
        | mcp_types.CompleteRequest
        | mcp_types.SetLevelRequest
        | mcp_types.GetPromptRequest
        | mcp_types.ListPromptsRequest
        | mcp_types.ListResourcesRequest
        | mcp_types.ListResourceTemplatesRequest
        | mcp_types.ReadResourceRequest
        | mcp_types.SubscribeRequest
        | mcp_types.UnsubscribeRequest
        | mcp_types.CallToolRequest
        | mcp_types.ListToolsRequest
        | ListActionsRequest
        | SearchRequest
    ]
):
    pass


class _ExtendedServerResult(
    RootModel[
        mcp_types.EmptyResult
        | mcp_types.InitializeResult
        | mcp_types.CompleteResult
        | mcp_types.GetPromptResult
        | mcp_types.ListPromptsResult
        | mcp_types.ListResourcesResult
        | mcp_types.ListResourceTemplatesResult
        | mcp_types.ReadResourceResult
        | mcp_types.CallToolResult
        | mcp_types.ListToolsResult
        | ListActionsResult
        | SearchResult
    ]
):
    pass


mcp_types.ActionDefinition = ActionDefinition
mcp_types.ListActionsRequest = ListActionsRequest
mcp_types.ListActionsResult = ListActionsResult
mcp_types.SearchRequestParams = SearchRequestParams
mcp_types.SearchRequest = SearchRequest
mcp_types.SearchResultItem = SearchResultItem
mcp_types.SearchResult = SearchResult
mcp_types.ClientRequest = _ExtendedClientRequest
mcp_types.ServerResult = _ExtendedServerResult

_ExtendedClientRequest.model_rebuild()
_ExtendedServerResult.model_rebuild()


_original_get_capabilities = LowLevelServer.get_capabilities


def _get_capabilities_with_actions(self, notification_options, experimental_capabilities):
    capabilities = _original_get_capabilities(self, notification_options, experimental_capabilities)
    if getattr(self, '_actions_enabled', False):
        try:
            setattr(capabilities, 'actions', {'listChanged': False})
        except Exception:
            capabilities.__dict__['actions'] = {'listChanged': False}
    return capabilities


LowLevelServer.get_capabilities = _get_capabilities_with_actions


def _truncate_text(value: Optional[str], max_length: int = 400) -> str:
    if not value:
        return ''
    text = value.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + '…'


def _compute_match_score(name: str, comment: Optional[str], query: str) -> float:
    normalized_name = name.lower()
    normalized_query = query.lower()
    comment_text = (comment or '').lower()
    if normalized_name == normalized_query:
        return 1.0
    if normalized_name.startswith(normalized_query):
        return 0.9
    if normalized_query in normalized_name:
        return 0.75
    if comment_text and normalized_query in comment_text:
        return 0.6
    return 0.3


def _decode_search_cursor(cursor: str, expected_hash: str) -> Dict[str, int]:
    state = {'table': 0, 'column': 0}
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode('utf-8')).decode('utf-8')
        payload = json.loads(decoded)
        if payload.get('hash') != expected_hash:
            return state
        table_offset = int(payload.get('table', 0))
        column_offset = int(payload.get('column', 0))
        if table_offset < 0 or column_offset < 0:
            return state
        state['table'] = table_offset
        state['column'] = column_offset
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f'Ignoring invalid search cursor: {exc}')
    return state


def _encode_search_cursor(table_offset: int, column_offset: int, query_hash: str) -> str:
    payload = {'hash': query_hash, 'table': table_offset, 'column': column_offset}
    raw = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    return base64.urlsafe_b64encode(raw).decode('utf-8')


def _build_table_result(row: List[Any], query: str) -> SearchResultItem:
    schema, table, comment, table_type = row
    title_prefix = 'View' if (table_type or '').upper() == 'VIEW' else 'Table'
    summary_lines = [f'Database: {schema}', f'Table: {table}']
    if table_type:
        summary_lines.append(f'Type: {table_type}')
    if comment:
        summary_lines.append(f'Comment: {_truncate_text(comment)}')
    metadata = {
        'type': 'table',
        'database': schema,
        'table': table,
        'tableType': table_type or 'BASE TABLE',
    }
    return SearchResultItem(
        id=f'table:{schema}.{table}',
        title=f'{title_prefix} {schema}.{table}',
        content='\n'.join(summary_lines),
        score=_compute_match_score(str(table), comment, query),
        metadata=metadata,
    )


def _build_column_result(row: List[Any], query: str) -> SearchResultItem:
    schema, table, column, data_type, comment, ordinal = row
    summary_lines = [
        f'Database: {schema}',
        f'Table: {table}',
        f'Column: {column}',
        f'Type: {data_type}',
    ]
    if comment:
        summary_lines.append(f'Comment: {_truncate_text(comment)}')
    metadata = {
        'type': 'column',
        'database': schema,
        'table': table,
        'column': column,
        'dataType': data_type,
        'ordinalPosition': ordinal,
    }
    return SearchResultItem(
        id=f'column:{schema}.{table}.{column}',
        title=f'Column {column} in {schema}.{table}',
        content='\n'.join(summary_lines),
        score=_compute_match_score(str(column), comment, query),
        metadata=metadata,
    )


def perform_search(query: str, limit: Optional[int], cursor: Optional[str]) -> tuple[List[SearchResultItem], Optional[str], Dict[str, Any]]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError('Search query cannot be empty.')

    normalized_limit = DEFAULT_SEARCH_LIMIT if limit is None else max(1, min(limit, MAX_SEARCH_LIMIT))
    lowered_query = normalized_query.lower()
    query_hash = hashlib.sha1(lowered_query.encode('utf-8')).hexdigest()
    cursor_state = {'table': 0, 'column': 0}
    if cursor:
        cursor_state = _decode_search_cursor(cursor, query_hash)

    pattern = f'%{lowered_query}%'
    table_fetch_limit = cursor_state['table'] + normalized_limit + 1
    column_fetch_limit = cursor_state['column'] + normalized_limit + 1

    table_sql = (
        "SELECT table_schema, table_name, IFNULL(table_comment, ''), table_type "
        "FROM information_schema.tables "
        "WHERE LOWER(table_name) LIKE %s OR LOWER(IFNULL(table_comment, '')) LIKE %s "
        "ORDER BY CASE WHEN LOWER(table_name) LIKE %s THEN 0 ELSE 1 END, table_schema, table_name "
        "LIMIT %s"
    )
    table_result = db_client.execute(
        table_sql,
        params=(pattern, pattern, pattern, table_fetch_limit),
        db='information_schema',
    )
    if not table_result.success:
        raise RuntimeError(f"Failed to search tables: {table_result.error_message}")

    table_rows = table_result.rows or []
    table_rows = table_rows[cursor_state['table'] : cursor_state['table'] + normalized_limit + 1]

    column_sql = (
        "SELECT table_schema, table_name, column_name, data_type, IFNULL(column_comment, ''), ordinal_position "
        "FROM information_schema.columns "
        "WHERE LOWER(column_name) LIKE %s OR LOWER(IFNULL(column_comment, '')) LIKE %s "
        "ORDER BY CASE WHEN LOWER(column_name) LIKE %s THEN 0 ELSE 1 END, table_schema, table_name, ordinal_position "
        "LIMIT %s"
    )
    column_result = db_client.execute(
        column_sql,
        params=(pattern, pattern, pattern, column_fetch_limit),
        db='information_schema',
    )
    if not column_result.success:
        raise RuntimeError(f"Failed to search columns: {column_result.error_message}")

    column_rows = column_result.rows or []
    column_rows = column_rows[cursor_state['column'] : cursor_state['column'] + normalized_limit + 1]

    results: List[SearchResultItem] = []
    table_consumed = 0
    for row in table_rows[:normalized_limit]:
        results.append(_build_table_result(row, normalized_query))
        table_consumed += 1
        if len(results) >= normalized_limit:
            break

    column_consumed = 0
    if len(results) < normalized_limit:
        remaining = normalized_limit - len(results)
        for row in column_rows[:remaining]:
            results.append(_build_column_result(row, normalized_query))
            column_consumed += 1

    has_more_tables = len(table_rows) > table_consumed
    has_more_columns = len(column_rows) > column_consumed
    next_cursor: Optional[str] = None
    if has_more_tables or has_more_columns:
        next_cursor = _encode_search_cursor(
            cursor_state['table'] + table_consumed,
            cursor_state['column'] + column_consumed,
            query_hash,
        )

    metadata = {
        'query': normalized_query,
        'limit': normalized_limit,
        'tablesReturned': table_consumed,
        'columnsReturned': column_consumed,
    }
    return results, next_cursor, metadata


def _search_result_to_payload(item: SearchResultItem) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        'id': item.id,
        'title': item.title,
        'text': item.content,
        'metadata': item.metadata or {},
    }
    if item.score is not None:
        payload['score'] = item.score
    return payload


@mcp.tool(
    name='search',
    description='Search StarRocks tables and columns using keywords.',
    tags={'search'},
    annotations={'mcp:action': 'search'},
)
def search_metadata(
    query: Annotated[str, Field(description='Keyword or phrase to search for in StarRocks metadata.')],
    limit: Annotated[Optional[int], Field(description='Maximum number of results to return.', ge=1, le=MAX_SEARCH_LIMIT)] = None,
    cursor: Annotated[Optional[str], Field(description='Opaque cursor from a previous search call.')] = None,
) -> ToolResult:
    """Expose perform_search as a structured MCP tool response."""

    results, next_cursor, metadata = perform_search(query, limit, cursor)
    payload_results = [_search_result_to_payload(item) for item in results]

    structured: Dict[str, Any] = {
        'results': payload_results,
        'meta': metadata,
    }
    if next_cursor:
        structured['nextCursor'] = next_cursor
        structured['next_cursor'] = next_cursor

    summary_lines = [f"Found {len(payload_results)} results for '{metadata['query']}'."]
    for entry in payload_results[:5]:
        summary_lines.append(f"- {entry['title']}")
    if next_cursor:
        summary_lines.append('More results available via nextCursor.')

    return ToolResult(
        content=[TextContent(type='text', text='\n'.join(summary_lines))],
        structured_content=structured,
    )


def _build_column_detail(row: List[Any]) -> tuple[str, Dict[str, Any]]:
    schema, table, column, data_type, comment, ordinal, is_nullable, default_value = row
    lines = [
        f'Database: {schema}',
        f'Table: {table}',
        f'Column: {column}',
        f'Type: {data_type}',
        f'Nullable: {is_nullable}',
    ]
    if default_value not in (None, ''):
        lines.append(f'Default: {default_value}')
    if comment:
        lines.append(f'Comment: {_truncate_text(comment, 2000)}')

    metadata = {
        'type': 'column',
        'database': schema,
        'table': table,
        'column': column,
        'dataType': data_type,
        'ordinalPosition': ordinal,
        'nullable': is_nullable,
    }
    if default_value not in (None, ''):
        metadata['default'] = default_value

    return '\n'.join(lines), metadata


def _parse_result_identifier(result_id: str) -> tuple[str, List[str]]:
    if result_id.startswith('table:'):
        remainder = result_id.split(':', 1)[1]
        parts = remainder.split('.', 1)
        if len(parts) != 2:
            raise ToolError(f'Invalid table identifier: {result_id}')
        return 'table', parts
    if result_id.startswith('column:'):
        remainder = result_id.split(':', 1)[1]
        parts = remainder.split('.', 2)
        if len(parts) != 3:
            raise ToolError(f'Invalid column identifier: {result_id}')
        return 'column', parts
    raise ToolError(f'Unsupported result identifier: {result_id}')


@mcp.tool(
    name='fetch',
    description='Fetch detailed metadata for a StarRocks search result identifier.',
    tags={'search'},
    annotations={'mcp:action': 'fetch'},
)
def fetch_metadata(
    id: Annotated[str, Field(description='Identifier returned from the search tool.')],
) -> ToolResult:
    result_type, parts = _parse_result_identifier(id)

    if result_type == 'table':
        schema, table = parts
        overview = _get_table_details(schema, table)
        metadata = {
            'type': 'table',
            'database': schema,
            'table': table,
        }
        structured = {
            'id': id,
            'title': f'Table {schema}.{table}',
            'text': overview,
            'metadata': metadata,
        }
        return ToolResult(content=[TextContent(type='text', text=overview)], structured_content=structured)

    schema, table, column = parts
    column_sql = (
        "SELECT table_schema, table_name, column_name, data_type, IFNULL(column_comment, ''), ordinal_position, "
        "is_nullable, IFNULL(column_default, '') "
        "FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s AND column_name = %s"
    )
    column_result = db_client.execute(
        column_sql,
        params=(schema, table, column),
        db='information_schema',
    )
    if not column_result.success:
        raise ToolError(f"Failed to fetch column metadata: {column_result.error_message}")

    rows = column_result.rows or []
    if not rows:
        raise ToolError(f'Column {schema}.{table}.{column} not found.')

    detail_text, metadata = _build_column_detail(rows[0])
    structured = {
        'id': id,
        'title': f'Column {column} in {schema}.{table}',
        'text': detail_text,
        'metadata': metadata,
    }
    return ToolResult(content=[TextContent(type='text', text=detail_text)], structured_content=structured)


SEARCH_ACTION_DEFINITION = ActionDefinition(
    name='search',
    title='Search StarRocks metadata',
    description='Search StarRocks tables and columns using a keyword.',
    inputSchema={
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Keyword or phrase to search for in table and column metadata.',
            },
            'limit': {
                'type': 'integer',
                'minimum': 1,
                'maximum': MAX_SEARCH_LIMIT,
                'description': 'Maximum number of results to return per page.',
            },
            'cursor': {
                'type': 'string',
                'description': 'Opaque cursor returned from a previous search call for pagination.',
            },
        },
        'required': ['query'],
        'additionalProperties': False,
    },
    annotations={'tags': ['search']},
)


async def _list_actions_handler(_: ListActionsRequest) -> mcp_types.ServerResult:
    return mcp_types.ServerResult(ListActionsResult(actions=[SEARCH_ACTION_DEFINITION]))


async def _search_handler(request: SearchRequest) -> mcp_types.ServerResult:
    params = request.params
    results, next_cursor, metadata = perform_search(params.query, params.limit, params.cursor)
    search_result = SearchResult(results=results, nextCursor=next_cursor, meta=metadata)
    return mcp_types.ServerResult(search_result)


def _register_search_action():
    if getattr(mcp._mcp_server, '_search_action_registered', False):
        return
    mcp._mcp_server._actions_enabled = True
    mcp._mcp_server._search_action_registered = True
    mcp._mcp_server._search_action = SEARCH_ACTION_DEFINITION
    mcp._mcp_server.request_handlers[ListActionsRequest] = _list_actions_handler
    mcp._mcp_server.request_handlers[SearchRequest] = _search_handler


_register_search_action()

@mcp.resource(uri="starrocks:///databases", name="All Databases", description="List all databases in StarRocks",
              mime_type="text/plain")
def get_all_databases() -> str:
    logger.debug("Fetching all databases")
    result = db_client.execute("SHOW DATABASES")
    logger.debug(f"Found {len(result.rows) if result.success and result.rows else 0} databases")
    return result.to_string()


@mcp.resource(uri="starrocks:///{db}/{table}/schema", name="Table Schema",
              description="Get the schema of a table using SHOW CREATE TABLE", mime_type="text/plain")
def get_table_schema(db: str, table: str) -> str:
    logger.debug(f"Fetching schema for table {db}.{table}")
    return db_client.execute(f"SHOW CREATE TABLE {db}.{table}").to_string()


@mcp.resource(uri="starrocks:///{db}/tables", name="Database Tables",
              description="List all tables in a specific database", mime_type="text/plain")
def get_database_tables(db: str) -> str:
    logger.debug(f"Fetching tables from database {db}")
    result = db_client.execute(f"SHOW TABLES FROM {db}")
    logger.debug(f"Found {len(result.rows) if result.success and result.rows else 0} tables in {db}")
    return result.to_string()


@mcp.resource(uri="proc:///{path*}", name="System internal information", description=SR_PROC_DESC,
              mime_type="text/plain")
def get_system_internal_information(path: str) -> str:
    logger.debug(f"Fetching system information for proc path: {path}")
    return db_client.execute(f"show proc '{path}'").to_string(limit=overview_length_limit)


def _get_table_details(db_name, table_name, limit=None):
    """
    Helper function to get description, sample rows, and count for a table.
    Returns a formatted string. Handles DB errors internally and returns error messages.
    """
    global global_table_overview_cache
    logger.debug(f"Fetching table details for {db_name}.{table_name}")
    output_lines = []

    full_table_name = f"`{table_name}`"
    if db_name:
        full_table_name = f"`{db_name}`.`{table_name}`"
    else:
        output_lines.append(
            f"Warning: Database name missing for table '{table_name}'. Using potentially incorrect context.")
        logger.warning(f"Database name missing for table '{table_name}'")

    count = 0
    output_lines.append(f"--- Overview for {full_table_name} ---")

    # 1. Get Row Count
    query = f"SELECT COUNT(*) FROM {full_table_name}"
    count_result = db_client.execute(query, db=db_name)
    if count_result.success and count_result.rows:
        count = count_result.rows[0][0]
        output_lines.append(f"\nTotal rows: {count}")
        logger.debug(f"Table {full_table_name} has {count} rows")
    else:
        output_lines.append(f"\nCould not determine total row count.")
        if not count_result.success:
            output_lines.append(f"Error: {count_result.error_message}")
            logger.error(f"Failed to get row count for {full_table_name}: {count_result.error_message}")

    # 2. Get Columns (DESCRIBE)
    if count > 0:
        query = f"DESCRIBE {full_table_name}"
        desc_result = db_client.execute(query, db=db_name)
        if desc_result.success and desc_result.column_names and desc_result.rows:
            output_lines.append(f"\nColumns:")
            output_lines.append(desc_result.to_string(limit=limit))
        else:
            output_lines.append("(Could not retrieve column information or table has no columns).")
            if not desc_result.success:
                output_lines.append(f"Error getting columns for {full_table_name}: {desc_result.error_message}")
                return "\n".join(output_lines)

        # 3. Get Sample Rows (LIMIT 3)
        query = f"SELECT * FROM {full_table_name} LIMIT 3"
        sample_result = db_client.execute(query, db=db_name)
        if sample_result.success and sample_result.column_names and sample_result.rows:
            output_lines.append(f"\nSample rows (limit 3):")
            output_lines.append(sample_result.to_string(limit=limit))
        else:
            output_lines.append(f"(No rows found in {full_table_name}).")
            if not sample_result.success:
                output_lines.append(f"Error getting sample rows for {full_table_name}: {sample_result.error_message}")

    overview_string = "\n".join(output_lines)
    # Update cache even if there were partial errors, so we cache the error message too
    cache_key = (db_name, table_name)
    global_table_overview_cache[cache_key] = overview_string
    return overview_string


# tools

@mcp.tool(description="Execute a SELECT query or commands that return a ResultSet" + description_suffix)
def read_query(query: Annotated[str, Field(description="SQL query to execute")],
               db: Annotated[str|None, Field(description="database")] = None) -> ToolResult:
    # return csv like result set, with column names as first row
    logger.info(f"Executing read query: {query[:100]}{'...' if len(query) > 100 else ''}")
    result = db_client.execute(query, db=db)
    if result.success:
        logger.info(f"Query executed successfully, returned {len(result.rows) if result.rows else 0} rows")
    else:
        logger.error(f"Query failed: {result.error_message}")
    return ToolResult(content=[TextContent(type='text', text=result.to_string(limit=10000))],
                      structured_content=result.to_dict())


@mcp.tool(description="Execute a DDL/DML or other StarRocks command that do not have a ResultSet" + description_suffix)
def write_query(query: Annotated[str, Field(description="SQL to execute")],
                db: Annotated[str|None, Field(description="database")] = None) -> ToolResult:
    logger.info(f"Executing write query: {query[:100]}{'...' if len(query) > 100 else ''}")
    result = db_client.execute(query, db=db)
    if not result.success:
        logger.error(f"Write query failed: {result.error_message}")
    elif result.rows_affected is not None and result.rows_affected >= 0:
        logger.info(f"Write query executed successfully, {result.rows_affected} rows affected in {result.execution_time:.2f}s")
    else:
        logger.info(f"Write query executed successfully in {result.execution_time:.2f}s")
    return ToolResult(content=[TextContent(type='text', text=result.to_string(limit=2000))],
                      structured_content=result.to_dict())

@mcp.tool(description="Analyze a query and get analyze result using query profile" + description_suffix)
def analyze_query(
        uuid: Annotated[
            str|None, Field(description="Query ID, a string composed of 32 hexadecimal digits formatted as 8-4-4-4-12")]=None,
        sql: Annotated[str|None, Field(description="Query SQL")]=None,
        db: Annotated[str|None, Field(description="database")] = None
) -> str:
    if uuid:
        logger.info(f"Analyzing query profile for UUID: {uuid}")
        return db_client.execute(f"ANALYZE PROFILE FROM '{uuid}'", db=db).to_string()
    elif sql:
        logger.info(f"Analyzing query: {sql[:100]}{'...' if len(sql) > 100 else ''}")
        return db_client.execute(f"EXPLAIN ANALYZE {sql}", db=db).to_string()
    else:
        logger.warning("Analyze query called without valid UUID or SQL")
        return f"Failed to analyze query, the reasons maybe: 1.query id is not standard uuid format; 2.the SQL statement have spelling error."


@mcp.tool(description="Run a query to get it's query dump and profile, output very large, need special tools to do further processing")
def collect_query_dump_and_profile(
        query: Annotated[str, Field(description="query to execute")],
        db: Annotated[str|None, Field(description="database")] = None
) -> ToolResult:
    logger.info(f"Collecting query dump and profile for query: {query[:100]}{'...' if len(query) > 100 else ''}")
    result : PerfAnalysisInput = db_client.collect_perf_analysis_input(query, db=db)
    if result.get('error_message'):
        status = f"collecting query dump and profile failed, query_id={result.get('query_id')} error_message={result.get('error_message')}"
        logger.warning(status)
    else:
        status = f"collecting query dump and profile succeeded, but it's only for user/tool, not for AI, query_id={result.get('query_id')}"
        logger.info(status)
    return ToolResult(
        content=[TextContent(type='text', text=status)],
        structured_content=result,
    )


def validate_plotly_expr(expr: str):
    """
    Validates a string to ensure it represents a single call to a method
    of the 'px' object, without containing other statements or imports,
    and ensures its arguments do not contain nested function calls.

    Args:
        expr: The string expression to validate.

    Raises:
        ValueError: If the expression does not meet the security criteria.
        SyntaxError: If the expression is not valid Python syntax.
    """
    # 1. Check for valid Python syntax
    try:
        tree = ast.parse(expr)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid Python syntax in expression: {e}") from e

    # 2. Check that the tree contains exactly one top-level node (statement/expression)
    if len(tree.body) != 1:
        raise ValueError("Expression must be a single statement or expression.")

    node = tree.body[0]

    # 3. Check that the single node is an expression
    if not isinstance(node, ast.Expr):
        raise ValueError(
            "Expression must be a single expression, not a statement (like assignment, function definition, import, etc.).")

    # 4. Get the actual value of the expression and check it's a function call
    expr_value = node.value
    if not isinstance(expr_value, ast.Call):
        raise ValueError("Expression must be a function call.")

    # 5. Check that the function being called is an attribute lookup (like px.scatter)
    if not isinstance(expr_value.func, ast.Attribute):
        raise ValueError("Function call must be on an object attribute (e.g., px.scatter).")

    # 6. Check that the attribute is being accessed on a simple variable name
    if not isinstance(expr_value.func.value, ast.Name):
        raise ValueError("Function call must be on a simple variable name (e.g., px.scatter, not obj.px.scatter).")

    # 7. Check that the simple variable name is 'px'
    if expr_value.func.value.id != 'px':
        raise ValueError("Function call must be on the 'px' object.")

    # Check positional arguments
    for i, arg_node in enumerate(expr_value.args):
        for sub_node in ast.walk(arg_node):
            if isinstance(sub_node, ast.Call):
                raise ValueError(f"Positional argument at index {i} contains a disallowed nested function call.")
    # Check keyword arguments
    for kw in expr_value.keywords:
        for sub_node in ast.walk(kw.value):
            if isinstance(sub_node, ast.Call):
                keyword_name = kw.arg if kw.arg else '<unknown>'
                raise ValueError(f"Keyword argument '{keyword_name}' contains a disallowed nested function call.")


@mcp.tool(description="using sql `query` to extract data from database, then using python `plotly_expr` to generate a chart for UI to display" + description_suffix)
def query_and_plotly_chart(
        query: Annotated[str, Field(description="SQL query to execute")],
        plotly_expr: Annotated[
            str, Field(description="a one function call expression, with 2 vars binded: `px` as `import plotly.express as px`, and `df` as dataframe generated by query `plotly_expr` example: `px.scatter(df, x=\"sepal_width\", y=\"sepal_length\", color=\"species\", marginal_y=\"violin\", marginal_x=\"box\", trendline=\"ols\", template=\"simple_white\")`")],
        db: Annotated[str|None, Field(description="database")] = None
) -> ToolResult:
    """
    Executes an SQL query, creates a Pandas DataFrame, generates a Plotly chart
    using the provided expression, encodes the chart as a base64 PNG image,
    and returns it along with optional text.

    Args:
        query: The SQL query string to execute.
        plotly_expr: A Python string expression using 'px' (plotly.express)
                     and 'df' (the DataFrame from the query) to generate a figure.
                     Example: "px.scatter(df, x='col1', y='col2')"
        db: Optional database name to execute the query in.

    Returns:
        A list containing types.TextContent and types.ImageContent,
        or just types.TextContent in case of an error or no data.
    """
    try:
        result = db_client.execute(query, db=db, return_format="pandas")
        errmsg = None
        if not result.success:
            errmsg = result.error_message
        elif result.pandas is None:
            errmsg = 'Query did not return data suitable for plotting.'
        else:
            df = result.pandas
            if df.empty:
                errmsg = 'Query returned no data to plot.'
        if errmsg:
            logger.warning(f"Query or data issue: {errmsg}")
            return ToolResult(
                content=[TextContent(type='text', text=f'Error: {errmsg}')],
                structured_content={'success': False, 'error_message': errmsg},
            )
        # Validate and evaluate the plotly expression using px and df
        local_vars = {'df': df}
        validate_plotly_expr(plotly_expr)
        fig : plotly.graph_objs.Figure = eval(plotly_expr, {"px": px}, local_vars)
        if os.getenv('STARROCKS_PLOTLY_JSON'):
            # return json representation of the figure for front-end rendering
            plot_json = json.loads(fig.to_json())
            structured_content = result.to_dict()
            structured_content['data'] = plot_json['data']
            structured_content['layout'] = plot_json['layout']
            summary = result.to_string()
            return ToolResult(
                content=[
                    TextContent(type='text', text=f'{summary}\nChart Generated for UI rendering'),
                ],
                structured_content=structured_content,
            )
        if not hasattr(fig, 'to_image'):
            raise ToolError(f"The evaluated expression did not return a Plotly figure object. Result type: {type(fig)}")
        img_bytes = fig.to_image(format='jpg', width=960, height=720)
        return ToolResult(
            content=[
               TextContent(type='text', text=f'dataframe data:\n{df}\nChart generated but for UI only'),
               Image(data=img_bytes, format="jpeg").to_image_content()
            ],
            structured_content=result.to_dict(),
        )
    except Exception as err:
        return ToolResult(
            content=[TextContent(type='text', text=f'Error: {err}')],
            structured_content={'success': False, 'error_message': str(err)},
        )


@mcp.tool(description="Get an overview of a specific table: columns, sample rows (up to 5), and total row count. Uses cache unless refresh=true" + description_suffix)
def table_overview(
        table: Annotated[str, Field(
            description="Table name, optionally prefixed with database name (e.g., 'db_name.table_name'). If database is omitted, uses the default database.")],
        refresh: Annotated[
            bool, Field(description="Set to true to force refresh, ignoring cache. Defaults to false.")] = False
) -> str:
    try:
        logger.info(f"Getting table overview for: {table}, refresh={refresh}")
        if not table:
            logger.error("Table overview called without table name")
            return "Error: Missing 'table' argument."

        # Parse table argument: [db.]<table>
        parts = table.split('.', 1)
        db_name = None
        table_name = None
        if len(parts) == 2:
            db_name, table_name = parts[0], parts[1]
        elif len(parts) == 1:
            table_name = parts[0]
            db_name = db_client.default_database  # Use default if only table name is given

        if not table_name:  # Should not happen if table_arg exists, but check
            logger.error(f"Invalid table name format: {table}")
            return f"Error: Invalid table name format '{table}'."
        if not db_name:
            logger.error(f"No database specified for table {table_name}")
            return f"Error: Database name not specified for table '{table_name}' and no default database is set."

        cache_key = (db_name, table_name)

        # Check cache
        if not refresh and cache_key in global_table_overview_cache:
            logger.debug(f"Using cached overview for {db_name}.{table_name}")
            return global_table_overview_cache[cache_key]

        logger.debug(f"Fetching fresh overview for {db_name}.{table_name}")
        # Fetch details (will also update cache)
        overview_text = _get_table_details(db_name, table_name, limit=overview_length_limit)
        return overview_text
    except Exception as e:
        # Reset connections on unexpected errors
        logger.exception(f"Unexpected error in table_overview for {table}")
        reset_db_connections()
        stack_trace = traceback.format_exc()
        return f"Unexpected Error executing tool 'table_overview': {type(e).__name__}: {e}\nStack Trace:\n{stack_trace}"

# comment out to prefer db_summary tool
#@mcp.tool(description="Get an overview (columns, sample rows, row count) for ALL tables in a database. Uses cache unless refresh=True" + description_suffix)
def db_overview(
        db: Annotated[str, Field(
            description="Database name. Optional: uses the default database if not provided.")] = None,
        refresh: Annotated[
            bool, Field(description="Set to true to force refresh, ignoring cache. Defaults to false.")] = False
) -> str:
    try:
        db_name = db if db else db_client.default_database
        logger.info(f"Getting database overview for: {db_name}, refresh={refresh}")
        if not db_name:
            logger.error("Database overview called without database name")
            return "Error: Database name not provided and no default database is set."

        # List tables in the database
        query = f"SHOW TABLES FROM `{db_name}`"
        result = db_client.execute(query, db=db_name)

        if not result.success:
            logger.error(f"Failed to list tables in database {db_name}: {result.error_message}")
            return f"Database Error listing tables in '{db_name}': {result.error_message}"

        if not result.rows:
            logger.info(f"No tables found in database {db_name}")
            return f"No tables found in database '{db_name}'."

        tables = [row[0] for row in result.rows]
        logger.info(f"Found {len(tables)} tables in database {db_name}")
        all_overviews = [f"--- Overview for Database: `{db_name}` ({len(tables)} tables) ---"]

        total_length = 0
        limit_per_table = overview_length_limit * (math.log10(len(tables)) + 1) // len(tables)  # Limit per table
        for table_name in tables:
            cache_key = (db_name, table_name)
            overview_text = None

            # Check cache first
            if not refresh and cache_key in global_table_overview_cache:
                logger.debug(f"Using cached overview for {db_name}.{table_name}")
                overview_text = global_table_overview_cache[cache_key]
            else:
                logger.debug(f"Fetching fresh overview for {db_name}.{table_name}")
                # Fetch details for this table (will update cache via _get_table_details)
                overview_text = _get_table_details(db_name, table_name, limit=limit_per_table)

            all_overviews.append(overview_text)
            all_overviews.append("\n")  # Add separator
            total_length += len(overview_text) + 1

        logger.info(f"Database overview completed for {db_name}, total length: {total_length}")
        return "\n".join(all_overviews)

    except Exception as e:
        # Catch any other unexpected errors during tool execution
        logger.exception(f"Unexpected error in db_overview for database {db}")
        reset_db_connections()
        stack_trace = traceback.format_exc()
        return f"Unexpected Error executing tool 'db_overview': {type(e).__name__}: {e}\nStack Trace:\n{stack_trace}"


@mcp.tool(description="Get an intelligent database summary with table prioritization, size information, and efficient caching. Uses SHOW DATA and information_schema for optimal performance" + description_suffix)
def db_summary(
        db: Annotated[str|None, Field(
            description="Database name. Optional: uses the default database if not provided.")] = None,
        limit: Annotated[int, Field(
            description="Output length limit in characters. Defaults to 10000. Higher values show more tables and details.")] = 10000,
        refresh: Annotated[bool, Field(
            description="Set to true to force refresh, ignoring cache. Defaults to false.")] = False
) -> str:
    try:
        db_name = db if db else db_client.default_database
        logger.info(f"Getting database summary for: {db_name}, limit={limit}, refresh={refresh}")
        
        if not db_name:
            logger.error("Database summary called without database name")
            return "Error: Database name not provided and no default database is set."
        
        # Use the database summary manager
        summary = db_summary_manager.get_database_summary(db_name, limit=limit, refresh=refresh)
        logger.info(f"Database summary completed for {db_name}")
        return summary
        
    except Exception as e:
        # Reset connections on unexpected errors
        logger.exception(f"Unexpected error in db_summary for database {db}")
        reset_db_connections()
        stack_trace = traceback.format_exc()
        return f"Unexpected Error executing tool 'db_summary': {type(e).__name__}: {e}\nStack Trace:\n{stack_trace}"


async def main():
    parser = argparse.ArgumentParser(description='StarRocks MCP Server')
    parser.add_argument('--mode', choices=['stdio', 'sse', 'http', 'streamable-http'], 
                        default=os.getenv('MCP_TRANSPORT_MODE', 'stdio'),
                        help='Transport mode (default: stdio)')
    parser.add_argument('--host', default='localhost',
                        help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=3000,
                        help='Server port (default: 3000)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting StarRocks MCP Server with mode={args.mode}, host={args.host}, port={args.port} default_db={db_client.default_database or 'None'}")
    
    if args.test:
        try:
            logger.info("Starting tool test")
            # Use the test version without tool wrapper
            result = db_client.execute("show databases").to_string()
            print("Result:")
            print(result)
            logger.info("Tool test completed")
        finally:
            reset_db_connections()
        return
    
    try:
        # Add CORS middleware for HTTP transports to allow web frontend access
        if args.mode in ['http', 'streamable-http', 'sse']:
            cors_middleware = [
                Middleware(
                    CORSMiddleware,
                    allow_origins=["*"],  # Allow all origins for development. In production, specify exact origins
                    allow_credentials=True,
                    allow_methods=["*"],  # Allow all HTTP methods
                    allow_headers=["*"],  # Allow all headers
                )
            ]
            logger.info(f"CORS enabled for {args.mode} transport - allowing all origins")
            await mcp.run_async(
                transport=args.mode, 
                host=args.host, 
                port=args.port,
                middleware=cors_middleware
            )
        else:
            await mcp.run_async(transport=args.mode, host=args.host, port=args.port)
    except Exception as e:
        logger.exception("Failed to start MCP server")
        raise


if __name__ == "__main__":
    asyncio.run(main())

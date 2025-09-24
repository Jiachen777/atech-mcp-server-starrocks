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
import asyncio
import math
import sys
import os
import traceback
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from typing import Annotated
from pydantic import Field
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

# StarRocks MCP API Server Implementation Plan

## Objectives
- Provide an HTTP API that mirrors the functionality exposed by the existing MCP tools and resources.
- Expose the service on the self-hosted instance at `https://starrocks-mcp.devicesformula.com/api`, backed by a local server listening on port `8002`.
- Produce an OpenAPI 3.1 specification that can be hosted at `/openapi.json` (resolving to `https://starrocks-mcp.devicesformula.com/api/openapi.json`) so GPT Actions can import the API via URL.
- Reuse the existing `DBClient` and summary utilities where possible to keep behaviour consistent with the MCP server.

## Proposed API Surface
The table below maps each MCP resource/tool to a planned REST endpoint.

| # | HTTP Method | Path | Description | Request Parameters / Body | Response Shape | Backing Implementation |
|---|-------------|------|-------------|---------------------------|----------------|------------------------|
| 1 | `GET` | `/databases` | List all databases (MCP resource `get_all_databases`). | Query params: none. | JSON object with `databases` array and metadata. | `get_all_databases` -> `db_client.execute("SHOW DATABASES")` |
| 2 | `GET` | `/databases/{database}/tables` | List tables in a database (`get_database_tables`). | Path: `database` (string). | JSON with `tables` array. | `get_database_tables` |
| 3 | `GET` | `/databases/{database}/tables/{table}/schema` | Retrieve `SHOW CREATE TABLE` output (`get_table_schema`). | Path: `database`, `table`. Optional query: `format` for raw text vs structured JSON. | JSON with `schema_sql` and optional parsed fields. | `get_table_schema` |
| 4 | `GET` | `/proc/{path}` | Fetch StarRocks `/proc` style diagnostics (`get_system_internal_information`). | Path: `path` supports nested segments. Optional `limit` query to cap text length. | JSON with `content` string and metadata. | `get_system_internal_information` |
| 5 | `POST` | `/queries/read` | Execute read-only SQL and return rows (`read_query`). | JSON body: `{ "query": string, "database"?: string }`. Optional pagination parameters in future. | JSON with `success`, `columns`, `rows`, `execution_time`. | `read_query` |
| 6 | `POST` | `/queries/write` | Execute DDL/DML statements (`write_query`). | JSON body: `{ "query": string, "database"?: string }`. | JSON with `success`, `rows_affected`, `execution_time`, `message`. | `write_query` |
| 7 | `POST` | `/queries/analyze` | Analyze a query via profile or `EXPLAIN ANALYZE` (`analyze_query`). | JSON body: either `{ "uuid": string, "database"?: string }` or `{ "sql": string, "database"?: string }`. | JSON with `success` and textual `analysis` output. | `analyze_query` |
| 8 | `POST` | `/queries/perf-collect` | Collect query dump and profile assets (`collect_query_dump_and_profile`). | JSON body: `{ "query": string, "database"?: string }`. | JSON with `query_id`, `profile`, `query_dump`, `error_message`. | `collect_query_dump_and_profile` |
| 9 | `POST` | `/queries/plot` | Execute SQL and render Plotly visualization (`query_and_plotly_chart`). | JSON body: `{ "query": string, "plotly_expr": string, "database"?: string }`. Optional `response_format` to request Plotly JSON or base64 image. | JSON containing dataframe sample, plot metadata, optional base64 image. | `query_and_plotly_chart` |
|10 | `GET` | `/databases/{database}/tables/{table}/overview` | Retrieve cached table overview (`table_overview`). | Path: `database`, `table`. Query: `refresh` (bool). | JSON summarizing row count, schema, samples, cache metadata. | `_get_table_details` + `table_overview` |
|11 | `GET` | `/databases/{database}/summary` | Intelligent DB summary (`db_summary`). | Path: `database`. Query: `limit` (int), `refresh` (bool). | JSON or structured text summary with prioritised tables. | `db_summary_manager.get_database_summary` |
|12 | `GET` | `/health` | Liveness check for the API server. | None. | `{ "status": "ok" }`. | New lightweight handler. |
|13 | `GET` | `/openapi.json` | Serve generated OpenAPI document for GPT Actions import. | None. | OpenAPI 3.1 JSON document. | FastAPI (or similar) auto generation. |

> **Note:** The legacy `db_overview` helper is not exposed by the MCP decorators. If parity with internal tooling is required, add an optional `GET /databases/{database}/overview` endpoint that reuses `_get_table_details` across all tables.

All external consumers will reach these endpoints via the `/api` prefix once requests traverse Caddy (e.g., `https://starrocks-mcp.devicesformula.com/api/databases`).

## Implementation Steps
1. **Project Layout**
   - Add a new module (e.g., `src/starrocks_api_server/`) containing a FastAPI application.
   - Reuse existing database utilities by importing from `mcp_server_starrocks.db_client` and `db_summary_manager`.

2. **Endpoint Development**
   - For each endpoint, wrap the existing MCP functions, converting string outputs into structured JSON where appropriate.
   - Ensure error handling mirrors current behaviour (surface `success` flag, error messages, and stack traces when needed).
   - Implement caching for table overviews using the existing `global_table_overview_cache` or FastAPI dependency state.

3. **Authentication & Security**
   - Require every request (except health and OpenAPI) to include an `Authorization: Bearer <token>` header; token value configurable via environment variable or secret file.
   - Return HTTP 401 for missing/invalid tokens, and ensure FastAPI dependencies make the token requirement reusable across endpoints.
   - Sanitize SQL inputs only insofar as current MCP tools allow arbitrary SQL from trusted users.

4. **OpenAPI Exposure**
   - Use FastAPI's automatic schema generation to expose `/openapi.json` (served through the `/api` path once proxied by Caddy).
   - Publish the OpenAPI document at `https://starrocks-mcp.devicesformula.com/api/openapi.json` for GPT Actions `import from url`.

5. **Reverse Proxy & Deployment**
   - Run the FastAPI/Uvicorn server bound to `0.0.0.0:8002` on the self-managed host.
   - Configure Caddy to forward `https://starrocks-mcp.devicesformula.com/api/*` to `http://127.0.0.1:8002/`, stripping the `/api` prefix (e.g., using `handle_path /api/*` with `reverse_proxy 127.0.0.1:8002`).
   - Ensure TLS termination, rate limiting, and logging are handled by Caddy, while application logs remain available locally.

6. **Testing Strategy**
   - Unit tests for each endpoint covering success & failure paths.
   - Integration tests using StarRocks dummy client (`STARROCKS_DUMMY_TEST`) for predictable responses.
   - Contract tests ensuring OpenAPI spec matches implemented responses.

7. **Deployment & GPT Actions Integration**
   - Containerize the API server for deployment or provide systemd service scripts for bare-metal hosts.
   - Document the final spec URL `https://starrocks-mcp.devicesformula.com/api/openapi.json` for GPT Actions `import from url` and include the required bearer token instructions.


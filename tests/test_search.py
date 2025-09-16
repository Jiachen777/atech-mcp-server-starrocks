import base64
import json

import pytest
from unittest import mock
from fastmcp.exceptions import ToolError

from src.mcp_server_starrocks import server
from src.mcp_server_starrocks.db_client import ResultSet


_ORIGINAL_EXECUTE = server.db_client.execute


def _patch_execute(monkeypatch, side_effect):
    mock_execute = mock.create_autospec(_ORIGINAL_EXECUTE, side_effect=side_effect)
    monkeypatch.setattr(server.db_client, "execute", mock_execute)
    return mock_execute


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


def _make_result(rows):
    return ResultSet(success=True, column_names=[], rows=rows, execution_time=0.01)


def _table_rows():
    return [
        ["analytics", "orders", "Orders table", "BASE TABLE"],
        ["sales", "daily_orders", "Daily orders view", "VIEW"],
        ["analytics", "customers", "Customer snapshot", "BASE TABLE"],
    ]


def _column_rows():
    return [
        ["analytics", "orders", "order_id", "BIGINT", "Primary key", 1],
        ["analytics", "orders", "order_date", "DATE", "", 2],
        ["sales", "daily_orders", "order_total", "DECIMAL", "", 3],
    ]


def _column_detail_row():
    return [
        "analytics",
        "orders",
        "order_id",
        "BIGINT",
        "Primary key",
        1,
        "NO",
        None,
    ]


def test_perform_search_returns_tables_and_columns(monkeypatch):
    def fake_execute(statement, params=None, db=None, return_format="raw"):
        if "information_schema.tables" in statement:
            return _make_result(_table_rows())
        if "information_schema.columns" in statement:
            return _make_result(_column_rows())
        raise AssertionError("Unexpected statement")

    mock_execute = _patch_execute(monkeypatch, fake_execute)

    results, next_cursor, meta = server.perform_search("order", limit=2, cursor=None)

    assert [r.id for r in results] == [
        "table:analytics.orders",
        "table:sales.daily_orders",
    ]
    assert all(r.metadata["type"] == "table" for r in results)
    assert next_cursor is not None
    assert meta == {
        "query": "order",
        "limit": 2,
        "tablesReturned": 2,
        "columnsReturned": 0,
    }
    assert mock_execute.call_count == 2
    table_call, column_call = mock_execute.call_args_list
    assert table_call.kwargs["db"] == "information_schema"
    assert "params" in table_call.kwargs
    assert column_call.kwargs["db"] == "information_schema"
    assert "params" in column_call.kwargs

    mock_execute.reset_mock()

    # next page continues from cursor and returns remaining tables/columns
    results2, next_cursor2, meta2 = server.perform_search("order", limit=2, cursor=next_cursor)

    assert [r.id for r in results2] == [
        "table:analytics.customers",
        "column:analytics.orders.order_id",
    ]
    assert results2[0].metadata["type"] == "table"
    assert results2[1].metadata["type"] == "column"
    assert next_cursor2 is not None
    assert meta2 == {
        "query": "order",
        "limit": 2,
        "tablesReturned": 1,
        "columnsReturned": 1,
    }
    assert mock_execute.call_count == 2

    mock_execute.reset_mock()

    # final page contains remaining column entries
    results3, next_cursor3, meta3 = server.perform_search("order", limit=2, cursor=next_cursor2)

    assert [r.id for r in results3] == [
        "column:analytics.orders.order_date",
        "column:sales.daily_orders.order_total",
    ]
    assert next_cursor3 is None
    assert meta3 == {
        "query": "order",
        "limit": 2,
        "tablesReturned": 0,
        "columnsReturned": 2,
    }


def test_perform_search_ignores_invalid_cursor(monkeypatch):
    def fake_execute(statement, params=None, db=None, return_format="raw"):
        if "information_schema.tables" in statement:
            return _make_result(_table_rows())
        if "information_schema.columns" in statement:
            return _make_result(_column_rows())
        raise AssertionError("Unexpected statement")

    _patch_execute(monkeypatch, fake_execute)

    # supply a malformed cursor that should be ignored gracefully
    bad_cursor = base64.urlsafe_b64encode(json.dumps({"table": -5}).encode()).decode()
    results, next_cursor, meta = server.perform_search("order", limit=2, cursor=bad_cursor)

    assert results[0].id == "table:analytics.orders"
    assert next_cursor is not None
    assert meta == {
        "query": "order",
        "limit": 2,
        "tablesReturned": 2,
        "columnsReturned": 0,
    }


def test_perform_search_validates_query(monkeypatch):
    with pytest.raises(ValueError):
        server.perform_search("   ", limit=5, cursor=None)


def test_perform_search_propagates_db_errors(monkeypatch):
    failure = ResultSet(success=False, error_message="boom", execution_time=0.01)

    def fake_execute(statement, params=None, db=None, return_format="raw"):
        return failure

    _patch_execute(monkeypatch, fake_execute)

    with pytest.raises(RuntimeError):
        server.perform_search("order", limit=1, cursor=None)


@pytest.mark.anyio("asyncio")
async def test_list_actions_includes_search():
    request = server.ListActionsRequest(params=None)
    result = await server._list_actions_handler(request)
    actions = result.root.actions
    assert any(action.name == "search" for action in actions)


@pytest.mark.anyio("asyncio")
async def test_search_handler_wraps_results(monkeypatch):
    def fake_execute(statement, params=None, db=None, return_format="raw"):
        if "information_schema.tables" in statement:
            return _make_result(_table_rows())
        if "information_schema.columns" in statement:
            return _make_result(_column_rows())
        raise AssertionError("Unexpected statement")

    _patch_execute(monkeypatch, fake_execute)

    params = server.SearchRequestParams(query="order", limit=2, cursor=None)
    request = server.SearchRequest(params=params)
    result = await server._search_handler(request)

    assert isinstance(result.root, server.SearchResult)
    assert len(result.root.results) == 2


def test_search_tool_returns_structured_results(monkeypatch):
    def fake_execute(statement, params=None, db=None, return_format="raw"):
        if "information_schema.tables" in statement:
            return _make_result(_table_rows())
        if "information_schema.columns" in statement:
            return _make_result(_column_rows())
        raise AssertionError("Unexpected statement")

    _patch_execute(monkeypatch, fake_execute)

    tool_result = server.search_metadata.fn("order", limit=2)
    structured = tool_result.structured_content

    assert structured["results"][0]["id"] == "table:analytics.orders"
    assert structured["meta"]["limit"] == 2
    assert "nextCursor" in structured


def test_fetch_tool_returns_table_overview(monkeypatch):
    monkeypatch.setattr(server, "_get_table_details", lambda schema, table, limit=None: "TABLE OVERVIEW")

    result = server.fetch_metadata.fn("table:analytics.orders")

    assert result.structured_content["metadata"]["type"] == "table"
    assert "TABLE OVERVIEW" in result.structured_content["text"]


def test_fetch_tool_returns_column_details(monkeypatch):
    def fake_execute(statement, params=None, db=None, return_format="raw"):
        if "information_schema.columns" in statement:
            return _make_result([_column_detail_row()])
        raise AssertionError("Unexpected statement")

    monkeypatch.setattr(server.db_client, "execute", fake_execute)

    result = server.fetch_metadata.fn("column:analytics.orders.order_id")

    metadata = result.structured_content["metadata"]
    assert metadata["type"] == "column"
    assert metadata["column"] == "order_id"
    assert "order_id" in result.structured_content["text"]


def test_fetch_tool_handles_missing_column(monkeypatch):
    empty = ResultSet(success=True, column_names=[], rows=[], execution_time=0.01)

    def fake_execute(statement, params=None, db=None, return_format="raw"):
        return empty

    monkeypatch.setattr(server.db_client, "execute", fake_execute)

    with pytest.raises(ToolError):
        server.fetch_metadata.fn("column:analytics.orders.unknown")


def test_fetch_tool_validates_identifier():
    with pytest.raises(ToolError):
        server.fetch_metadata.fn("invalid-prefix:abc")

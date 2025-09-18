import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.mcp_server_starrocks.db_client import ResultSet
from src.starrocks_api_server.app import (
    create_app,
    get_db_client_dependency,
    get_db_summary_manager_dependency,
)


class FakeDBClient:
    """Minimal stand-in for the production DB client used in API tests."""

    def __init__(self) -> None:
        self.default_database: Optional[str] = None
        self.execute_calls: List[Tuple[str, Optional[str], Optional[str]]] = []
        self.perf_calls: List[Tuple[str, Optional[str]]] = []

    def execute(
        self, query: str, db: Optional[str] = None, return_format: Optional[str] = None
    ) -> ResultSet:
        self.execute_calls.append((query, db, return_format))

        if return_format == "pandas":
            dataframe = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
            return ResultSet(
                success=True,
                column_names=["x", "y"],
                rows=dataframe.values.tolist(),
                execution_time=0.01,
                pandas=dataframe,
            )

        if query == "SHOW DATABASES":
            return ResultSet(
                success=True,
                column_names=["Database"],
                rows=[["default"], ["analytics"]],
                execution_time=0.01,
            )

        if query.startswith("SHOW TABLES"):
            return ResultSet(
                success=True,
                column_names=["Tables_in_db"],
                rows=[["events"], ["users"]],
                execution_time=0.01,
            )

        if query.startswith("SHOW CREATE TABLE"):
            return ResultSet(
                success=True,
                column_names=["Table", "Create Table"],
                rows=[["users", "CREATE TABLE users (...)" ]],
                execution_time=0.01,
            )

        if query.lower().startswith("show proc"):
            return ResultSet(
                success=True,
                column_names=["Key", "Value"],
                rows=[["status", "ok"]],
                execution_time=0.01,
            )

        if query.startswith("SELECT COUNT(*)"):
            return ResultSet(
                success=True,
                column_names=["count"],
                rows=[[3]],
                execution_time=0.01,
            )

        if query.startswith("DESCRIBE"):
            return ResultSet(
                success=True,
                column_names=["Field", "Type"],
                rows=[["id", "INT"], ["name", "VARCHAR"]],
                execution_time=0.01,
            )

        if query.startswith("SELECT *"):
            return ResultSet(
                success=True,
                column_names=["id", "name"],
                rows=[[1, "Alice"], [2, "Bob"]],
                execution_time=0.01,
            )

        if query.startswith("ANALYZE PROFILE"):
            return ResultSet(
                success=True,
                column_names=["analysis"],
                rows=[["Profile output"]],
                execution_time=0.01,
            )

        if query.startswith("EXPLAIN ANALYZE"):
            return ResultSet(
                success=True,
                column_names=["plan"],
                rows=[["Plan output"]],
                execution_time=0.01,
            )

        # Default fall-back for read/write style queries
        return ResultSet(
            success=True,
            column_names=["value"],
            rows=[[1]],
            rows_affected=1,
            execution_time=0.01,
        )

    def collect_perf_analysis_input(
        self, query: str, db: Optional[str] = None
    ) -> Dict[str, Any]:
        self.perf_calls.append((query, db))
        return {
            "query": query,
            "query_id": "abcd-1234",
            "profile": "profile-content",
            "query_dump": {"path": "/tmp/query.dump"},
        }


class FakeSummaryManager:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, int, bool]] = []

    def get_database_summary(self, database: str, limit: int, refresh: bool) -> str:
        self.calls.append((database, limit, refresh))
        return f"Summary for {database} with limit {limit}"


@pytest.fixture
def api_client():
    os.environ["STARROCKS_API_BEARER_TOKEN"] = "test-token"

    fake_db = FakeDBClient()
    fake_summary = FakeSummaryManager()

    app = create_app()
    app.dependency_overrides[get_db_client_dependency] = lambda: fake_db
    app.dependency_overrides[get_db_summary_manager_dependency] = lambda: fake_summary

    with TestClient(app) as client:
        yield client, fake_db, fake_summary

    app.dependency_overrides.clear()


def auth_headers() -> Dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def test_openapi_advertises_devicesformula_host():
    app = create_app()
    schema = app.openapi()
    assert schema["servers"] == [
        {"url": "https://starrocks-mcp.devicesformula.com"}
    ]


def test_health_endpoint(api_client):
    client, _, _ = api_client
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_authentication_required(api_client):
    client, _, _ = api_client
    response = client.get("/api/databases")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing bearer token."


def test_list_databases(api_client):
    client, _, _ = api_client
    response = client.get("/api/databases", headers=auth_headers())
    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["databases"] == ["default", "analytics"]


def test_list_tables_endpoint(api_client):
    client, _, _ = api_client
    response = client.get(
        "/api/databases/tables",
        headers=auth_headers(),
        params={"database": "default"},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["database"] == "default"
    assert payload["tables"] == ["events", "users"]


def test_get_table_schema_endpoint(api_client):
    client, _, _ = api_client
    response = client.get(
        "/api/tables/schema",
        headers=auth_headers(),
        params={"database": "default", "table": "users"},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["database"] == "default"
    assert payload["table"] == "users"
    assert "CREATE TABLE users" in payload["schema_sql"]


def test_get_proc_information_endpoint(api_client):
    client, _, _ = api_client
    response = client.get(
        "/api/proc",
        headers=auth_headers(),
        params={"path": "frontends"},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["path"] == "frontends"
    assert "status" in payload["content"]


def test_table_overview_caching(api_client):
    client, fake_db, _ = api_client

    params = {"database": "default", "table": "users"}
    first = client.get(
        "/api/tables/overview",
        headers=auth_headers(),
        params=params,
    )
    assert first.status_code == 200
    assert first.json()["cache"]["hit"] is False

    second = client.get(
        "/api/tables/overview",
        headers=auth_headers(),
        params=params,
    )
    assert second.status_code == 200
    assert second.json()["cache"]["hit"] is True
    # Ensure no additional DESCRIBE call during cache hit
    describe_calls = [call for call in fake_db.execute_calls if call[0].startswith("DESCRIBE")]
    assert len(describe_calls) == 1


def test_query_plot_json_format(api_client):
    client, fake_db, _ = api_client

    response = client.post(
        "/api/queries/plot",
        headers=auth_headers(),
        json={
            "query": "SELECT x, y FROM metrics",
            "plotly_expr": "px.scatter(df, x='x', y='y')",
            "response_format": "plotly_json",
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["chart"]["format"] == "plotly_json"
    # Ensure pandas result path executed
    assert any(call[2] == "pandas" for call in fake_db.execute_calls)


def test_database_summary_endpoint(api_client):
    client, _, fake_summary = api_client
    response = client.get(
        "/api/databases/summary",
        headers=auth_headers(),
        params={"database": "analytics", "limit": 5000, "refresh": True},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["summary"].startswith("Summary for analytics")
    assert fake_summary.calls[-1] == ("analytics", 5000, True)


def test_collect_perf_data(api_client):
    client, fake_db, _ = api_client
    response = client.post(
        "/api/queries/perf-collect",
        headers=auth_headers(),
        json={"query": "SELECT * FROM events"},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert fake_db.perf_calls[-1] == ("SELECT * FROM events", None)

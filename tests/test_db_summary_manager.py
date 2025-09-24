from src.mcp_server_starrocks.db_summary_manager import DatabaseSummaryManager
from src.mcp_server_starrocks.db_client import ResultSet


class DummyDBClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute(self, statement, db=None):
        self.calls.append((statement, db))
        assert statement == "SHOW DATABASES"
        return ResultSet(success=True, column_names=["Database"], rows=self.rows, execution_time=0.0)


def test_get_all_database_summaries_runs_for_each_database():
    class RecordingManager(DatabaseSummaryManager):
        def __init__(self, db_client):
            super().__init__(db_client)
            self.summary_calls = []

        def get_database_summary(self, database: str, limit: int = 10000, refresh: bool = False) -> str:  # type: ignore[override]
            self.summary_calls.append((database, limit, refresh))
            return f"summary-{database}"

    db_client = DummyDBClient([["db1"], ["db2"]])
    manager = RecordingManager(db_client)

    result = manager.get_all_database_summaries(limit=2048, refresh=True)

    assert "summary-db1" in result
    assert "summary-db2" in result
    assert manager.summary_calls == [("db1", 2048, True), ("db2", 2048, True)]
    assert db_client.calls == [("SHOW DATABASES", None)]


def test_get_all_database_summaries_handles_show_databases_error():
    class FailingDBClient:
        def execute(self, statement, db=None):
            return ResultSet(success=False, error_message="boom", execution_time=0.0)

    manager = DatabaseSummaryManager(FailingDBClient())
    result = manager.get_all_database_summaries()

    assert "Error: Failed to retrieve database list" in result


def test_get_all_database_summaries_handles_empty_result():
    manager = DatabaseSummaryManager(DummyDBClient([]))
    assert manager.get_all_database_summaries() == "No databases found."


def test_get_all_database_summaries_handles_summary_exception():
    class FaultyManager(DatabaseSummaryManager):
        def __init__(self, db_client):
            super().__init__(db_client)

        def get_database_summary(self, database: str, limit: int = 10000, refresh: bool = False) -> str:  # type: ignore[override]
            if database == "db2":
                raise RuntimeError("boom")
            return f"summary-{database}"

    db_client = DummyDBClient([["db1"], ["db2"]])
    manager = FaultyManager(db_client)
    result = manager.get_all_database_summaries()

    assert "summary-db1" in result
    assert "Error generating summary for database 'db2': boom" in result

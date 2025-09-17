"""Command-line entry point for running the StarRocks API server."""

import os
from typing import Optional

import uvicorn

from .app import app


def main(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Run the API server using Uvicorn."""

    server_host = host or os.getenv("STARROCKS_API_HOST", "0.0.0.0")
    server_port = port or int(os.getenv("STARROCKS_API_PORT", "8002"))

    uvicorn.run(
        app,
        host=server_host,
        port=server_port,
        log_level=os.getenv("STARROCKS_API_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()

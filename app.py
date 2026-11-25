"""Posit Connect / ASGI entry point.

    uvicorn app:app --host 0.0.0.0 --port 8000          # local
    rsconnect deploy fastapi --entrypoint app:app .     # Posit Connect (see docs/POSIT_CONNECT.md)

The module only exposes ``app``; all behaviour lives in the ``lumen`` package.
"""

from lumen.api.app import create_app

app = create_app()

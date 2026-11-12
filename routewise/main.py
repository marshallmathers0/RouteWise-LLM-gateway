"""ASGI entry point: `uvicorn routewise.main:app`."""

import logging

from routewise.api.app import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()

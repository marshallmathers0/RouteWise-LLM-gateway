"""Writes request logs. Sync SQLAlchemy run off the event loop (FastAPI background task).

Logging failures must never break a user request, so errors are caught and logged.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from routewise.storage.models import Base, RequestLog

log = logging.getLogger(__name__)


def make_engine(database_url: str) -> Engine:
    # psycopg3 driver for plain "postgresql://" URLs.
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **kwargs)


class RequestLogRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._session = sessionmaker(engine, expire_on_commit=False)

    def create_schema(self) -> None:
        # create_all is idempotent. For a project this size we skip Alembic migrations.
        Base.metadata.create_all(self.engine)

    def write(self, record: dict[str, Any]) -> None:
        try:
            with self._session.begin() as s:
                s.add(RequestLog(**record))
        except Exception:  # noqa: BLE001 - logging must never take the gateway down
            log.exception("failed to write request log")

    def healthy(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False

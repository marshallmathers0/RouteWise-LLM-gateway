"""Graceful degradation without Redis, and compatibility with the real OpenAI Python SDK."""

from __future__ import annotations

import socket
import threading
import time

import openai
import pytest
import uvicorn
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from routewise.router.baselines import LengthHeuristicRouter
from routewise.api.app import create_app
from tests.conftest import API_KEY, AUTH, chat_body


class BrokenRedis:
    """Every command fails as if the Redis server went away after startup."""

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            raise RedisConnectionError("redis is down")

        if name == "register_script":
            return lambda script: fail
        return fail


@pytest.mark.parametrize("redis_obj", [None, BrokenRedis()], ids=["no_redis", "redis_dies"])
def test_gateway_serves_without_redis(settings, app_config, fake_clients, repo, redis_obj):
    app = create_app(settings, app_config=app_config, clients=fake_clients, redis=redis_obj, repo=repo,
                     router=LengthHeuristicRouter(), semantic_cache=None)
    with TestClient(app) as c:
        for _ in range(2):
            r = c.post("/v1/chat/completions", json=chat_body(), headers=AUTH)
            assert r.status_code == 200
            assert r.json()["choices"][0]["message"]["content"] == "answer from groq"
        assert c.get("/health").json()["redis"] is False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(settings, app_config, fake_clients, fake_redis, repo):
    app = create_app(settings, app_config=app_config, clients=fake_clients, redis=fake_redis, repo=repo,
                     router=LengthHeuristicRouter(), semantic_cache=None)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


def test_official_openai_sdk_works(live_server):
    client = openai.OpenAI(base_url=live_server, api_key=API_KEY, max_retries=0)
    resp = client.chat.completions.create(
        model="auto", temperature=0, messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    assert resp.choices[0].message.content
    assert resp.usage.total_tokens == 30
    # Our extra metadata survives in the SDK object's extra fields.
    meta = resp.model_extra["routewise"]
    assert meta["route"] in {"cheap", "strong"} and meta["cache_status"]["status"] == "miss"

    stream = client.chat.completions.create(
        model="auto", temperature=0, stream=True, messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert text == resp.choices[0].message.content

    with pytest.raises(openai.AuthenticationError):
        openai.OpenAI(base_url=live_server, api_key="wrong", max_retries=0).chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model="no-such-model", messages=[{"role": "user", "content": "hi"}])

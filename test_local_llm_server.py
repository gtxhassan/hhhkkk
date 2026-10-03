"""
Test Local LLM Server — OpenAI-compatible proxy for Ollama (or any local LLM).

Exposes POST /v1/chat/completions that proxies to a local Ollama server.
Register http://localhost:9999/v1/chat/completions as your AI system's endpoint_url.

Usage (bare metal — Ollama must be running):
    pip install fastapi uvicorn openai traceloop-sdk
    python test_local_llm_server.py
    python test_local_llm_server.py model=codellama:7b port=8888

Usage (Docker — build context must be server/ so Dockerfile paths resolve):
    docker build -f server/Dockerfile.testlocalllm -t test-local-llm server/
    docker run --rm -p 9999:9999 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 test-local-llm

Usage (docker-compose — also starts the backend; reads server/.env):
    docker compose --profile monitoring up --build test-local-llm

Then create/update an AI system with:
    endpoint_url = "http://localhost:9999"       # bare metal / single host
    endpoint_url = "http://test-local-llm:9999"  # docker-compose (reached from the server container)
    model_name = "qwen2.5:3b"

Inline blocking: this server calls guard() before each LLM request, and again on
the model's reply when a linked guard asks for output checks. To see it block,
create a guardrail config with mode=block (e.g. the pii validator), link it to this
AI system, and set TRACELOOP_API_KEY to that system's ingest key. Set the guard's
apply_to to "output" or "both" to exercise the second check — the pre-call one
cannot see a reply that does not exist yet.
"""

import logging
import os
import secrets
import socket
import sys
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

try:
    from app.api.ai_governance.monitoring.sdk.governance_sdk import (
        GuardrailBlocked,
        guard,
        output_check_required,
    )
    from app.api.ai_governance.monitoring.sdk.governance_sdk import (
        init as governance_init,
    )
except ModuleNotFoundError:
    from governance_sdk import (  # type: ignore[no-redef]
        GuardrailBlocked,
        guard,
        output_check_required,
    )
    from governance_sdk import (
        init as governance_init,
    )

logger = logging.getLogger("test_local_llm_server")

TRACELOOP_ENDPOINT = os.getenv(
    "TRACELOOP_API_ENDPOINT", "http://localhost:8080/monitoring"
)
# The ingest key this server's traces and guard checks are sent under — exactly
# like a customer app configured with their own key. To attribute traces to a
# specific AI system (and for the app's "Test connection" to pass), set
# TRACELOOP_API_KEY to that system's ingest key. The random default is
# intentionally invalid, so traces and guard checks are rejected (401) until you
# configure a real key.
TRACELOOP_API_KEY = os.getenv(
    "TRACELOOP_API_KEY",
    "gov_" + secrets.token_urlsafe(32),
)

os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")

# Telemetry + guard, both keyed on this server's ingest key. Prefetches the
# guard policy (fail_mode) at startup; if the backend is down it fails open.
governance_init(
    api_key=TRACELOOP_API_KEY,
    endpoint=TRACELOOP_ENDPOINT,
    app_name="test-local-llm-server",
    disable_batch=True,
)

app = FastAPI(title="Test Local LLM Server")

# ── Parse CLI key=value args (e.g. model=qwen2.5:3b port=8888) ──────────
_cli: dict[str, str] = {}
for _arg in sys.argv[1:]:
    if "=" in _arg:
        _k, _v = _arg.split("=", 1)
        _cli[_k] = _v

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = _cli.get("model") or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
PORT = int(_cli.get("port") or os.getenv("PORT", "9999"))
ENDPOINT_HOST = os.getenv("ENDPOINT_HOST")

# Ollama's OpenAI-compatible endpoint lives at <base>/v1 — a dummy API key is
# required by the client constructor but Ollama ignores it.
client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


def _completion_payload(model, content, *, usage=None, finish_reason="stop"):
    """Build an OpenAI-compatible chat.completion response body."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _refusal(e: GuardrailBlocked) -> str:
    """Safe message returned in place of the blocked prompt or reply."""
    return (
        "This request was blocked by an organisational guardrail "
        f"({', '.join(e.failed) or e.action})."
    )


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    model = MODEL

    user_prompt = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    try:
        guard(user_prompt)
    except GuardrailBlocked as e:
        logger.warning(
            "prompt blocked by guardrail: failed=%s severity=%s", e.failed, e.severity
        )
        return _completion_payload(model, _refusal(e), finish_reason="content_filter")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": m.role, "content": m.content} for m in req.messages],
        temperature=1,
        max_completion_tokens=req.max_tokens,
    )
    answer = response.choices[0].message.content

    # Second check, on the model's reply. The pre-call guard only ever saw the
    # prompt, so an apply_to="output" guard had nothing to examine there — this is
    # the only point at which a bad answer can still be withheld rather than merely
    # recorded after the user has it.
    if output_check_required():
        try:
            guard(user_prompt, answer)
        except GuardrailBlocked as e:
            logger.warning(
                "response blocked by guardrail: failed=%s severity=%s", e.failed, e.severity
            )
            return _completion_payload(model, _refusal(e), finish_reason="content_filter")

    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return _completion_payload(model, answer, usage=usage)


@app.get("/health")
def health():
    return {"status": "ok"}


def _container_ip() -> str:
    """Best-effort container/host IP for log output."""
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        return "0.0.0.0"


if __name__ == "__main__":
    import uvicorn

    ip = _container_ip()
    hostname = socket.gethostname()
    endpoint_host = ENDPOINT_HOST or ip
    print("=" * 60, flush=True)
    print(f"Test Local LLM Server starting (model={MODEL})", flush=True)
    print(f"  Ollama endpoint    : {OLLAMA_BASE_URL}", flush=True)
    print(f"  container hostname : {hostname}", flush=True)
    print(f"  container IP       : {ip}", flush=True)
    print(f"  Endpoint URL       : http://{endpoint_host}:{PORT}", flush=True)
    print("=" * 60, flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)

"""
Test LLM Server — OpenAI-compatible proxy for red teaming.

Exposes POST /v1/chat/completions that proxies to the real OpenAI API.
Register http://localhost:9999/v1/chat/completions as your AI system's endpoint_url.

Usage (bare metal):
    pip install fastapi uvicorn openai traceloop-sdk
    export OPENAI_API_KEY=sk-...
    python test_llm_server.py

Usage (Docker — build context must be server/ so Dockerfile.testllm's COPY paths resolve):
    docker build -f server/Dockerfile.testllm -t test-llm server/
    docker run --rm -p 9999:9999 -e OPENAI_API_KEY=sk-... -e TEST_LLM_MODEL=gpt-4o-mini test-llm

Usage (docker-compose — also starts the backend it depends_on; reads server/.env):
    docker compose --profile monitoring up --build test-llm

Then create/update an AI system with:
    endpoint_url = "http://localhost:9999"   # bare metal / single host
    endpoint_url = "http://test-llm:9999"    # docker-compose (reached from the server container)
    model_name = "gpt-4o-mini"

Inline blocking: this server calls guard() before each OpenAI request, and again
on the model's reply when a linked guard asks for output checks. To see it block,
create a guardrail config with mode=block (e.g. the pii validator), link it to this
AI system, and set TRACELOOP_API_KEY to that system's ingest key. Set the guard's
apply_to to "output" or "both" to exercise the second check -- the pre-call one
cannot see a reply that does not exist yet.
"""

import logging
import os
import secrets
import socket
import uuid
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI


# governance_sdk.init() wraps Traceloop.init() (telemetry) AND arms the inline
# guard — guard()/GuardrailBlocked block a prompt before it reaches the vendor,
# and a reply before it reaches the caller. See
# app/api/ai_governance/monitoring/sdk/README.md.
# Local runs import it via the app package; the testllm Docker image copies the
# standalone module next to this file, so fall back to the flat import there.
try:
    from app.api.ai_governance.monitoring.sdk.governance_sdk import (
        GuardrailBlocked,
        guard,
        output_check_required,
        init as governance_init,
    )
except ModuleNotFoundError:
    from governance_sdk import (  # type: ignore[no-redef]
        GuardrailBlocked,
        guard,
        output_check_required,
        init as governance_init,
    )

logger = logging.getLogger("test_llm_server")

# Allow overriding the monitoring endpoint and API key via env so the same
# image works locally (host) and inside docker-compose (where the backend is
# reachable at http://server:8080).
TRACELOOP_ENDPOINT = os.getenv(
    "TRACELOOP_API_ENDPOINT", "https://api.deepnotch.tech/monitoring"
)
# The ingest key this server's traces are sent under — exactly like a customer app
# configured with their own key. To attribute traces to a specific AI system (and
# for the app's "Test connection" to pass), set TRACELOOP_API_KEY to that system's
# ingest key. The random default is intentionally invalid, so traces are rejected
# (401) until you configure a real key.
TRACELOOP_API_KEY = os.getenv(
    "TRACELOOP_API_KEY",
    "gov_5p_CPWnYCPcpw2cC8yKrq4eSiEzCfAsUp2D6ltbhc_s",
    # "gov_" + secrets.token_urlsafe(32),
)

# Disable the OTel PeriodicExportingMetricReader — it pings /v1/metrics every
# 60 seconds even when no one is using the monitoring pages. We only need traces.
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")

# Telemetry + guard, both keyed on this server's ingest key. Prefetches the
# guard policy (fail_mode) at startup; if the backend is down it fails open.
governance_init(
    api_key="gov_5p_CPWnYCPcpw2cC8yKrq4eSiEzCfAsUp2D6ltbhc_s",
    endpoint=TRACELOOP_ENDPOINT,
    app_name="test-llm-server",
    disable_batch=True,
)

app = FastAPI(title="Test LLM Server")
client = OpenAi()

# Default to a small NON-reasoning model. Reasoning models (gpt-5*, o*) count
# their hidden reasoning tokens against max_completion_tokens, so with the small
# budgets used here (16 for the connection test, 256 for test-guardrail, 512
# default) the budget is spent on reasoning and the visible content comes back
# empty (finish_reason="length") — i.e. traces show a prompt but no response.
MODEL = os.getenv("TEST_LLM_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "9999"))
# Hostname other containers use to reach this server. In docker-compose the
# backend reaches it at the service alias (test-llm); bare metal uses the
# container/host IP. This is the value to register as the AI system's
# endpoint_url, since the red team engine runs in the `server` container.
ENDPOINT_HOST = os.getenv("ENDPOINT_HOST")


class Message(BaseModel):
    role: str
    content: str


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE / MULTIMODAL CAPTURE SUPPORT — DISABLED FOR DEMO.
# This block logged each request to a JSONL file and decoded/saved any image
# payloads, to investigate whether garak probes send images. Finding: garak's REST
# generator is text-only, so visual probes (e.g. visual_jailbreak.FigStep) deliver
# nothing. Re-enable by uncommenting + switching Message.content back to a Union.
#
# from typing import Any, Dict, Union  # (Message.content: Union[str, List[Any], Dict[str, Any], None] = None)
# import base64
# import json
#
# CAPTURE_DIR = os.getenv("TESTLLM_CAPTURE_DIR", "/app/data/testllm")
# REQUEST_LOG = os.path.join(CAPTURE_DIR, "requests.jsonl")
# IMAGE_DIR = os.path.join(CAPTURE_DIR, "images")
#
# def _describe_content(content) -> str:
#     """Human-readable summary of a message's content for the request log."""
#     if isinstance(content, str):
#         return f"text(len={len(content)}): {content[:500]!r}"
#     if isinstance(content, list):
#         types = [p.get("type", "?") if isinstance(p, dict) else type(p).__name__ for p in content]
#         return f"MULTIMODAL parts={types}: {str(content)[:500]!r}"
#     if content is None:
#         return "None"
#     return f"{type(content).__name__}: {str(content)[:300]!r}"
#
# def _text_from_content(content) -> str:
#     """Extract just the text from a (possibly multimodal) content value."""
#     if isinstance(content, str):
#         return content
#     if isinstance(content, list):
#         return " ".join(
#             p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
#         )
#     return ""
#
# def _save_images(messages, req_id) -> list:
#     """Decode any OpenAI image_url parts (base64 data URIs) to files; record plain URLs."""
#     saved = []
#     for mi, m in enumerate(messages):
#         if not isinstance(m.content, list):
#             continue
#         for pi, part in enumerate(m.content):
#             if not (isinstance(part, dict) and part.get("type") == "image_url"):
#                 continue
#             url = (part.get("image_url") or {}).get("url", "")
#             if url.startswith("data:"):
#                 header, _, b64 = url.partition(",")
#                 ext = header.split("image/")[1].split(";")[0] if "image/" in header else "png"
#                 try:
#                     os.makedirs(IMAGE_DIR, exist_ok=True)
#                     path = os.path.join(IMAGE_DIR, f"{req_id}_m{mi}_p{pi}.{ext}")
#                     with open(path, "wb") as fh:
#                         fh.write(base64.b64decode(b64))
#                     saved.append(path)
#                 except Exception as e:
#                     logger.warning("failed to save image: %s", e)
#             else:
#                 saved.append(f"url:{url}")
#     return saved
#
# def _capture_request(req_id, messages) -> list:
#     """Append one request record (and saved-image paths) to the JSONL capture log."""
#     saved = []
#     try:
#         os.makedirs(CAPTURE_DIR, exist_ok=True)
#         saved = _save_images(messages, req_id)
#         record = {
#             "id": req_id,
#             "ts": datetime.now().isoformat(),
#             "messages": [{"role": m.role, "content": _describe_content(m.content)} for m in messages],
#             "images_saved": saved,
#         }
#         with open(REQUEST_LOG, "a") as fh:
#             fh.write(json.dumps(record) + "\n")
#     except Exception as e:
#         logger.warning("request capture failed: %s", e)
#     return saved
# ──────────────────────────────────────────────────────────────────────────────


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
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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

    # ── Diagnostic request logging + image capture: DISABLED FOR DEMO ──
    # (see the IMAGE / MULTIMODAL CAPTURE SUPPORT block above for the disabled code)
    # req_id = uuid.uuid4().hex[:12]
    # for m in req.messages:
    #     logger.info("  [%s] %s", m.role, _describe_content(m.content))
    # saved_images = _capture_request(req_id, req.messages)

    # ── Inline guardrail check — block before the prompt reaches the vendor ──
    # guard() POSTs the prompt to <endpoint>/guardrails/check and raises
    # GuardrailBlocked when a block-mode guard linked to this AI system trips
    # (or when fail_mode=closed and the check endpoint is unreachable). Put it
    # immediately before the LLM call so a blocked prompt never goes out.
    user_prompt = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    try:
        guard(user_prompt)
    except GuardrailBlocked as e:
        logger.warning("prompt blocked by guardrail: failed=%s severity=%s", e.failed, e.severity)
        return _completion_payload(model, _refusal(e), finish_reason="content_filter")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": m.role, "content": m.content} for m in req.messages],
        temperature=req.temperature,
        max_completion_tokens=req.max_tokens,
    )
    answer = response.choices[0].message.content

    # ── Second check, on the model's reply ──────────────────────────────────
    # The pre-call guard above only ever saw the prompt, so a guard configured
    # apply_to="output" had nothing to examine. This is the only point at which a
    # bad *answer* can still be withheld; skip it and output violations are merely
    # recorded after the user already has the text. output_check_required() reflects
    # the system's guards, so input-only setups pay no extra round-trip.
    if output_check_required():
        try:
            guard(user_prompt, answer)
        except GuardrailBlocked as e:
            logger.warning("response blocked by guardrail: failed=%s severity=%s", e.failed, e.severity)
            return _completion_payload(model, _refusal(e), finish_reason="content_filter")

    # The trace is emitted by Traceloop under this server's TRACELOOP_API_KEY (just
    # like a customer's instrumented app uses their own ingest key). Set that env to
    # the key generated for the AI system you want the traces attributed to.
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
    print(f"Test LLM Server starting (model={MODEL})", flush=True)
    print(f"  container hostname : {hostname}", flush=True)
    print(f"  container IP       : {ip}", flush=True)
    print(f"  Endpoint URL             : http://{endpoint_host}:{PORT}", flush=True)
    print("=" * 60, flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)

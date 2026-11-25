"""Chat adapters: the approved endpoint, and a deterministic mock for tests."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol
from urllib.parse import urlparse

from ..errors import FeatureDisabledError, LumenError


class ChatAdapter(Protocol):
    name: str

    def complete(self, system: str, user: str, timeout: float) -> str:
        """Return the model's reply text (expected to be one JSON object)."""
        ...


def endpoint_allowed(base_url: str) -> bool:
    """HTTPS only, except a loopback address for local development."""
    try:
        u = urlparse(base_url)
    except ValueError:
        return False
    if u.scheme == "https" and u.hostname:
        return True
    return u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1", "::1")


class OpenAICompatibleAdapter:
    """POST ``{base_url}/chat/completions`` with a server-side key.

    The key is read from an environment variable named by the administrator;
    it is never sent to the browser, logged or stored in a session.
    """

    name = "openai_compatible"

    def __init__(self, base_url: str, model: str, api_key_env: str) -> None:
        if not base_url or not model:
            raise LumenError("assistant_not_configured", "The assistant endpoint or model is not configured.", http_status=503)
        if not endpoint_allowed(base_url):
            raise LumenError("assistant_endpoint_refused", "The assistant endpoint must use HTTPS.", http_status=503)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._key_env = api_key_env

    def complete(self, system: str, user: str, timeout: float) -> str:
        key = os.environ.get(self._key_env, "")
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - scheme is checked in __init__
                data = json.loads(resp.read(2_000_000).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LumenError("assistant_upstream_error", f"The assistant endpoint answered with status {exc.code}.", http_status=502) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise LumenError("assistant_unreachable", "The assistant endpoint did not answer.", http_status=504) from None
        except ValueError:
            raise LumenError("assistant_bad_reply", "The assistant endpoint sent something that is not JSON.", http_status=502) from None
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            raise LumenError("assistant_bad_reply", "The assistant endpoint sent an unexpected reply.", http_status=502) from None


class MockAdapter:
    """Deterministic stand-in: restates the failed checks it was given, with citations.

    It lets the whole assistant path (context building, schema validation,
    grounding checks, API, interface) be tested without any network call.
    """

    name = "mock"

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.calls: list[dict[str, str]] = []

    def complete(self, system: str, user: str, timeout: float) -> str:
        self.calls.append({"system": system, "user": user})
        if self.reply is not None:
            return self.reply
        ctx = json.loads(user[user.index("{") : user.rindex("}") + 1])
        failed = [g for g in ctx.get("gates", []) if not g["passed"]]
        steps, cites = [], []
        for g in failed[:4]:
            if g.get("suggestion"):
                steps.append(g["suggestion"])
            cites.append({"source": f"gate:{g['code']}", "quote": g["message"]})
        for a in ctx.get("advice", [])[:2]:
            steps.append(a["text"])
            cites.append({"source": f"advice:{a['code']}", "quote": a["text"]})
        # the status itself is a statement of fact and needs its source like any other
        cites.insert(0, {"source": "status", "quote": f"{ctx.get('status')}: {ctx.get('status_meaning', '')}"})
        if failed:
            answer = "LUMEN reports status '" + str(ctx.get("status")) + "'. " + " ".join(g["message"] for g in failed[:3])
        else:
            answer = "LUMEN reports status '" + str(ctx.get("status")) + "'. No check is failing for this view."
        return json.dumps({"answer": answer, "steps": steps[:6], "citations": cites, "refusal": None})


def make_adapter(settings: Any) -> ChatAdapter:
    if not settings.llm_enabled:
        raise FeatureDisabledError("assistant_disabled", "The assistant is not enabled on this deployment.")
    if os.environ.get("LUMEN_LLM_MOCK", "") == "1":
        return MockAdapter()
    return OpenAICompatibleAdapter(settings.llm_base_url, settings.llm_model, settings.llm_api_key_env)

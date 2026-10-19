"""Telemetry: one PostHog $ai_generation event per call, and nothing the user wrote. See SPEC.md Telemetry."""
import os
import uuid
from typing import TYPE_CHECKING, Any

from any_llm.exceptions import AnyLLMError
from posthog import AsyncClient

from llm_master.io.response import Attempt, Response

if TYPE_CHECKING:
    from llm_master.pipeline import CallKind, CallReturn


class Telemetry:
    """PostHog's async client when POSTHOG_API_KEY is set; otherwise every event is dropped."""

    def __init__(self) -> None:
        key = os.environ.get("POSTHOG_API_KEY")
        # one retry of 5 s at most, so aclose() can't wait long on a PostHog that's down
        self._posthog = AsyncClient(key, host=os.environ.get("POSTHOG_HOST"), timeout=5, max_retries=1) if key else None

    def answered(self, kind: "CallKind", made: "CallReturn") -> None:
        properties: dict[str, Any] = {}
        if u := made.usage:
            properties |= {"$ai_input_tokens": u.input, "$ai_output_tokens": u.output,
                           "$ai_cache_read_input_tokens": u.cache_read, "$ai_cache_creation_input_tokens": u.cache_write,
                           "$ai_cache_creation_1h_input_tokens": u.cache_write_1h,
                           "$ai_cache_reporting_exclusive": False}  # input counts the cache tokens
        if made.cost is not None:
            properties |= {"$ai_total_cost_usd": made.cost, "$ai_cost_passthrough": True}  # genai-prices' price stands
        if isinstance(made, Response):
            properties["$ai_stop_reason"] = made.finish
        self._send(kind, made.attempts, made.latency, properties)

    def failed(self, kind: "CallKind", attempts: list[Attempt], latency: float, error: Exception) -> None:
        """error is what the call raises once every candidate failed."""
        properties: dict[str, Any] = {"$ai_is_error": True, "$ai_error": _safe(error)}
        if isinstance(error, AnyLLMError) and error.status_code:
            properties["$ai_http_status"] = error.status_code
        self._send(kind, attempts, latency, properties)

    async def aclose(self) -> None:
        """Sends what's queued."""
        if self._posthog is not None:
            await self._posthog.shutdown()

    def _send(self, kind: "CallKind", attempts: list[Attempt], latency: float, properties: dict[str, Any]) -> None:
        if self._posthog is None:
            return
        last = attempts[-1]
        errors = [f"{a.provider}/{a.model}: {_safe(e)}" for a in attempts for e in a.errors]
        self._posthog.capture("$ai_generation", properties={
            "$ai_trace_id": uuid.uuid4().hex, "$ai_span_name": kind, "$ai_provider": last.provider,
            "$ai_model": last.model, "$ai_latency": latency, "failed_tries": len(errors),
            "fallbacks": len(attempts) - 1, "errors": errors, "$process_person_profile": False, **properties})


def _safe(e: Exception) -> str:
    """The error's type, status and code, never its message: a message can quote the model or the prompt."""
    if isinstance(e, AnyLLMError):
        return " ".join(str(part) for part in (type(e).__name__, e.status_code, e.code) if part)
    return type(e).__name__

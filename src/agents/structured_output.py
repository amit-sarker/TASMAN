from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from src.BRS.retry import record_request_failure

_Response = TypeVar("_Response")
_Result = TypeVar("_Result")

def _brs_delivery_blocked(request: dict[str, Any]) -> bool:
    safety_finishes = {
        "CONTENT_FILTER", "SAFETY", "RECITATION", "BLOCKLIST",
        "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
    }
    block = request.get("block_reason")
    if block and str(block).upper() not in {"NONE", "UNSPECIFIED", "BLOCK_REASON_UNSPECIFIED"}:
        return True
    response = request.get("response_payload", {})
    for choice in [*response.get("choices", []), *response.get("candidates", [])]:
        if str(choice.get("finish_reason", "")).upper() in safety_finishes:
            return True
        if choice.get("message", {}).get("refusal"):
            return True
    return False

def brs_delivery_phase(
    request: Callable[[], _Response],
    validate: Callable[[_Response], _Result],
    correct: Callable[[], None],
    *, metadata: dict[str, Any], phase: str,
) -> _Result:
    metadata["delivery_phase"] = phase
    for output_attempt in range(2):
        start = len(metadata["requests"])
        try:
            result = validate(request())
            metadata["status"] = "completed"
            return result
        except ValueError as exc:
            requests = metadata["requests"]
            if len(requests) == start or "response_payload" not in requests[-1]:
                raise
            if _brs_delivery_blocked(requests[-1]):
                metadata["delivery_blocked"] = True
                raise BRSDeliveryBlockedError(str(exc)) from exc
            record_request_failure(metadata, exc)
            metadata.setdefault("output_recovery", []).append({
                "phase": phase, "output_attempt": output_attempt,
                "request_index": len(requests) - 1, "error": str(exc),
                "action": "correct" if output_attempt == 0 else "exhausted",
            })
            if output_attempt:
                metadata["output_recovery_exhausted"] = True
                raise
            correct()
        finally:
            for attempt in metadata["requests"][start:]:
                attempt["output_attempt"] = output_attempt
    raise AssertionError("unreachable BRS delivery correction state")

class StructuredOutputError(RuntimeError):
    pass

class GeminiStructuredOutputError(StructuredOutputError):
    pass

class KimiStructuredOutputError(StructuredOutputError):
    pass

class DeepSeekStructuredOutputError(StructuredOutputError):
    pass

class OpenAIStructuredOutputError(StructuredOutputError):
    pass

class BRSFeedbackError(RuntimeError):
    pass

class BRSDeliveryError(RuntimeError):
    pass

class BRSDeliveryBlockedError(RuntimeError):
    pass

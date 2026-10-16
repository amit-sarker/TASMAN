from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .retry import MAX_ATTEMPTS, is_retryable_failure_record

if TYPE_CHECKING:
    from .models import AgentFeedback

_EMPTY_SUMMARY_ERROR = re.compile(
    r"\AInvalid BRS agent feedback: 1 validation error for AgentFeedback\s+"
    r"scratchpad\s+Value error, each scratchpad summary must be a nonempty string\b"
)

_REQUEST_FIELDS = frozenset({
    "system", "user", "max_tokens", "temperature", "schema", "seed", "role",
})
_RESPONSE_FIELDS = frozenset({
    "raw_completion", "response_payload", "response", "provider_response",
    "response_id", "response_model", "finish_reason", "refusal", "block_reason",
    "choices", "candidates",
})

_DEEPSEEK_RECOVERABLE_FINISH_REASONS = frozenset({
    "", "insufficient_system_resource", "aborted",
})

def parse_saved_feedback(
    observer: Any, target_ids: list[str],
) -> AgentFeedback | None:
    from .feedback import build_agent_feedback_schema, parse_agent_feedback_output

    if (not isinstance(observer, Mapping) or observer.get("phase") != "failed"
            or not isinstance(observer.get("error"), str)
            or any(observer.get(field) is not None
                   for field in ("counts", "feedback", "scratchpad", "usage_snapshot"))
            or observer.get("local_validation_recovery") is not None):
        return None
    request = observer.get("request")
    schema = request.get("schema") if isinstance(request, Mapping) else None
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    feedback_schema = properties.get("feedback") if isinstance(properties, Mapping) else None
    ordered_targets = feedback_schema.get("required") if isinstance(feedback_schema, Mapping) else None
    if (not isinstance(ordered_targets, list)
            or any(not isinstance(target, str) for target in ordered_targets)
            or len(ordered_targets) != len(target_ids) or set(ordered_targets) != set(target_ids)):
        return None
    if (not isinstance(request, Mapping) or set(request) != _REQUEST_FIELDS
            or request.get("role") != "trust_feedback"
            or schema != build_agent_feedback_schema(ordered_targets)
            or any(not isinstance(request[field], str) or not request[field].strip()
                   for field in ("system", "user"))):
        return None
    metadata = observer.get("metadata")
    if (not isinstance(metadata, Mapping)
            or (metadata.get("status") != "completed"
                and not (metadata.get("status") is None and metadata.get("provider") == "stub"))
            or metadata.get("role") != "trust_feedback"
            or metadata.get("error") or metadata.get("error_type")
            or metadata.get("refusal") or metadata.get("block_reason")
            or metadata.get("incomplete")):
        return None
    dispatches = metadata.get("requests")
    if (not isinstance(dispatches, list) or not 1 <= len(dispatches) <= MAX_ATTEMPTS
            or any(not isinstance(row, Mapping) for row in dispatches)):
        return None
    for dispatch in dispatches[:-1]:
        if (dispatch.get("phase") != "feedback" or dispatch.get("status") != "failed"
                or not is_retryable_failure_record(dispatch)
                or any(field in dispatch for field in _RESPONSE_FIELDS)):
            return None
    last = dispatches[-1]
    raw = metadata.get("raw_completion")
    if (last.get("phase") != "feedback" or last.get("status") != "completed"
            or last.get("finish_reason") not in ("stop", "STOP")
            or not isinstance(raw, str) or not raw.strip()
            or last.get("raw_completion") != raw
            or last.get("refusal") or last.get("incomplete")
            or last.get("block_reason") not in (None, "BLOCK_REASON_UNSPECIFIED")
            or last.get("error") or last.get("error_type")):
        return None
    payload = last.get("response_payload")
    if (not isinstance(payload, Mapping)
            or ("choices" in payload and "candidates" in payload)):
        return None

    if "choices" in payload:
        choices = payload["choices"]
        if (not isinstance(choices, list) or len(choices) != 1
                or not isinstance(choices[0], Mapping)
                or choices[0].get("finish_reason") != "stop"):
            return None
        message = choices[0].get("message")
        if (not isinstance(message, Mapping) or message.get("content") != raw
                or message.get("refusal") or message.get("tool_calls")):
            return None
    elif "candidates" in payload:
        candidates = payload["candidates"]
        if (not isinstance(candidates, list) or len(candidates) != 1
                or not isinstance(candidates[0], Mapping)
                or candidates[0].get("finish_reason") != "STOP"
                or payload.get("block_reason") not in (None, "BLOCK_REASON_UNSPECIFIED")):
            return None
        content = candidates[0].get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else None
        if (not isinstance(parts, list) or not parts
                or any(not isinstance(part, Mapping) or set(part) != {"text"}
                       or not isinstance(part["text"], str) for part in parts)
                or "".join(part["text"] for part in parts) != raw):
            return None
    else:
        return None
    try:
        assessment = parse_agent_feedback_output(raw, target_ids, allow_wrapping=True)
    except (ValueError, RuntimeError, TypeError, OverflowError):
        return None
    return assessment

def parse_saved_blank_scratchpad_feedback(
    observer: Any, target_ids: list[str],
) -> AgentFeedback | None:
    if (not isinstance(observer, Mapping)
            or not isinstance(observer.get("error"), str)
            or not _EMPTY_SUMMARY_ERROR.match(observer["error"])):
        return None
    assessment = parse_saved_feedback(observer, target_ids)
    if assessment is None or not any(not text.strip() for text in assessment.scratchpad.values()):
        return None
    return assessment

def output_correction_error(observer: Any) -> str | None:
    if not isinstance(observer, Mapping) or observer.get("phase") != "failed":
        return "only a saved invalid response can receive output correction"
    attempt = observer.get("feedback_correction_attempt", 0)
    if type(attempt) is not int or attempt != 0:
        return "the single feedback-output correction has already been used"
    if any(observer.get(field) is not None for field in
           ("counts", "feedback", "scratchpad", "usage_snapshot")):
        return "a saved assessment must not be replaced"
    request = observer.get("request")
    if (not isinstance(request, Mapping) or set(request) != _REQUEST_FIELDS
            or request.get("role") != "trust_feedback"
            or any(not isinstance(request[field], str) or not request[field].strip()
                   for field in ("system", "user"))
            or not isinstance(request.get("schema"), dict)):
        return "the complete feedback request is unavailable"
    metadata = observer.get("metadata")
    calls = metadata.get("requests") if isinstance(metadata, Mapping) else None
    if (not isinstance(calls, list) or not 1 <= len(calls) <= MAX_ATTEMPTS
            or any(not isinstance(call, Mapping) for call in calls)):
        return "a recorded provider response is required for output correction"
    for call in calls[:-1]:
        if (call.get("phase") != "feedback" or call.get("status") != "failed"
                or not is_retryable_failure_record(call)
                or any(field in call for field in _RESPONSE_FIELDS)):
            return "earlier dispatch history is not an unanswered transient failure"
    last = calls[-1]
    if (last.get("phase") != "feedback" or metadata.get("refusal")
            or last.get("refusal")
            or metadata.get("block_reason") not in (None, "BLOCK_REASON_UNSPECIFIED")
            or last.get("block_reason") not in (None, "BLOCK_REASON_UNSPECIFIED")):
        return "refused or blocked feedback cannot receive a format correction"
    payload = last.get("response_payload")
    if not isinstance(payload, Mapping):
        return "the saved response envelope is unavailable"
    choices, candidates = payload.get("choices"), payload.get("candidates")
    if choices is not None and candidates is None:
        eligible_finish_reasons = {"stop", "length"}
        if metadata.get("provider") == "deepseek":
            eligible_finish_reasons.update(_DEEPSEEK_RECOVERABLE_FINISH_REASONS)
        if (not isinstance(choices, list) or len(choices) != 1
                or not isinstance(choices[0], Mapping)
                or not isinstance(choices[0].get("finish_reason"), str)
                or choices[0].get("finish_reason") not in eligible_finish_reasons):
            return "the response did not have a supported finish reason for text correction"
        message = choices[0].get("message")
        if (not isinstance(message, Mapping) or message.get("refusal")
                or message.get("tool_calls")
                or (message.get("content") is not None
                    and not isinstance(message.get("content"), str))):
            return "the response was not a feedback text answer"
    elif candidates is not None and choices is None:
        if (not isinstance(candidates, list) or len(candidates) != 1
                or not isinstance(candidates[0], Mapping)
                or candidates[0].get("finish_reason") not in ("STOP", "MAX_TOKENS")
                or payload.get("block_reason") not in (None, "BLOCK_REASON_UNSPECIFIED")):
            return "the response was not a normal or truncated text answer"
        content = candidates[0].get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else None
        if (not isinstance(parts, list) or any(not isinstance(part, Mapping)
                or set(part) != {"text"} or not isinstance(part["text"], str) for part in parts)):
            return "the response was not a feedback text answer"
    else:
        return "the response envelope is ambiguous"
    return None

def saved_feedback_is_blocked(observer: Any) -> bool:
    if (not isinstance(observer, Mapping) or observer.get("phase") != "failed"
            or any(observer.get(field) is not None for field in
                   ("counts", "feedback", "scratchpad", "usage_snapshot"))):
        return False
    metadata = observer.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    calls = metadata.get("requests")
    if not isinstance(calls, list) or not calls or not isinstance(calls[-1], Mapping):
        return False
    last = calls[-1]
    if last.get("phase") != "feedback":
        return False
    for item in (metadata, last):
        codes = item.get("error_codes", [])
        if isinstance(codes, list) and any(
            isinstance(code, str) and re.sub(r"[^a-z]", "", code.lower())
            in {"contentfilter", "contentfiltererror", "responsibleaipolicyviolation", "invalidprompt"}
            for code in codes
        ):
            return True
        if (item.get("refusal") or item.get("error_type") == "ContentFilterError"
                or item.get("finish_reason") in ("content_filter", "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT")
                or item.get("block_reason") not in (None, "BLOCK_REASON_UNSPECIFIED")):
            return True
    payload = last.get("response_payload")
    if not isinstance(payload, Mapping):
        return False
    choices = payload.get("choices")
    for item in choices if isinstance(choices, list) else []:
        if isinstance(item, Mapping) and (item.get("finish_reason") == "content_filter"
                or isinstance(item.get("message"), Mapping) and item["message"].get("refusal")):
            return True
    candidates = payload.get("candidates")
    for item in candidates if isinstance(candidates, list) else []:
        if isinstance(item, Mapping) and item.get("finish_reason") in (
                "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
            return True
    return payload.get("block_reason") not in (None, "BLOCK_REASON_UNSPECIFIED")

def transport_retry_error(observer: Any) -> str | None:
    if not isinstance(observer, Mapping) or observer.get("phase") != "failed":
        return "only recorded failed feedback can be explicitly retried; requested feedback is ambiguous"
    assessment_fields = ("counts", "feedback", "scratchpad", "usage_snapshot")
    if any(observer.get(field) is not None for field in assessment_fields):
        return "feedback already contains saved assessment or usage state"
    request = observer.get("request")
    if not isinstance(request, dict) or set(request) != _REQUEST_FIELDS:
        return "the complete original feedback request is unavailable"
    temperature = request["temperature"]
    try:
        valid_temperature = (not isinstance(temperature, bool)
                             and isinstance(temperature, (int, float))
                             and math.isfinite(temperature))
    except OverflowError:
        valid_temperature = False
    if (any(not isinstance(request[field], str) or not request[field].strip() for field in ("system", "user"))
            or request["role"] != "trust_feedback"
            or not isinstance(request["schema"], dict) or not request["schema"]
            or type(request["max_tokens"]) is not int or request["max_tokens"] < 1
            or not valid_temperature
            or (request["seed"] is not None and type(request["seed"]) is not int)):
        return "the saved feedback request is malformed"
    metadata = observer.get("metadata")
    if not isinstance(metadata, Mapping):
        return "transient failure metadata is unavailable"
    requests = metadata.get("requests")
    if (not isinstance(requests, list) or not 1 <= len(requests) <= MAX_ATTEMPTS
            or any(not isinstance(dispatch, Mapping) for dispatch in requests)):
        return "retry requires one to eight recorded feedback requests"
    for dispatch in requests:
        if dispatch.get("phase") != "feedback" or dispatch.get("status") != "failed":
            return "every feedback request must be a recorded failed attempt"
        if not is_retryable_failure_record(dispatch):
            return "a recorded error is not an eligible transient failure"
    last_error_type = requests[-1].get("error_type")
    if metadata.get("error_type", last_error_type) != last_error_type:
        return "the recorded final failure types disagree"

    sources = (observer, metadata, *requests)
    if any(field in source for source in sources for field in _RESPONSE_FIELDS):
        return "the feedback attempt contains a recorded provider response"
    if any(source.get(field) is not None for source in sources for field in assessment_fields):
        return "feedback already contains saved assessment or usage state"
    return None

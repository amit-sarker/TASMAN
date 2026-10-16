from __future__ import annotations

from typing import Any

_CALL_FIELDS = (
    "phase", "status", "error", "error_type", "inference_settings",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
    "latency_seconds", "finish_reason", "response_id", "response_model",
    "system_fingerprint", "request_seed", "refusal", "block_reason",
    "status_code", "response_headers", "error_codes", "retry_delay_seconds",
    "output_attempt", "validation_error", "feedback_output_invalid",
    "response_recovery_policy", "local_parse_repairs", "acknowledgment_status",
)

_ASSESSMENT_FIELDS = ("response_recovery_policy", "feedback_request_attempt", "retry_reason")
_FINAL_RESPONSE_FIELDS = (
    "validation_error", "feedback_output_invalid", "local_parse_repairs",
    "empty_scratchpad_targets", "acknowledgment_status", "delivery_blocked",
    "output_recovery_exhausted",
)

def log_saved_call(
    logger: Any, *, identity: dict, agent_id: str, role: str,
    metadata: dict, request: dict | None = None,
    scratchpad: dict[str, str] | None = None,
) -> None:
    if not hasattr(logger, "log_prompt_completion"):
        return
    calls = metadata.get("requests") or []
    if not calls and request is not None:
        calls = [{"phase": "feedback", "status": "unavailable"}]
    for index, call in enumerate(calls):
        payload = call.get("request_payload")
        if payload is None and request is not None:
            payload = {
                "system_prompt": request["system"], "user_prompt": request["user"],
                "schema": request["schema"], "max_tokens": request["max_tokens"],
                "temperature": request["temperature"], "seed": request["seed"],
            }
        response = call.get("response_payload")
        text = call.get("raw_completion")

        if text is None and call.get("status") == "completed":
            text = metadata.get("raw_completion") if role == "trust_feedback" else (
                metadata.get("acknowledgment") if call.get("phase") == "acknowledgment" else None
            )
        details = {key: call[key] for key in _CALL_FIELDS if key in call}
        details.update({key: metadata[key] for key in _ASSESSMENT_FIELDS if key in metadata})

        if index == len(calls) - 1:
            details.update({key: metadata[key] for key in _FINAL_RESPONSE_FIELDS if key in metadata})
        recovery = [event for event in metadata.get("output_recovery", [])
                    if event.get("request_index") == index]
        if recovery:
            details["output_recovery"] = recovery
        details["request_snapshot_available"] = payload is not None
        details["response_snapshot_available"] = response is not None or text is not None
        completed_scratchpad = scratchpad if call.get("status") == "completed" else None
        flattened = "\n".join(f"{target}: {value}" for target, value in (completed_scratchpad or {}).items())
        logger.log_prompt_completion(
            **identity, agent_id=agent_id, round_idx=-1, role=role,
            system_prompt=(payload or {}).get("system_prompt", ""),
            user_prompt=(payload or {}).get("user_prompt", ""),
            completion=text or "", raw_completion=text,
            request_payload=payload, response_payload=response,
            call_metadata=details, call_id=metadata.get("call_id"),
            request_idx=index, request_phase=call.get("phase"),
            request_status=call.get("status", "unavailable"),
            provider=metadata.get("provider"), model=metadata.get("model"),
            finish_reason=call.get("finish_reason"),
            prompt_tokens=call.get("prompt_tokens"),
            candidate_tokens=call.get("completion_tokens"),
            thinking_tokens=call.get("reasoning_tokens"),
            request_seed=call.get("request_seed"),
            system_fingerprint=call.get("system_fingerprint"),
            response_length=len(text) if isinstance(text, str) else None,
            structured_output=metadata.get("structured_output"),
            scratchpad_length=len(flattened) if completed_scratchpad else None,
            scratchpad_max_target_length=max(map(len, completed_scratchpad.values())) if completed_scratchpad else None,
        )

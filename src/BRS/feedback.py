from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .brs import validate_agent_ids
from .models import AgentFeedback, FeedbackPair

class FeedbackError(RuntimeError):
    pass

class FeedbackOutputError(FeedbackError):
    pass

class FeedbackCallError(FeedbackError):
    pass

def build_count_schema(agent_ids: Iterable[str]) -> dict[str, Any]:
    ids = validate_agent_ids(agent_ids)
    return {
        "type": "object",
        "properties": {
            agent_id: {
                "type": "object",
                "properties": {
                    "P": {"type": "integer", "minimum": 0},
                    "N": {"type": "integer", "minimum": 0},
                },
                "required": ["P", "N"],
                "additionalProperties": False,
            }
            for agent_id in ids
        },
        "required": ids,
        "additionalProperties": False,
    }

def build_agent_feedback_schema(agent_ids: Iterable[str]) -> dict[str, Any]:
    ids = validate_agent_ids(agent_ids)
    return {
        "type": "object",
        "properties": {
            "scratchpad": {
                "type": "object",
                "properties": {agent_id: {"type": "string", "minLength": 1} for agent_id in ids},
                "required": ids,
                "additionalProperties": False,
            },
            "feedback": build_count_schema(ids),
        },
        "required": ["scratchpad", "feedback"],
        "additionalProperties": False,
    }

def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")

def _unquoted_delimiters(text: str) -> list[tuple[int, str]]:
    delimiters: list[tuple[int, str]] = []
    in_string = escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif text.startswith("```", index):
            delimiters.append((index, "```"))
            index += 2
        elif character in "{}[]":
            delimiters.append((index, character))
        index += 1
    if in_string:
        raise ValueError("incomplete quoted string in feedback response")
    return delimiters

def _unwrap_json_envelope(text: str) -> str:
    delimiters = _unquoted_delimiters(text)
    object_index = next(
        (index for index, (_, token) in enumerate(delimiters) if token == "{"),
        None,
    )
    if object_index is None:
        raise ValueError("response contains no complete JSON object")
    prefix = delimiters[:object_index]
    if prefix and not (len(prefix) == 1 and prefix[0][1] == "```"):
        raise ValueError("ambiguous JSON-object presentation envelope")

    start = delimiters[object_index][0]
    stack: list[str] = []
    end = None
    suffix: list[tuple[int, str]] = []
    for index in range(object_index, len(delimiters)):
        position, token = delimiters[index]
        if token in ("{", "["):
            stack.append(token)
        elif token in ("}", "]"):
            expected = "{" if token == "}" else "["
            if not stack or stack.pop() != expected:
                raise ValueError("mismatched JSON object or array delimiters")
            if not stack:
                end = position + 1
                suffix = delimiters[index + 1:]
                break
        else:
            raise ValueError("unquoted code fence inside JSON object")
    if end is None:
        raise ValueError("incomplete JSON object")

    if prefix:
        fence_start = prefix[0][0]
        if text[fence_start + 3:start].strip().lower() not in ("", "json"):
            raise ValueError("JSON fence contains non-JSON content before the object")
        if len(suffix) != 1 or suffix[0][1] != "```":
            raise ValueError("ambiguous or incomplete JSON fences")
        if text[end:suffix[0][0]].strip():
            raise ValueError("JSON fence contains non-JSON content after the object")
    elif suffix:
        raise ValueError("ambiguous or incomplete additional JSON object")
    return text[start:end]

def _escape_literal_control_characters(text: str) -> str:
    output: list[str] = []
    in_string = escaped = False
    for character in text:
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
        elif escaped:
            output.append(character)
            escaped = False
        elif character == "\\":
            output.append(character)
            escaped = True
        elif character == '"':
            output.append(character)
            in_string = False
        elif ord(character) < 0x20:
            output.append(json.dumps(character)[1:-1])
        else:
            output.append(character)
    return "".join(output)

def _parse_json(
    text: str, *, allow_wrapping: bool = False, diagnostics: list[str] | None = None,
) -> Any:
    if not isinstance(text, str):
        raise ValueError("output must be JSON text")
    try:
        parsed = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except json.JSONDecodeError:
        if not allow_wrapping:
            raise
        candidate = _unwrap_json_envelope(text)
        if candidate != text.strip() and diagnostics is not None:
            diagnostics.append("presentation_envelope_removed")
        escaped = _escape_literal_control_characters(candidate)
        if escaped != candidate and diagnostics is not None:
            diagnostics.append("literal_control_characters_escaped")
        parsed = json.loads(escaped, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    return parsed

def _parse_agent_objects(
    text: str, agent_ids: Iterable[str], *, allow_wrapping: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    ids = validate_agent_ids(agent_ids)
    parsed = _parse_json(text, allow_wrapping=allow_wrapping)
    if not isinstance(parsed, dict) or set(parsed) != set(ids):
        raise ValueError("output must contain exactly the expected counterpart IDs")
    return ids, parsed

def parse_agent_feedback_output(
    text: str, agent_ids: Iterable[str], *, allow_wrapping: bool = False,
) -> AgentFeedback:
    result, _ = parse_agent_feedback_output_with_diagnostics(
        text, agent_ids, allow_wrapping=allow_wrapping,
    )
    return result

def parse_agent_feedback_output_with_diagnostics(
    text: str, agent_ids: Iterable[str], *, allow_wrapping: bool = False,
) -> tuple[AgentFeedback, list[str]]:
    try:
        ids = validate_agent_ids(agent_ids)
        diagnostics: list[str] = []
        result = AgentFeedback.model_validate(_parse_json(
            text, allow_wrapping=allow_wrapping, diagnostics=diagnostics,
        ))
        if set(result.feedback) != set(ids):
            raise ValueError("output must contain exactly the expected counterpart IDs")
        return AgentFeedback(
            scratchpad={agent_id: result.scratchpad[agent_id] for agent_id in ids},
            feedback={agent_id: result.feedback[agent_id] for agent_id in ids},
        ), diagnostics
    except (ValueError, TypeError, OverflowError) as exc:
        raise FeedbackOutputError(f"Invalid BRS agent feedback: {exc}") from exc

def parse_feedback_output(text: str, agent_ids: Iterable[str]) -> dict[str, FeedbackPair]:
    try:
        ids, parsed = _parse_agent_objects(text, agent_ids)
        return {agent_id: FeedbackPair.model_validate(parsed[agent_id]) for agent_id in ids}
    except (ValueError, TypeError, OverflowError) as exc:
        raise FeedbackOutputError(f"Invalid normalized BRS feedback: {exc}") from exc

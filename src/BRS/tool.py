from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .brs import validate_agent_ids
from .models import finite_number

TOOL_NAME = "get_counterpart_trust_scores"
TOOL_REQUEST_INSTRUCTION = (
    "The negotiation episode is complete. Call get_counterpart_trust_scores exactly once "
    "with no arguments ({}). The tool supplies your updated trust scores for every other "
    "party. Use those returned scores unchanged. Do not estimate or revise scores yourself."
)

def normalize_empty_tool_arguments(arguments: Any) -> tuple[str, str | None]:
    if not isinstance(arguments, str):
        raise ValueError("Tool arguments must be JSON text encoding an empty object")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, dict) and not parsed:
            return arguments, None
        raise ValueError("The trust tool accepts no arguments; expected {}")

    payload = arguments.strip(" \t\r\n")
    if not payload:
        return "{}", "empty_arguments"

    if re.fullmatch(r'\{[ \t\r\n]*\}[ \t\r\n]*""', payload):
        return "{}", "trailing_empty_string"
    fence = re.fullmatch(r"```(?:json)?[ \t\r\n]*(.*?)[ \t\r\n]*```", payload,
                         flags=re.IGNORECASE | re.DOTALL)
    if fence:
        payload = fence.group(1).strip(" \t\r\n")
    decoder = json.JSONDecoder()
    count = 0
    index = 0
    while index < len(payload):
        try:
            value, index = decoder.raw_decode(payload, index)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Tool arguments must contain only complete empty JSON objects"
            ) from exc
        if not isinstance(value, dict) or value:
            raise ValueError("The trust tool accepts no arguments; expected {}")
        count += 1
        while index < len(payload) and payload[index] in " \t\r\n":
            index += 1
    if count == 0:
        raise ValueError("Tool arguments must contain an empty JSON object")
    return "{}", "fenced_empty_arguments" if fence else "repeated_empty_objects"

def counterpart_scores(all_scores: Mapping[str, Mapping[str, float]], caller_id: str) -> dict[str, float]:
    if not isinstance(all_scores, Mapping):
        raise ValueError("all_scores must be a mapping")
    ids = validate_agent_ids(all_scores)
    if len(ids) < 2:
        raise ValueError("score matrix requires at least two agents")
    if not isinstance(caller_id, str) or caller_id not in all_scores:
        raise ValueError("caller_id must identify a scored episode participant")
    validated: dict[str, dict[str, float]] = {}
    for observer in ids:
        row = all_scores[observer]
        if not isinstance(row, Mapping) or set(row) != set(ids) - {observer}:
            raise ValueError("each score row must contain exactly its observer's counterparts")
        validated[observer] = {}
        for target, raw in row.items():
            value = finite_number(raw)
            if not 0 <= value <= 1:
                raise ValueError("trust scores must be in [0,1]")
            validated[observer][target] = value
    return validated[caller_id]

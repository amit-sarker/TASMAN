from __future__ import annotations

import json
import math
import re
import threading
import uuid
from typing import Any

from src.agents.structured_output import BRSDeliveryError, BRSFeedbackError
from src.BRS.tool import TOOL_NAME

class StubLLMClient:
    def __init__(
        self,
        canned: dict[tuple[str, int, str], str],
        fallback_proposer: str | None = None,
        fallback_voter: str | None = None,
        fallback_trust: str | None = None,
    ) -> None:
        self.canned = dict(canned)
        self.fallback_proposer = fallback_proposer
        self.fallback_voter = fallback_voter
        self.fallback_trust = fallback_trust
        self.feedback_calls = 0
        self.tool_calls = 0
        self.tool_requests = 0
        self.brs_feedback_output: str | None = None
        self.brs_feedback_error: Exception | None = None
        self.brs_tool_failure_phase: str | None = None
        self._brs_lock = threading.Lock()
        self._call_local = threading.local()

    def consume_call_metadata(self) -> dict[str, Any]:
        metadata = getattr(self._call_local, "metadata", None)
        self._call_local.metadata = None
        return dict(metadata or {})

    @property
    def last_call_metadata(self) -> dict[str, Any]:
        return dict(getattr(self._call_local, "metadata", None) or {})

    def complete_feedback_once(
        self, system: str, user: str, max_tokens: int = 16000,
        temperature: float = 0, schema: dict | None = None,
        seed: int | None = None, role: str = "trust_feedback",
    ) -> str:
        with self._brs_lock:
            self.feedback_calls += 1
        metadata: dict[str, Any] = {
            "provider": "stub", "model": "stub", "role": role,
            "call_id": uuid.uuid4().hex, "system_prompt": system, "user_prompt": user,
            "requests": [{"phase": "feedback", "request_seed": seed,
                          "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                          "request_payload": {
                              "model": "stub", "system_prompt": system, "user_prompt": user,
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user}],
                              "schema": schema, "max_tokens": max_tokens,
                              "temperature": temperature, "seed": seed,
                          }}],
        }
        self._call_local.metadata = metadata
        if self.brs_feedback_error is not None:
            metadata["requests"][-1].update(status="failed", error=str(self.brs_feedback_error),
                                             error_type=type(self.brs_feedback_error).__name__)
            raise BRSFeedbackError(str(self.brs_feedback_error)) from self.brs_feedback_error
        if self.brs_feedback_output is not None:
            output = self.brs_feedback_output
        else:
            targets = list((schema or {}).get("properties", {}).get("feedback", {}).get("properties", {}))
            if not targets:
                raise BRSFeedbackError("Stub feedback requires a fixed schema for the targets")
            output = json.dumps({
                "scratchpad": {target: "One supported positive and one negative observation." for target in targets},
                "feedback": {target: {"P": 1, "N": 1} for target in targets},
            })
        metadata["raw_completion"] = output
        metadata["requests"][-1].update(status="completed", raw_completion=output,
            finish_reason="stop", response_payload={"choices": [{"finish_reason": "stop",
                "message": {"role": "assistant", "content": output}}]})
        return output

    def trust_tool_roundtrip(
        self, system: str, user: str, scores: dict[str, float],
        max_tokens: int = 16000, temperature: float = 0.7,
        seed: int | None = None,
    ) -> dict[str, float]:
        authoritative = dict(scores)
        if not authoritative or any(
            not isinstance(k, str) or isinstance(v, bool)
            or not isinstance(v, (int, float)) or not math.isfinite(v)
            or not 0 <= v <= 1 for k, v in authoritative.items()
        ):
            raise BRSDeliveryError("Stub tool requires finite counterpart scores in [0,1]")
        with self._brs_lock:
            self.tool_calls += 1
            self.tool_requests += 1
            call_id = f"stub-trust-{self.tool_calls}"
        metadata: dict[str, Any] = {
            "provider": "stub", "model": "stub", "role": "trust_tool",
            "call_id": uuid.uuid4().hex, "system_prompt": system, "user_prompt": user,
            "tool_name": TOOL_NAME, "tool_call_id": call_id,
            "tool_arguments": {}, "tool_result": authoritative,
            "requests": [{"phase": "tool_call", "request_seed": seed,
                          "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}],
        }
        tool_call = {"id": call_id, "type": "function",
                     "function": {"name": metadata["tool_name"], "arguments": "{}"}}
        tool_message = {"role": "assistant", "tool_calls": [tool_call]}
        request_payload = {
            "model": "stub", "system_prompt": system, "user_prompt": user,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "tools": [{"type": "function", "function": {
                "name": metadata["tool_name"], "parameters": {"type": "object", "properties": {},
                    "required": [], "additionalProperties": False},
            }}],
            "tool_choice": {"type": "function", "function": {"name": metadata["tool_name"]}},
            "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
        }
        metadata["requests"][0]["request_payload"] = request_payload
        self._call_local.metadata = metadata
        if self.brs_tool_failure_phase == "call":
            metadata["requests"][-1].update(status="failed", error="Injected stub tool-call failure",
                                             error_type="BRSDeliveryError")
            raise BRSDeliveryError("Injected stub tool-call failure")
        metadata["requests"][-1].update(status="completed", raw_completion="",
            finish_reason="tool_calls", response_payload={"choices": [{
                "finish_reason": "tool_calls", "message": tool_message}]})
        with self._brs_lock:
            self.tool_requests += 1
        metadata["requests"].append({"phase": "acknowledgment", "request_seed": seed,
                                     "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                                     "request_payload": {**request_payload, "tool_choice": "none",
                                         "messages": [*request_payload["messages"], tool_message,
                                             {"role": "tool", "tool_call_id": call_id,
                                              "content": json.dumps(authoritative, sort_keys=True)}]}})
        if self.brs_tool_failure_phase == "ack":
            metadata["requests"][-1].update(status="failed", error="Injected stub acknowledgment failure",
                                             error_type="BRSDeliveryError")
            raise BRSDeliveryError("Injected stub acknowledgment failure")
        metadata["acknowledgment"] = "Received."
        metadata["requests"][-1].update(status="completed", raw_completion="Received.",
            finish_reason="stop", response_payload={"choices": [{"finish_reason": "stop",
                "message": {"role": "assistant", "content": "Received."}}]})
        return dict(authoritative)

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        agent_id = kwargs.get("agent_id")
        round_idx = kwargs.get("round_idx")
        role = kwargs.get("role")
        if agent_id is not None and round_idx is not None and role is not None:
            key = (str(agent_id), int(round_idx), str(role))
            if key in self.canned:
                return self.canned[key]

        inferred_role = role or "propose"
        if inferred_role == "propose" and self.fallback_proposer is not None:
            return self._schema_compatible_proposal(
                self.fallback_proposer,
                kwargs.get("schema"),
            )
        if inferred_role == "vote" and self.fallback_voter is not None:
            if "cover identity" in system or "public picture is a cover" in system:
                return json.dumps({
                    "scratchpad": "Dry-run malicious voter: no-deal BATNA is better than any real deal, reject unless other rejections are already certain.",
                    "public_answer": "I cannot support this proposal as stated. It does not adequately address my party's core concerns.",
                    "vote": False,
                    "plan": "Look for a wedge that at least one other party will also reject.",
                })
            return self.fallback_voter
        if inferred_role == "trust":
            if self.fallback_trust is not None:
                return self.fallback_trust
            schema = kwargs.get("schema")
            schema_properties = (
                schema.get("properties", {}) if isinstance(schema, dict) else {}
            )
            targets_match = re.search(r"Parties to score:\s*(.+)", user)
            target_ids: list[str] = []
            if "episode_signals" in schema_properties:
                target_ids = list(
                    schema_properties.get("episode_signals", {}).get("required", [])
                )
            elif "weights_rationale" in schema_properties:
                target_ids = list(
                    schema_properties.get("scratchpad", {}).get("required", [])
                )
            elif targets_match:
                target_ids = [t.strip() for t in targets_match.group(1).split(",") if t.strip()]

            if "episode_signals" in user:
                scratchpad_schema = schema_properties.get("scratchpad", {})
                structured_scratchpad = (
                    isinstance(scratchpad_schema, dict)
                    and scratchpad_schema.get("type") == "object"
                )
                scratchpad: str | dict[str, str]
                if structured_scratchpad:
                    scratchpad = {
                        aid: (
                            "Dry-run freeform trust update: neutral evidence "
                            "for this party."
                        )
                        for aid in target_ids
                    }
                else:
                    scratchpad = (
                        "Dry-run freeform trust update: neutral signal for all parties."
                    )
                return json.dumps({
                    "scratchpad": scratchpad,
                    "episode_signals": {aid: 0.5 for aid in target_ids},
                    "metrics_weights": {"cooperation": 1.0},
                    "trust_scores": {aid: 0.5 for aid in target_ids},
                })

            return json.dumps({
                "scratchpad": {
                    aid: "Dry-run: b=0 C=0.5 O=0.5 T=0 F=0 -> S=0.3 -> trust_new=0.5. neutral."
                    for aid in target_ids
                },
                "metrics_weights": {
                    "vote_inconsistency": 0.4,
                    "counterfactual_blame": 0.2,
                    "outlier_rejection": 0.2,
                    "targeted_failure": 0.1,
                    "cooperation": 0.1,
                },
                "weights_rationale": "Dry-run: neutral weighting; no obstruction signal in the history.",
                "trust_scores": {aid: 0.5 for aid in target_ids},
            })
        raise KeyError(
            f"No canned stub response for agent={agent_id!r}, round={round_idx!r}, role={role!r}"
        )

    @staticmethod
    def _schema_compatible_proposal(fallback: str, schema: Any) -> str:
        if not isinstance(schema, dict):
            return fallback
        try:
            payload = json.loads(fallback)
            deal_schema = schema["properties"]["deal"]
            properties = deal_schema["properties"]
            required = deal_schema.get("required", list(properties))
        except (KeyError, TypeError, json.JSONDecodeError):
            return fallback

        if not isinstance(payload, dict) or not isinstance(properties, dict):
            return fallback
        deal = payload.get("deal")
        valid = isinstance(deal, dict) and set(deal) == set(required)
        if valid:
            for issue_id in required:
                options = properties.get(issue_id, {}).get("enum", [])
                if not options or deal.get(issue_id) not in options:
                    valid = False
                    break
        if valid:
            return fallback

        generated_deal: dict[str, str] = {}
        for issue_id in required:
            options = properties.get(issue_id, {}).get("enum", [])
            if not options:
                return fallback
            generated_deal[str(issue_id)] = str(options[0])
        payload["deal"] = generated_deal
        return json.dumps(payload)

    def usage_summary(self) -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

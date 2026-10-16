from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import importlib.metadata
import json
import logging
import math
import re
import threading
import time
import uuid
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.llm_client import ContentFilterError
from src.agents.structured_output import (
    BRSDeliveryError,
    BRSFeedbackError,
    GeminiStructuredOutputError,
    brs_delivery_phase,
)
from src.BRS.tool import TOOL_NAME
from src.BRS.retry import call_with_brs_retries, record_request_failure

_log = logging.getLogger(__name__)

class GeminiSchemaValidationError(ValueError):
    pass

class GeminiScratchpadRecoveryError(RuntimeError):
    pass

_SUPPORTED_SCHEMA_KEYWORDS = {
    "$anchor",
    "$defs",
    "$id",
    "$ref",
    "additionalProperties",
    "anyOf",
    "description",
    "enum",
    "format",
    "items",
    "maxItems",
    "maximum",
    "minItems",
    "minimum",
    "oneOf",
    "prefixItems",
    "properties",
    "propertyOrdering",
    "required",
    "title",
    "type",
}

_UNSUPPORTED_SCHEMA_KEYWORDS = {
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maxLength",
    "maxProperties",
    "minLength",
    "minProperties",
    "multipleOf",
    "pattern",
    "patternProperties",
    "uniqueItems",
}

_SAFETY_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
}
_SAFETY_BLOCK_REASONS = {
    "SAFETY",
    "OTHER",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "IMAGE_SAFETY",
    "MODEL_ARMOR",
    "JAILBREAK",
}

def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()

def _normalise_schema_node(node: Any, *, path: str = "$") -> Any:
    if isinstance(node, list):
        return [
            _normalise_schema_node(value, path=f"{path}[{index}]")
            for index, value in enumerate(node)
        ]
    if not isinstance(node, dict):
        return copy.deepcopy(node)

    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_SCHEMA_KEYWORDS:
            continue
        if key not in _SUPPORTED_SCHEMA_KEYWORDS:
            raise ValueError(
                f"Unsupported Gemini JSON Schema keyword {key!r} at {path}"
            )

        if key in {"properties", "$defs"}:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object at {path}")
            result[key] = {
                property_name: _normalise_schema_node(
                    property_schema,
                    path=f"{path}.{key}.{property_name}",
                )
                for property_name, property_schema in value.items()
            }
        elif key == "additionalProperties" and isinstance(value, dict):
            result[key] = _normalise_schema_node(
                value,
                path=f"{path}.additionalProperties",
            )
        elif key in {"items"} and isinstance(value, dict):
            result[key] = _normalise_schema_node(value, path=f"{path}.{key}")
        elif key in {"anyOf", "oneOf", "prefixItems"}:
            result[key] = [
                _normalise_schema_node(item, path=f"{path}.{key}[{index}]")
                for index, item in enumerate(value)
            ]
        else:
            result[key] = copy.deepcopy(value)
    return result

def normalize_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise TypeError("schema must be a JSON object")
    normalized = _normalise_schema_node(schema)
    if not isinstance(normalized, dict):
        raise TypeError("normalized schema must be a JSON object")
    return normalized

def _json_types_match(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    return False

def _schema_error(path: str, message: str) -> GeminiSchemaValidationError:
    return GeminiSchemaValidationError(f"Schema violation at {path}: {message}")

def _json_instances_equal(left: Any, right: Any) -> bool:
    left_is_number = isinstance(left, (int, float)) and not isinstance(left, bool)
    right_is_number = isinstance(right, (int, float)) and not isinstance(right, bool)
    if left_is_number or right_is_number:
        if not (left_is_number and right_is_number):
            return False
        try:
            return Decimal(str(left)) == Decimal(str(right))
        except InvalidOperation:
            return False
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_instances_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_instances_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right

def validate_json_schema_instance(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> None:
    if not isinstance(schema, dict):
        raise TypeError("schema nodes must be JSON objects")

    if "anyOf" in schema:
        failures: list[str] = []
        for candidate in schema["anyOf"]:
            try:
                validate_json_schema_instance(value, candidate, path=path)
                break
            except GeminiSchemaValidationError as exc:
                failures.append(str(exc))
        else:
            raise _schema_error(path, f"does not match anyOf: {failures}")

    if "oneOf" in schema:
        matches = 0
        for candidate in schema["oneOf"]:
            try:
                validate_json_schema_instance(value, candidate, path=path)
            except GeminiSchemaValidationError:
                continue
            matches += 1
        if matches != 1:
            raise _schema_error(
                path,
                f"must match exactly one oneOf branch; got {matches}",
            )

    expected_type = schema.get("type")
    if expected_type is not None:
        accepted_types = (
            list(expected_type)
            if isinstance(expected_type, (list, tuple))
            else [expected_type]
        )
        if not any(_json_types_match(value, str(item)) for item in accepted_types):
            raise _schema_error(
                path,
                f"expected type {expected_type!r}, got {type(value).__name__}",
            )

    if "enum" in schema and not any(
        _json_instances_equal(value, candidate)
        for candidate in schema["enum"]
    ):
        raise _schema_error(path, f"value {value!r} is not in enum {schema['enum']!r}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise _schema_error(path, f"missing required keys {missing!r}")

        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in properties:
                validate_json_schema_instance(item, properties[key], path=item_path)
            elif additional is False:
                raise _schema_error(path, f"unexpected key {key!r}")
            elif isinstance(additional, dict):
                validate_json_schema_instance(item, additional, path=item_path)

        if "minProperties" in schema and len(value) < int(schema["minProperties"]):
            raise _schema_error(
                path,
                f"requires at least {schema['minProperties']} properties",
            )
        if "maxProperties" in schema and len(value) > int(schema["maxProperties"]):
            raise _schema_error(
                path,
                f"allows at most {schema['maxProperties']} properties",
            )

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                validate_json_schema_instance(
                    item,
                    item_schema,
                    path=f"{path}[{index}]",
                )
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise _schema_error(path, f"requires at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise _schema_error(path, f"allows at most {schema['maxItems']} items")
        if schema.get("uniqueItems") is True:
            for index, item in enumerate(value):
                for earlier_index, earlier_item in enumerate(value[:index]):
                    if _json_instances_equal(item, earlier_item):
                        raise _schema_error(
                            path,
                            "requires unique items; indices "
                            f"{earlier_index} and {index} are equal",
                        )

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise _schema_error(
                path,
                f"requires at least {schema['minLength']} characters",
            )
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise _schema_error(
                path,
                f"allows at most {schema['maxLength']} characters",
            )
        if "pattern" in schema:
            try:
                matches = re.search(str(schema["pattern"]), value)
            except re.error as exc:
                raise ValueError(
                    f"Invalid JSON Schema pattern at {path}: {exc}"
                ) from exc
            if matches is None:
                raise _schema_error(
                    path,
                    f"does not match pattern {schema['pattern']!r}",
                )

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        if "minimum" in schema and value < schema["minimum"]:
            raise _schema_error(path, f"value is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise _schema_error(path, f"value is above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise _schema_error(
                path,
                "value must be greater than exclusiveMinimum "
                f"{schema['exclusiveMinimum']}",
            )
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise _schema_error(
                path,
                "value must be less than exclusiveMaximum "
                f"{schema['exclusiveMaximum']}",
            )
        if "multipleOf" in schema:
            divisor = schema["multipleOf"]
            if (
                isinstance(divisor, bool)
                or not isinstance(divisor, (int, float))
                or not math.isfinite(float(divisor))
                or divisor <= 0
            ):
                raise ValueError(
                    f"JSON Schema multipleOf at {path} must be a positive number"
                )
            try:
                remainder = Decimal(str(value)) % Decimal(str(divisor))
            except (InvalidOperation, ZeroDivisionError) as exc:
                raise ValueError(
                    f"Could not evaluate JSON Schema multipleOf at {path}"
                ) from exc
            if remainder != 0:
                raise _schema_error(
                    path,
                    f"value is not a multiple of {divisor}",
                )

def _append_description(schema: dict[str, Any], text: str) -> None:
    existing = str(schema.get("description", "")).strip()
    schema["description"] = f"{existing} {text}".strip()

def _retryable_gemini_error(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        code = int(getattr(exc, "code", 0) or 0)
        return code in {408, 429} or 500 <= code <= 599
    return isinstance(exc, httpx.TransportError)

class GeminiClient:
    provider = "gemini"
    backend = "gemini_developer_api"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 16000,
        thinking_budget: int | None = None,
        supports_seed: bool = True,
        structured_output_protocol: str | None = None,
        structured_output_regeneration_attempts: int = 0,
        structured_output_reformat_attempts: int = 0,
        trajectory_cell_retry_attempts: int = 0,
        scratchpad_limits: dict[str, int] | None = None,
        scratchpad_recovery_targets: dict[str, list[int]] | None = None,
    ) -> None:
        self.model = model
        self.thinking_budget = thinking_budget
        self._max_tokens = max_tokens
        self._seed_supported = supports_seed
        self.structured_output_protocol = structured_output_protocol
        self.structured_output_error_class = GeminiStructuredOutputError
        regeneration_attempts = int(structured_output_regeneration_attempts)
        if regeneration_attempts not in {0, 1}:
            raise ValueError(
                "Gemini structured output permits at most one full-action "
                "regeneration per decision"
            )
        reformat_attempts = int(structured_output_reformat_attempts)
        if reformat_attempts != 0:
            raise ValueError(
                "Gemini scratchpad-length violations are diagnostic and must "
                "not trigger reformat generations"
            )
        if scratchpad_recovery_targets:
            raise ValueError(
                "scratchpad_recovery_targets is incompatible with the "
                "no-regeneration-on-length policy"
            )
        self.structured_output_regeneration_attempts = regeneration_attempts
        self.structured_output_reformat_attempts = reformat_attempts
        self.trajectory_cell_retry_attempts = int(
            trajectory_cell_retry_attempts
        )
        self.scratchpad_limits = dict(scratchpad_limits or {})
        self.scratchpad_recovery_targets = {
            str(key): [int(value) for value in values]
            for key, values in (scratchpad_recovery_targets or {}).items()
        }
        self.structured_output_recovery_protocol = (
            "diagnostic_scratchpad_limits_v1"
        )
        self.structured_output_regeneration_protocol = (
            "single_full_action_regeneration_v2"
        )
        self.trajectory_cell_retry_protocol = "atomic_same_episode_v1"
        self.google_genai_version = importlib.metadata.version("google-genai")
        self._client = genai.Client(api_key=api_key)

        self._usage_lock = threading.Lock()
        self._call_local = threading.local()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0

    @property
    def native_structured_output_enabled(self) -> bool:
        return self.structured_output_protocol == "gemini_native_v1"

    def _prepare_schema(
        self,
        schema: dict[str, Any] | None,
        role: str | None,
    ) -> dict[str, Any] | None:
        if schema is None or not self.native_structured_output_enabled:
            return None
        prepared = normalize_gemini_schema(schema)
        if role == "trust":
            properties = prepared.get("properties", {})
            for field_name in (
                "trust_scores",
                "episode_signals",
                "metrics_weights",
            ):
                field_schema = properties.get(field_name)
                if not isinstance(field_schema, dict):
                    continue
                value_schemas = list(
                    (field_schema.get("properties") or {}).values()
                )
                dynamic_values = field_schema.get("additionalProperties")
                if isinstance(dynamic_values, dict):
                    value_schemas.append(dynamic_values)
                for value_schema in value_schemas:
                    if not isinstance(value_schema, dict):
                        continue
                    value_schema.setdefault("minimum", 0.0)
                    value_schema.setdefault("maximum", 1.0)
        scratchpad = prepared.get("properties", {}).get("scratchpad")
        if not isinstance(scratchpad, dict):
            return prepared

        if role in {"propose", "vote"}:
            limit = self.scratchpad_limits.get(role)
            if limit:
                _append_description(
                    scratchpad,
                    f"Must contain at most {limit} Unicode characters.",
                )
        elif role == "trust":
            per_target = self.scratchpad_limits.get("trust_per_target")
            total = self.scratchpad_limits.get("trust_total")
            if scratchpad.get("type") == "object" and per_target:
                for target_schema in scratchpad.get("properties", {}).values():
                    if isinstance(target_schema, dict):
                        _append_description(
                            target_schema,
                            "Must contain at most "
                            f"{per_target} Unicode characters.",
                        )
            elif scratchpad.get("type") == "string" and total:
                _append_description(
                    scratchpad,
                    f"Must contain at most {total} Unicode characters.",
                )
            if total:
                _append_description(
                    scratchpad,
                    "The complete scratchpad must contain at most "
                    f"{total} Unicode characters.",
                )
        return prepared

    def consume_call_metadata(self) -> dict[str, Any]:
        metadata = getattr(self._call_local, "metadata", None)
        self._call_local.metadata = None
        return dict(metadata or {})

    @property
    def last_call_metadata(self) -> dict[str, Any]:
        return dict(getattr(getattr(self, "_call_local", None), "metadata", None) or {})

    @retry(
        retry=retry_if_exception(_retryable_gemini_error),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
        before_sleep=before_sleep_log(_log, logging.WARNING),
    )
    def _generate_content(self, *, contents: str, config: Any) -> Any:
        return self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        try:
            return response.text or ""
        except (AttributeError, ValueError):
            return ""

    @staticmethod
    def _serialize_parsed(parsed: Any) -> str | None:
        if parsed is None:
            return None
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump()
        if not isinstance(parsed, (dict, list)):
            return None
        return json.dumps(parsed, ensure_ascii=False)

    @staticmethod
    def _scratchpad_metadata(parsed: Any) -> dict[str, Any]:
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump()
        if not isinstance(parsed, dict):
            return {}
        scratchpad = parsed.get("scratchpad")
        if isinstance(scratchpad, str):
            return {
                "scratchpad_length": len(scratchpad),
                "scratchpad_max_target_length": None,
            }
        if isinstance(scratchpad, dict):
            lengths = {
                str(key): len(str(value)) for key, value in scratchpad.items()
            }
            flattened = "\n".join(
                f"{key}: {scratchpad[key]}" for key in scratchpad
            )
            return {
                "scratchpad_length": len(flattened),
                "scratchpad_max_target_length": max(lengths.values(), default=0),
                "scratchpad_target_lengths": lengths,
            }
        return {}

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        schema: dict | None = None,
        seed: int | None = None,
        role: str | None = None,
        recovery_stage: str = "initial",
        recovery_attempt: int = 0,
        mechanical_reformat: bool = False,
        **_: Any,
    ) -> str:
        self._call_local.metadata = None
        effective_thinking_budget = 0 if mechanical_reformat else self.thinking_budget
        thinking_config = None
        if effective_thinking_budget is not None:
            thinking_config = types.ThinkingConfig(
                thinking_budget=effective_thinking_budget
            )

        prepared_schema = self._prepare_schema(schema, role)
        response_mime_type = "application/json" if schema is not None else None
        seed_val = (
            int(seed) % (2**31)
            if self._seed_supported and seed is not None
            else None
        )
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,

            temperature=0.0 if mechanical_reformat else temperature,
            response_mime_type=response_mime_type,
            response_json_schema=prepared_schema,
            thinking_config=thinking_config,
            seed=seed_val,
        )

        _log.debug(
            "→ Gemini API call: model=%s thinking_budget=%s stage=%s attempt=%s",
            self.model,
            effective_thinking_budget,
            recovery_stage,
            recovery_attempt,
        )
        try:
            response = self._generate_content(contents=user, config=config)
        except (errors.APIError, httpx.TransportError) as exc:
            if mechanical_reformat:
                raise GeminiScratchpadRecoveryError(
                    "Gemini scratchpad compression request failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            raise

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        with self._usage_lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += candidate_tokens
            self.reasoning_tokens += thinking_tokens

        raw_completion = self._response_text(response)
        candidate = (
            response.candidates[0]
            if getattr(response, "candidates", None)
            else None
        )
        finish_reason = _enum_value(getattr(candidate, "finish_reason", None))
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = _enum_value(getattr(prompt_feedback, "block_reason", None))

        parsed = getattr(response, "parsed", None)
        serialized = self._serialize_parsed(parsed)
        completion = serialized if serialized is not None else raw_completion
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "finish_reason": finish_reason,
            "block_reason": block_reason,
            "response_length": len(raw_completion),
            "returned_length": len(completion),
            "prompt_tokens": prompt_tokens,
            "candidate_tokens": candidate_tokens,
            "thinking_tokens": thinking_tokens,
            "structured_output": prepared_schema is not None,
            "recovery_stage": recovery_stage,
            "recovery_attempt": recovery_attempt,
            "mechanical_reformat": mechanical_reformat,
            "request_seed": seed_val,
            "raw_completion": raw_completion,
            "incomplete": bool(finish_reason and finish_reason != "STOP"),
            **self._scratchpad_metadata(parsed),
        }
        self._call_local.metadata = metadata

        if block_reason in _SAFETY_BLOCK_REASONS or finish_reason in _SAFETY_FINISH_REASONS:
            raise ContentFilterError(
                "Gemini safety system blocked the prompt or completion "
                f"(model={self.model}, block_reason={block_reason!r}, "
                f"finish_reason={finish_reason!r})"
            )

        if finish_reason and finish_reason != "STOP":
            _log.warning(
                "Unexpected Gemini finish_reason=%r model=%s stage=%s",
                finish_reason,
                self.model,
                recovery_stage,
            )
        return completion

    def _brs_config(
        self, system: str, max_tokens: int, temperature: float,
        seed: int | None, **kwargs: Any,
    ) -> Any:
        thinking = None
        if self.thinking_budget is not None:
            thinking = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens,
            temperature=temperature, thinking_config=thinking,
            seed=int(seed) % (2**31) if self._seed_supported and seed is not None else None,

            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            **kwargs,
        )

    def _brs_metadata(self, role: str) -> dict[str, Any]:
        if not hasattr(self, "_call_local"):
            self._call_local = threading.local()
        metadata: dict[str, Any] = {
            "provider": self.provider, "model": self.model,
            "configured_model": self.model,
            "role": role, "requests": [], "call_id": uuid.uuid4().hex,
        }
        self._call_local.metadata = metadata
        return metadata

    @staticmethod
    def _brs_content_snapshot(content: Any) -> dict[str, Any]:
        if isinstance(content, str):
            return {"role": "user", "parts": [{"text": content}]}
        parts = []
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            public = {}
            if getattr(part, "text", None) is not None:
                public["text"] = part.text
            for key in ("function_call", "function_response"):
                function = getattr(part, key, None)
                if function is not None:
                    public[key] = {
                        field: getattr(function, field) for field in (
                            "name", "id", "args" if key == "function_call" else "response",
                        ) if getattr(function, field, None) is not None
                    }
            if public:
                parts.append(public)
        return {"role": getattr(content, "role", None), "parts": parts}

    @staticmethod
    def _brs_record_failure(metadata: dict[str, Any], exc: Exception) -> None:
        record_request_failure(metadata, exc)

    def _brs_generate_once(
        self, contents: Any, config: Any, metadata: dict[str, Any], phase: str,
    ) -> Any:
        request: dict[str, Any] = {
            "phase": phase, "request_seed": config.seed,
            "inference_settings": {
                key: getattr(config, key) for key in (
                    "max_output_tokens", "temperature", "seed",
                ) if getattr(config, key) is not None
            },
        }
        if config.thinking_config is not None:
            request["inference_settings"]["thinking_config"] = (
                config.thinking_config.model_dump(exclude_none=True)
            )
        request["request_payload"] = {
            "model": self.model,
            "system_prompt": metadata.get("system_prompt", ""),
            "user_prompt": metadata.get("user_prompt", ""),
            "messages": [self._brs_content_snapshot(content) for content in (
                contents if isinstance(contents, list) else [contents]
            )],
            "config": config.model_dump(mode="json", exclude_none=True),
        }
        request["request_payload"] = json.loads(json.dumps(request["request_payload"]))
        metadata["requests"].append(request)
        started = time.monotonic()
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config,
            )
        except Exception as exc:
            self._brs_record_failure(metadata, exc)
            raise
        finally:
            request["latency_seconds"] = time.monotonic() - started
        usage = getattr(response, "usage_metadata", None)
        counts = {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "reasoning_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
        }
        with self._usage_lock:
            self.prompt_tokens += counts["prompt_tokens"]
            self.completion_tokens += counts["completion_tokens"]
            self.reasoning_tokens += counts["reasoning_tokens"]
        request.update(counts)
        request.update(
            response_id=getattr(response, "response_id", None),
            response_model=getattr(response, "model_version", None),
        )
        candidates = getattr(response, "candidates", None)
        request["response_payload"] = {
            "candidates": [{
                "finish_reason": _enum_value(getattr(candidate, "finish_reason", None)),
                "content": self._brs_content_snapshot(getattr(candidate, "content", None)),
            } for candidate in candidates or []],
        }
        request["raw_completion"] = self._response_text(response)
        request["status"] = "completed"
        if phase == "feedback":
            metadata["raw_completion"] = self._response_text(response)
        block = _enum_value(getattr(getattr(response, "prompt_feedback", None), "block_reason", None))
        if block is not None:
            request["block_reason"] = block
            request["response_payload"]["block_reason"] = block
        if not candidates or len(candidates) != 1:
            raise ValueError("BRS response must contain exactly one candidate")
        finish = _enum_value(getattr(candidates[0], "finish_reason", None))
        request["finish_reason"] = finish
        if finish != "STOP" or block in _SAFETY_BLOCK_REASONS:
            raise ValueError(f"BRS response incomplete or blocked: {finish!r}, {block!r}")
        return response

    @staticmethod
    def _brs_function_calls(response: Any) -> list[Any]:
        content = response.candidates[0].content
        return [part.function_call for part in (getattr(content, "parts", None) or [])
                if getattr(part, "function_call", None) is not None]

    def complete_feedback_once(
        self, system: str, user: str, max_tokens: int = 16000,
        temperature: float = 0, schema: dict | None = None,
        seed: int | None = None, role: str = "trust_feedback",
    ) -> str:
        metadata = self._brs_metadata(role)
        metadata.update(system_prompt=system, user_prompt=user)
        metadata["structured_output"] = schema is not None
        try:
            config = self._brs_config(
                system, max_tokens, temperature, seed,
                response_mime_type="application/json" if schema is not None else None,
                response_json_schema=self._prepare_schema(schema, role),
            )
            response = call_with_brs_retries(
                lambda: self._brs_generate_once(user, config, metadata, "feedback"),
                metadata=metadata, phase="feedback",
            )
            if self._brs_function_calls(response):
                raise ValueError("Feedback returned an unexpected function call")
            completion = self._response_text(response)
            if not completion.strip():
                raise ValueError("Feedback response contained no completed text")
            metadata.update(raw_completion=completion, structured_output=schema is not None)
            return completion
        except Exception as exc:
            self._brs_record_failure(metadata, exc)
            raise BRSFeedbackError(f"Feedback request failed: {exc}") from exc

    def trust_tool_roundtrip(
        self, system: str, user: str, scores: dict[str, float],
        max_tokens: int = 16000, temperature: float = 0.7,
        seed: int | None = None,
    ) -> dict[str, float]:
        metadata = self._brs_metadata("trust_tool")
        metadata.update(system_prompt=system, user_prompt=user)
        authoritative = dict(scores)
        name = TOOL_NAME
        metadata.update(tool_name=name, tool_result=authoritative)
        try:
            if not authoritative or any(
                not isinstance(k, str) or isinstance(v, bool)
                or not isinstance(v, (int, float)) or not math.isfinite(v)
                or not 0 <= v <= 1 for k, v in authoritative.items()
            ):
                raise ValueError("BRS tool requires finite counterpart scores in [0,1]")
            declaration = types.FunctionDeclaration(
                name=name,
                description="Return your current trust scores for the other parties.",
            )
            tools = [types.Tool(function_declarations=[declaration])]
            config = self._brs_config(
                system, max_tokens, temperature, seed, tools=tools,
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[name],
                )),
            )
            contents = [types.Content(role="user", parts=[types.Part(text=user)])]
            def validate_call(response: Any) -> Any:
                calls = self._brs_function_calls(response)
                if len(calls) != 1 or calls[0].name != name or calls[0].args not in (None, {}):
                    raise ValueError("Expected exactly one get_counterpart_trust_scores() call without arguments")
                return response

            def correct_call() -> None:
                nonlocal contents
                contents = [*contents, types.Content(role="user", parts=[types.Part(text=(
                    "The previous response did not contain a valid function call. "
                    "Call get_counterpart_trust_scores exactly once with no arguments. "
                    "Do not calculate or return trust scores."
                ))])]

            response = brs_delivery_phase(
                lambda: call_with_brs_retries(
                    lambda: self._brs_generate_once(contents, config, metadata, "tool_call"),
                    metadata=metadata, phase="tool_call",
                ), validate_call, correct_call, metadata=metadata, phase="tool_call",
            )
            call = self._brs_function_calls(response)[0]
            metadata.update(tool_call_id=call.id, tool_arguments={})

            contents = [*contents, response.candidates[0].content, types.Content(
                role="user", parts=[types.Part(function_response=types.FunctionResponse(
                    name=name, id=call.id, response=dict(authoritative),
                ))],
            )]
            ack_config = self._brs_config(
                system, max_tokens, temperature, seed, tools=tools,
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
            )
            def validate_ack(response: Any) -> str:
                if self._brs_function_calls(response):
                    raise ValueError("Unexpected function call in tool acknowledgment")
                return self._response_text(response)

            def correct_ack() -> None:
                nonlocal contents

                contents = [*contents, types.Content(role="user", parts=[types.Part(text=(
                    "The tool result has already been delivered. Acknowledge receipt "
                    "briefly in text. Do not call tools, recalculate or return scores."
                ))])]

            text = brs_delivery_phase(
                lambda: call_with_brs_retries(
                    lambda: self._brs_generate_once(contents, ack_config, metadata, "acknowledgment"),
                    metadata=metadata, phase="acknowledgment",
                ), validate_ack, correct_ack, metadata=metadata, phase="acknowledgment",
            )
            metadata["acknowledgment"] = text
            metadata["acknowledgment_status"] = "text_receipt" if text.strip() else "empty_receipt"
            return dict(authoritative)
        except Exception as exc:
            self._brs_record_failure(metadata, exc)
            raise BRSDeliveryError(f"BRS tool delivery failed: {exc}") from exc

    def usage_summary(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,

            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

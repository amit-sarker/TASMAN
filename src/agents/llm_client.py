from __future__ import annotations

import importlib.metadata
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AzureOpenAI, BadRequestError, InternalServerError, OpenAI, RateLimitError, UnprocessableEntityError
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agents.structured_output import (
    BRSDeliveryError,
    BRSFeedbackError,
    DeepSeekStructuredOutputError,
    KimiStructuredOutputError,
    OpenAIStructuredOutputError,
    brs_delivery_phase,
)
from src.BRS.tool import TOOL_NAME, normalize_empty_tool_arguments
from src.BRS.retry import call_with_brs_retries, record_request_failure

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_ENV = object()
_OPENAI_DYNAMIC_MAP_PROTOCOL = (
    "strict_fixed_json_schema_dynamic_maps_json_object_v1"
)

def _schema_contains_typed_dynamic_map(node: Any) -> bool:
    if isinstance(node, list):
        return any(_schema_contains_typed_dynamic_map(value) for value in node)
    if not isinstance(node, dict):
        return False
    if isinstance(node.get("additionalProperties"), dict):
        return True
    return any(_schema_contains_typed_dynamic_map(value) for value in node.values())

class ContentFilterError(RuntimeError):
    pass

def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if not load_dotenv(repo_root / ".env"):
        load_dotenv(repo_root / ".env.example")

def _getenv(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None

class LLMClient:
    def __init__(
        self,
        api_version: str | None = _ENV,
        reasoning_prefixes: tuple[str, ...] = ("o1", "o3", "o4", "grok", "phi"),
        model_override: str | None = None,
        base_url_override: str | None = None,
        api_key_override: str | None = None,
        thinking_budget: int | None = None,
        reasoning_effort: str | None = None,
        is_reasoning_model: bool | None = None,
        merge_system_into_user: bool | None = None,
        supports_response_format: bool | None = None,
        openai_response_format_protocol: str | None = None,
        supports_temperature: bool = False,
        supports_seed: bool = True,
        require_seed_support: bool = False,
        max_token_parameter: str = "max_completion_tokens",
        thinking_mode: str | None = None,
        thinking_keep: str | None = None,
        structured_output_protocol: str | None = None,
        structured_output_regeneration_attempts: int = 0,
        structured_output_reformat_attempts: int = 0,
        trajectory_cell_retry_attempts: int = 0,
        scratchpad_limits: dict[str, int] | None = None,
        scratchpad_recovery_targets: dict[str, list[int]] | None = None,
    ) -> None:
        _load_env()

        self.reasoning_prefixes = tuple(p.lower() for p in reasoning_prefixes)
        self.model = model_override or _getenv("LLM_MODEL", "AZURE_OPENAI_DEPLOYMENT_NAME")
        self.base_url = base_url_override or _getenv("LLM_BASE_URL", "AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key_override or _getenv("LLM_API_KEY", "AZURE_OPENAI_API_KEY")

        if api_version is _ENV:
            self.api_version = _getenv("LLM_API_VERSION", "AZURE_OPENAI_API_VERSION")
        else:
            self.api_version = api_version or None

        missing = [
            name
            for name, value in {
                "LLM_BASE_URL": self.base_url,
                "LLM_API_KEY": self.api_key,
                "LLM_MODEL": self.model,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

        if self.api_version:
            self.client: OpenAI = AzureOpenAI(
                azure_endpoint=self.base_url,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        else:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

        self.thinking_budget = thinking_budget
        if thinking_mode not in {None, "provider_default", "enabled", "disabled"}:
            raise ValueError(f"Unsupported thinking_mode: {thinking_mode!r}")
        if thinking_keep not in {None, "all"}:
            raise ValueError(f"Unsupported thinking_keep: {thinking_keep!r}")
        if thinking_keep is not None and thinking_mode != "enabled":
            raise ValueError("thinking_keep requires thinking_mode='enabled'")
        self.thinking_mode = thinking_mode
        self.thinking_keep = thinking_keep

        self.reasoning_effort = reasoning_effort

        self._is_reasoning_model_override = is_reasoning_model
        self._merge_system_into_user_override = merge_system_into_user
        self._supports_response_format_override = supports_response_format
        if openai_response_format_protocol not in {
            None,
            _OPENAI_DYNAMIC_MAP_PROTOCOL,
        }:
            raise ValueError(
                "Unsupported OpenAI response-format protocol: "
                f"{openai_response_format_protocol!r}"
            )
        self.openai_response_format_protocol = openai_response_format_protocol
        if max_token_parameter not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError(
                f"Unsupported max_token_parameter: {max_token_parameter!r}"
            )
        self.max_token_parameter = max_token_parameter
        if structured_output_protocol not in {
            None,
            "kimi_text_json_v1",
            "deepseek_text_json_v1",
            "openai_native_v1",
        }:
            raise ValueError(
                "Unsupported generic structured-output protocol: "
                f"{structured_output_protocol!r}"
            )
        regeneration_attempts = int(structured_output_regeneration_attempts)
        if regeneration_attempts not in {0, 1}:
            raise ValueError(
                "Safe structured output permits at most one full-action "
                "regeneration per decision"
            )
        if int(structured_output_reformat_attempts) != 0:
            raise ValueError(
                "Scratchpad-length violations are diagnostic and must not "
                "trigger reformat generations"
            )
        if scratchpad_recovery_targets:
            raise ValueError(
                "scratchpad_recovery_targets is incompatible with the "
                "no-regeneration-on-length policy"
            )
        self.structured_output_protocol = structured_output_protocol
        self.structured_output_regeneration_attempts = regeneration_attempts
        self.structured_output_reformat_attempts = 0
        self.trajectory_cell_retry_attempts = int(trajectory_cell_retry_attempts)
        self.scratchpad_limits = dict(scratchpad_limits or {})
        self.scratchpad_recovery_targets: dict[str, list[int]] = {}
        self.structured_output_recovery_protocol = (
            "diagnostic_scratchpad_limits_v1"
        )
        self.structured_output_regeneration_protocol = (
            "single_full_action_regeneration_v2"
        )
        self.trajectory_cell_retry_protocol = "atomic_same_episode_v1"
        if structured_output_protocol == "kimi_text_json_v1":
            self.provider = "kimi"
            self.backend = "azure_foundry_openai_v1"
        elif structured_output_protocol == "deepseek_text_json_v1":
            self.provider = "deepseek"
            self.backend = "azure_foundry_openai_v1"
        elif structured_output_protocol == "openai_native_v1":
            self.provider = "openai"
            self.backend = (
                "azure_openai" if self.api_version else "openai_compatible"
            )
        else:
            self.provider = "openai_compatible"
            self.backend = "openai_compatible"
        self.openai_sdk_version = importlib.metadata.version("openai")
        if self.provider == "kimi":
            self.structured_output_error_class = KimiStructuredOutputError
        elif self.provider == "deepseek":
            self.structured_output_error_class = DeepSeekStructuredOutputError
        elif self.provider == "openai":
            self.structured_output_error_class = OpenAIStructuredOutputError
        else:
            self.structured_output_error_class = None

        self._temp_for_reasoning = supports_temperature
        self._seed_supported = supports_seed
        self._require_seed_support = require_seed_support

        self._usage_lock = threading.Lock()
        self._call_local = threading.local()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0

        self._json_schema_supported = True

    @property
    def native_structured_output_enabled(self) -> bool:
        return getattr(self, "structured_output_protocol", None) in {
            "kimi_text_json_v1",
            "deepseek_text_json_v1",
            "openai_native_v1",
        }

    def consume_call_metadata(self) -> dict[str, Any]:
        metadata = getattr(self._call_local, "metadata", None)
        self._call_local.metadata = None
        return dict(metadata or {})

    @property
    def last_call_metadata(self) -> dict[str, Any]:
        return dict(getattr(getattr(self, "_call_local", None), "metadata", None) or {})

    def _is_reasoning_model(self) -> bool:
        override = getattr(self, "_is_reasoning_model_override", None)
        if override is not None:
            return bool(override)
        model = (self.model or "").lower()
        return any(model.startswith(prefix) for prefix in self.reasoning_prefixes)

    def _should_merge_system_into_user(self) -> bool:
        override = getattr(self, "_merge_system_into_user_override", None)
        if override is not None:
            return bool(override)
        return self._is_reasoning_model()

    def _supports_schema_response_format(self) -> bool:
        if getattr(self, "structured_output_protocol", None) in {
            "kimi_text_json_v1",
            "deepseek_text_json_v1",
        }:
            return False
        override = getattr(self, "_supports_response_format_override", None)
        if override is not None:
            return bool(override)
        return not self._is_reasoning_model()

    def uses_dynamic_map_json_object(self, schema: dict | None) -> bool:
        return bool(
            schema is not None
            and self.openai_response_format_protocol
            == _OPENAI_DYNAMIC_MAP_PROTOCOL
            and _schema_contains_typed_dynamic_map(schema)
        )

    @staticmethod
    def _is_content_filter_bad_request(exc: BadRequestError) -> bool:
        def is_filter_signal(value: Any) -> bool:
            normalised = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
            return normalised in {

                "invalidprompt",

                "contentfilter",
                "responsibleaipolicyviolation",
            }

        if is_filter_signal(getattr(exc, "code", None)):
            return True

        structured_body = getattr(exc, "body", None)

        def contains_structured_signal(value: Any) -> bool:
            if isinstance(value, dict):
                for key, child in value.items():
                    normalised_key = re.sub(
                        r"[^a-z0-9]", "", str(key).lower()
                    )
                    if normalised_key in {"code", "finishreason"}:
                        if is_filter_signal(child):
                            return True
                    if contains_structured_signal(child):
                        return True
            elif isinstance(value, (list, tuple)):
                return any(contains_structured_signal(child) for child in value)
            return False

        if contains_structured_signal(structured_body):
            return True

        if structured_body is None:
            signal = re.search(
                r"(?i)(?:code|finish[_-]?reason)[\"']?\s*[:=]\s*[\"']?"
                r"(invalid[_-]?prompt|content[_-]?filter|"
                r"responsibleaipolicyviolation)[\"']?",
                str(exc),
            )
            return signal is not None
        return False

    @staticmethod
    def _usage_value(usage: Any, field: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(field)
        return getattr(usage, field, None)

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        details = self._usage_value(usage, "completion_tokens_details")
        with self._usage_lock:
            self.prompt_tokens += int(
                self._usage_value(usage, "prompt_tokens") or 0
            )
            self.completion_tokens += int(
                self._usage_value(usage, "completion_tokens") or 0
            )
            self.reasoning_tokens += int(
                self._usage_value(details, "reasoning_tokens") or 0
            )

    def _create_completion(self, kwargs: dict[str, Any]) -> Any:
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if not self._is_content_filter_bad_request(exc):
                raise
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                self._record_usage(body.get("usage"))
            raise ContentFilterError(
                "Content safety filter blocked the prompt or generated completion "
                f"(model={self.model}): {exc}"
            ) from exc

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
        before_sleep=before_sleep_log(_log, logging.WARNING),
    )
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
        call_local = getattr(self, "_call_local", None)
        if call_local is None:
            call_local = threading.local()
            self._call_local = call_local
        call_local.metadata = None
        kwargs: dict[str, Any] = {"model": self.model}

        reasoning_effort = getattr(self, "reasoning_effort", None)
        thinking_mode = getattr(self, "thinking_mode", None)
        thinking_keep = getattr(self, "thinking_keep", None)
        max_token_parameter = getattr(
            self, "max_token_parameter", "max_completion_tokens"
        )
        is_reasoning_model = self._is_reasoning_model()
        if self._should_merge_system_into_user():
            kwargs["messages"] = [{"role": "user", "content": f"{system}\n\n---\n\n{user}"}]
        else:
            kwargs["messages"] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        kwargs[max_token_parameter] = max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if thinking_mode in {"enabled", "disabled"}:
            thinking: dict[str, Any] = {"type": thinking_mode}
            if thinking_keep is not None:
                thinking["keep"] = thinking_keep
            kwargs["extra_body"] = {"thinking": thinking}

        if is_reasoning_model:
            if self._temp_for_reasoning and reasoning_effort in (None, "none"):
                kwargs["temperature"] = temperature
        elif reasoning_effort is not None:
            if self._temp_for_reasoning and reasoning_effort == "none":
                kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = temperature

        if (
            schema is not None
            and self._supports_schema_response_format()
        ):
            if self._json_schema_supported:
                if self.uses_dynamic_map_json_object(schema):
                    kwargs["response_format"] = {"type": "json_object"}
                else:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "output",
                            "schema": schema,
                            "strict": True,
                        },
                    }
            else:
                kwargs["response_format"] = {"type": "json_object"}

        if self._seed_supported and seed is not None:
            kwargs["seed"] = int(seed) % (2**31)

        _log.debug("→ API call: model=%s  base_url=%s", self.model, self.base_url)
        try:
            response = self._create_completion(kwargs)
        except BadRequestError as exc:
            body = str(exc)
            lowered = body.lower()

            if (
                reasoning_effort is not None
                and "temperature" in kwargs
                and "temperature" in lowered
            ):
                raise

            dropped = False
            if (
                "seed" in kwargs
                and "seed" in lowered
                and not getattr(self, "_require_seed_support", False)
            ):
                kwargs.pop("seed", None)
                self._seed_supported = False
                dropped = True
            if "temperature" in kwargs and "temperature" in lowered:
                kwargs.pop("temperature", None)
                self._temp_for_reasoning = False
                dropped = True
            if dropped:
                _log.warning(
                    "Provider rejected sampling param(s) for model=%s; retrying without. (%s)",
                    self.model, str(exc)[:120],
                )
                response = self._create_completion(kwargs)
            else:
                raise
        except UnprocessableEntityError as exc:
            if (
                reasoning_effort is None
                and self._json_schema_supported
                and schema is not None
                and "response_format" in kwargs
                and "json_schema" in str(exc)
            ):
                self._json_schema_supported = False
                kwargs["response_format"] = {"type": "json_object"}
                response = self._create_completion(kwargs)
            else:
                raise
        usage = getattr(response, "usage", None)
        self._record_usage(usage)

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        normalised_finish_reason = re.sub(
            r"[^a-z0-9]", "", str(finish_reason or "").lower()
        )
        content = choice.message.content
        if isinstance(content, list):
            completion = "".join(
                part.text for part in content if getattr(part, "type", "") == "text"
            )
        else:
            completion = content or ""

        details = self._usage_value(usage, "completion_tokens_details")
        prompt_tokens = int(self._usage_value(usage, "prompt_tokens") or 0)
        candidate_tokens = int(
            self._usage_value(usage, "completion_tokens") or 0
        )
        thinking_tokens = int(
            self._usage_value(details, "reasoning_tokens") or 0
        )
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        reasoning_content_length = (
            len(reasoning_content)
            if isinstance(reasoning_content, str)
            else 0
        )
        accepted_finish_reasons = {"stop", "end_turn"}
        incomplete = bool(
            (finish_reason and finish_reason not in accepted_finish_reasons)
            or not completion.strip()
        )
        if self.native_structured_output_enabled:
            self._call_local.metadata = {
                "provider": getattr(self, "provider", "kimi"),
                "backend": getattr(
                    self, "backend", "azure_foundry_openai_v1"
                ),
                "model": getattr(response, "model", None) or self.model,
                "finish_reason": finish_reason,
                "response_length": len(completion),
                "returned_length": len(completion),
                "prompt_tokens": prompt_tokens,
                "candidate_tokens": candidate_tokens,
                "thinking_tokens": thinking_tokens,
                "reasoning_content_present": reasoning_content_length > 0,
                "reasoning_content_length": reasoning_content_length,
                "structured_output": True,
                "recovery_stage": recovery_stage,
                "recovery_attempt": recovery_attempt,
                "mechanical_reformat": mechanical_reformat,
                "request_seed": (
                    kwargs.get("seed") if self._seed_supported else None
                ),
                "thinking_mode": thinking_mode,
                "max_token_parameter": max_token_parameter,
                "raw_completion": completion,
                "incomplete": incomplete,
                "system_fingerprint": getattr(
                    response, "system_fingerprint", None
                ),
            }

        if normalised_finish_reason == "contentfilter":
            raise ContentFilterError(
                "Content safety filter blocked the generated completion "
                f"(model={self.model}, finish_reason={finish_reason!r})"
            )
        if finish_reason and finish_reason not in accepted_finish_reasons:
            _log.warning(
                "Unexpected finish_reason=%r  model=%s  (length=token limit hit, "
                "content_filter=safety block)",
                finish_reason, self.model,
            )
        return completion

    def _brs_request_kwargs(
        self, system: str, user: str, max_tokens: int,
        temperature: float, seed: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model}
        if self._should_merge_system_into_user():
            kwargs["messages"] = [
                {"role": "user", "content": f"{system}\n\n---\n\n{user}"}
            ]
        else:
            kwargs["messages"] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        kwargs[getattr(self, "max_token_parameter", "max_completion_tokens")] = max_tokens
        effort = getattr(self, "reasoning_effort", None)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        thinking_mode = getattr(self, "thinking_mode", None)
        if thinking_mode in {"enabled", "disabled"}:
            thinking = {"type": thinking_mode}
            keep = getattr(self, "thinking_keep", None)
            if keep is not None:
                thinking["keep"] = keep
            kwargs["extra_body"] = {"thinking": thinking}
        supports_temperature = getattr(self, "_temp_for_reasoning", False)
        if self._is_reasoning_model():
            if supports_temperature and effort in (None, "none"):
                kwargs["temperature"] = temperature
        elif effort is not None:
            if supports_temperature and effort == "none":
                kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = temperature
        if getattr(self, "_seed_supported", False) and seed is not None:
            kwargs["seed"] = int(seed) % (2**31)
        return kwargs

    def _brs_metadata(self, role: str) -> dict[str, Any]:
        if not hasattr(self, "_call_local"):
            self._call_local = threading.local()
        metadata: dict[str, Any] = {
            "provider": getattr(self, "provider", "openai_compatible"),
            "model": self.model, "configured_model": self.model,
            "role": role, "requests": [], "call_id": uuid.uuid4().hex,
        }
        self._call_local.metadata = metadata
        return metadata

    @staticmethod
    def _brs_message_snapshot(message: Any) -> dict[str, Any]:
        def value(obj: Any, key: str) -> Any:
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        result = {key: value(message, key) for key in (
            "role", "content", "name", "tool_call_id", "refusal",
        ) if value(message, key) is not None}
        calls = value(message, "tool_calls")
        if calls is not None:
            result["tool_calls"] = []
            for call in calls:
                function = value(call, "function")
                result["tool_calls"].append({
                    "id": value(call, "id"), "type": value(call, "type"),
                    "function": {key: value(function, key) for key in ("name", "arguments")},
                })
        return result

    @staticmethod
    def _brs_record_failure(metadata: dict[str, Any], exc: Exception) -> None:
        if isinstance(exc, ContentFilterError):
            record_request_failure(metadata, exc.__cause__ or exc)
            fields = {"error": str(exc), "error_type": "ContentFilterError",
                      "block_reason": "content_filter"}
            metadata.update(fields)
            if metadata.get("requests"):
                metadata["requests"][-1].update(fields)
            if metadata.get("role") == "trust_tool":
                metadata["delivery_blocked"] = True
            return
        record_request_failure(metadata, exc)

    def _brs_create_once(
        self, kwargs: dict[str, Any], metadata: dict[str, Any], phase: str,
    ) -> Any:
        request: dict[str, Any] = {
            "phase": phase, "request_seed": kwargs.get("seed"),
            "inference_settings": {
                key: kwargs[key] for key in (
                    "max_tokens", "max_completion_tokens", "temperature", "seed",
                    "reasoning_effort", "extra_body",
                ) if key in kwargs
            },
        }
        request["request_payload"] = {
            "model": kwargs["model"],
            "system_prompt": metadata.get("system_prompt", ""),
            "user_prompt": metadata.get("user_prompt", ""),
            "messages": [self._brs_message_snapshot(message) for message in kwargs["messages"]],
            **{key: kwargs[key] for key in (
                "tools", "tool_choice", "parallel_tool_calls", "response_format",
                "max_tokens", "max_completion_tokens", "temperature", "seed",
                "reasoning_effort", "extra_body",
            ) if key in kwargs},
        }

        request["request_payload"] = json.loads(json.dumps(request["request_payload"]))
        metadata["requests"].append(request)
        started = time.monotonic()
        try:
            response = self.client.with_options(max_retries=0).chat.completions.create(**kwargs)
        except Exception as exc:
            usage = getattr(exc, "body", None)
            if isinstance(usage, dict):
                error_usage = usage.get("usage")
                self._record_usage(error_usage)
                if error_usage is not None:
                    request.update({
                        "prompt_tokens": int(self._usage_value(error_usage, "prompt_tokens") or 0),
                        "completion_tokens": int(self._usage_value(error_usage, "completion_tokens") or 0),
                        "reasoning_tokens": int(self._usage_value(
                            self._usage_value(error_usage, "completion_tokens_details"), "reasoning_tokens",
                        ) or 0),
                    })
            if isinstance(exc, BadRequestError) and self._is_content_filter_bad_request(exc):
                blocked = ContentFilterError(
                    "Content safety filter blocked the prompt or generated completion "
                    f"(model={self.model}): {exc}"
                )
                blocked.__cause__ = exc
                self._brs_record_failure(metadata, blocked)
                raise blocked from exc
            self._brs_record_failure(metadata, exc)
            raise
        finally:
            request["latency_seconds"] = time.monotonic() - started
        usage = getattr(response, "usage", None)
        self._record_usage(usage)
        request.update({
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "prompt_tokens": int(self._usage_value(usage, "prompt_tokens") or 0),
            "completion_tokens": int(self._usage_value(usage, "completion_tokens") or 0),
            "reasoning_tokens": int(self._usage_value(
                self._usage_value(usage, "completion_tokens_details"), "reasoning_tokens"
            ) or 0),
        })
        choices = getattr(response, "choices", None)
        request["response_payload"] = {
            "choices": [{
                "finish_reason": getattr(choice, "finish_reason", None),
                "message": self._brs_message_snapshot(getattr(choice, "message", None)),
            } for choice in choices or []],
        }
        request["raw_completion"] = ""
        if choices:
            content = getattr(getattr(choices[0], "message", None), "content", None)
            if isinstance(content, str):
                request["raw_completion"] = content
        request["status"] = "completed"
        if phase == "feedback" and choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                metadata["raw_completion"] = content
            refusal = getattr(message, "refusal", None)
            if isinstance(refusal, str):
                metadata["refusal"] = refusal
        if not choices or len(choices) != 1:
            raise ValueError("BRS response must contain exactly one choice")
        request["finish_reason"] = getattr(choices[0], "finish_reason", None)
        return choices[0]

    @staticmethod
    def _brs_text(choice: Any, *, allow_empty: bool = False) -> str:
        message = choice.message
        if getattr(message, "refusal", None):
            raise ValueError("BRS response was refused")
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason != "stop":
            raise ValueError(
                f"BRS response did not finish normally (finish_reason={finish_reason!r})"
            )
        if getattr(message, "tool_calls", None):
            raise ValueError("Unexpected tool call in BRS text response")
        content = getattr(message, "content", None)
        if allow_empty and (content is None or isinstance(content, str) and not content.strip()):
            return ""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("BRS response contained no completed text")
        return content

    def complete_feedback_once(
        self, system: str, user: str, max_tokens: int = 16000,
        temperature: float = 0, schema: dict | None = None,
        seed: int | None = None, role: str = "trust_feedback",
    ) -> str:
        metadata = self._brs_metadata(role)
        metadata.update(system_prompt=system, user_prompt=user)
        try:
            kwargs = self._brs_request_kwargs(system, user, max_tokens, temperature, seed)

            formatted_json = schema is not None and self._supports_schema_response_format()
            metadata["structured_output"] = formatted_json
            if formatted_json:
                if _schema_contains_typed_dynamic_map(schema):
                    raise ValueError("Feedback schema must use a fixed set of target keys")
                if getattr(self, "_json_schema_supported", True):
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "trust_feedback", "strict": True, "schema": schema},
                    }
                else:
                    kwargs["response_format"] = {"type": "json_object"}
                metadata["response_format"] = kwargs["response_format"]["type"]
            choice = call_with_brs_retries(
                lambda: self._brs_create_once(kwargs, metadata, "feedback"),
                metadata=metadata, phase="feedback",
            )
            text = self._brs_text(choice)
            metadata["raw_completion"] = text
            return text
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
        tool_name = TOOL_NAME
        metadata.update(tool_name=tool_name, tool_result=authoritative)
        try:
            if not authoritative or any(
                not isinstance(k, str) or isinstance(v, bool)
                or not isinstance(v, (int, float)) or not math.isfinite(v)
                or not 0 <= v <= 1 for k, v in authoritative.items()
            ):
                raise ValueError("BRS tool requires finite counterpart scores in [0,1]")
            kwargs = self._brs_request_kwargs(system, user, max_tokens, temperature, seed)
            kwargs.update(
                tools=[{"type": "function", "function": {
                    "name": tool_name,
                    "description": "Return your current trust scores for the other parties.",
                    "parameters": {"type": "object", "properties": {},
                                   "required": [], "additionalProperties": False},
                }}],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                parallel_tool_calls=False,
            )
            if getattr(self, "structured_output_protocol", None) == "deepseek_text_json_v1":
                kwargs["tool_choice"] = "auto"
                kwargs.pop("parallel_tool_calls")
            def validate_call(choice: Any) -> Any:
                message = choice.message
                calls = getattr(message, "tool_calls", None) or []
                if (choice.finish_reason != "tool_calls" or len(calls) != 1
                        or getattr(message, "refusal", None)):
                    raise ValueError("Expected exactly one completed native tool call")
                call = calls[0]
                arguments = getattr(getattr(call, "function", None), "arguments", None)
                if (getattr(call, "type", None) != "function"
                        or getattr(getattr(call, "function", None), "name", None) != tool_name
                        or not isinstance(getattr(call, "id", None), str) or not call.id.strip()):
                    raise ValueError("Expected get_counterpart_trust_scores() with no arguments")
                normalized, reason = normalize_empty_tool_arguments(arguments)
                if reason is not None:
                    metadata.setdefault("tool_argument_repairs", []).append({
                        "phase": "tool_call", "request_index": len(metadata["requests"]) - 1,
                        "tool_call_id": call.id, "raw_arguments": arguments,
                        "normalized_arguments": normalized, "reason": reason,
                    })
                return choice

            def correct_call() -> None:
                kwargs["messages"] = [*kwargs["messages"], {
                    "role": "user", "content": (
                        "The previous response did not contain a valid function call. "
                        "Call get_counterpart_trust_scores exactly once with no arguments. "
                        "Set the function arguments to exactly {}: one empty JSON object, "
                        "without additional objects, text, or Markdown. "
                        "Do not calculate or return trust scores."
                    ),
                }]

            choice = brs_delivery_phase(
                lambda: call_with_brs_retries(
                    lambda: self._brs_create_once(kwargs, metadata, "tool_call"),
                    metadata=metadata, phase="tool_call",
                ), validate_call, correct_call, metadata=metadata, phase="tool_call",
            )
            message = choice.message
            call = message.tool_calls[0]
            metadata.update(tool_call_id=call.id, tool_arguments={})
            if hasattr(message, "model_dump"):
                assistant_message = deepcopy(message.model_dump(exclude_none=True))
            else:
                assistant_message = {
                    "role": "assistant", "content": getattr(message, "content", None),
                    "tool_calls": [{"id": call.id, "type": "function", "function": {
                        "name": tool_name, "arguments": call.function.arguments,
                    }}],
                }

                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content is not None:
                    assistant_message["reasoning_content"] = reasoning_content

            assistant_message["tool_calls"][0]["function"]["arguments"] = (
                normalize_empty_tool_arguments(call.function.arguments)[0]
            )

            kwargs["messages"] = [*kwargs["messages"], assistant_message, {
                "role": "tool", "tool_call_id": call.id,
                "content": json.dumps(authoritative, allow_nan=False, sort_keys=True),
            }]
            kwargs["tool_choice"] = "none"
            def correct_ack() -> None:
                kwargs["messages"] = [*kwargs["messages"], {
                    "role": "user", "content": (
                        "The tool result has already been delivered. Acknowledge receipt "
                        "briefly in text. Do not call tools, recalculate or return scores."
                    ),
                }]

            acknowledgment = brs_delivery_phase(
                lambda: call_with_brs_retries(
                    lambda: self._brs_create_once(kwargs, metadata, "acknowledgment"),
                    metadata=metadata, phase="acknowledgment",
                ), lambda ack: self._brs_text(ack, allow_empty=True), correct_ack,
                metadata=metadata, phase="acknowledgment",
            )
            metadata["acknowledgment"] = acknowledgment
            metadata["acknowledgment_status"] = "text_receipt" if acknowledgment else "empty_receipt"
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

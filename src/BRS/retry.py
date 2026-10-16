from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

_log = logging.getLogger(__name__)
_T = TypeVar("_T")
MAX_ATTEMPTS = 8
_TRANSPORT_ERRORS = {
    "APITimeoutError", "APIConnectionError", "TimeoutError", "ReadTimeout",
    "ConnectTimeout", "WriteTimeout", "PoolTimeout", "ConnectError",
    "ReadError", "WriteError", "RemoteProtocolError",
    "TransportError", "TimeoutException", "NetworkError", "ProtocolError",
    "CloseError", "LocalProtocolError", "ProxyError", "UnsupportedProtocol",
}
_PROVIDER_ERRORS = {
    "APIError", "APIStatusError", "RateLimitError", "InternalServerError",
    "ClientError", "ServerError", "HttpError", "HTTPStatusError",
    "BadRequestError", "UnprocessableEntityError",
}
_TRANSIENT_CODES = {
    "toomanyrequests", "ratelimiterror", "ratelimitexceeded", "ratelimited",
    "resourceexhausted", "internalservererror", "serviceunavailable",
    "servererror", "requesttimeout", "408", "429", "500", "502", "503", "504",
}
_ERROR_CODE = re.compile(r'''["']code["']\s*:\s*(?:["']([^"']+)["']|(\d{3})\b)''', re.I)
_SAFE_HEADERS = {
    "retry-after", "retry-after-ms", "request-id", "x-request-id",
    "x-ms-request-id", "x-ms-client-request-id",
    "ms-azureml-model-error-statuscode", "ms-azureml-model-error-reason",
}
_FAILURE_FIELDS = ("error", "error_type", "status_code", "response_headers", "error_codes")

def _status(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if 100 <= result <= 599 else None

def _error_codes(value: Any, depth: int = 0) -> set[str]:
    if depth > 10:
        return set()
    codes: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() == "code" and isinstance(child, (str, int)):
                codes.add(str(child))

            elif str(key).lower() in {"error", "errors", "message", "detail", "details"}:
                codes.update(_error_codes(child, depth + 1))
    elif isinstance(value, (list, tuple)):
        for child in value:
            codes.update(_error_codes(child, depth + 1))
    elif isinstance(value, str):
        codes.update(text or number for text, number in _ERROR_CODE.findall(value))
    return codes

def is_retryable_failure_record(record: Mapping[str, Any]) -> bool:
    if not isinstance(record, Mapping):
        return False
    error_type = record.get("error_type")
    if not isinstance(error_type, str):
        return False
    if error_type in _TRANSPORT_ERRORS:
        return True
    if error_type in {"RateLimitError", "InternalServerError"}:
        return True
    if error_type not in _PROVIDER_ERRORS:
        return False
    status = _status(record.get("status_code"))
    if status in {408, 429} or (status is not None and status >= 500):
        return True
    if status in {401, 403, 404, 422}:
        return False
    headers = record.get("response_headers") or {}
    if not isinstance(headers, Mapping):
        headers = {}
    upstream_status = _status(headers.get("ms-azureml-model-error-statuscode"))
    if upstream_status in {408, 429} or (upstream_status is not None and upstream_status >= 500):
        return True
    codes = record.get("error_codes") or []
    if not isinstance(codes, (list, tuple, set)):
        codes = []
    codes = set(str(code) for code in codes) | _error_codes(record.get("error", ""))
    return any(re.sub(r"[^a-z0-9]", "", code.lower()) in _TRANSIENT_CODES for code in codes)

def _failure_details(exc: Exception) -> dict[str, Any]:
    details: dict[str, Any] = {
        "status": "failed", "error": str(exc), "error_type": type(exc).__name__,
    }
    response = getattr(exc, "response", None)
    status = (_status(getattr(exc, "status_code", None))
              or _status(getattr(exc, "code", None))
              or _status(getattr(response, "status_code", None)))
    if status is not None:
        details["status_code"] = status
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if hasattr(headers, "items"):
        selected = {
            str(key).lower(): str(value) for key, value in headers.items()
            if str(key).lower() in _SAFE_HEADERS or str(key).lower().startswith("x-ratelimit-")
        }
        if selected:
            details["response_headers"] = selected
    codes = _error_codes(getattr(exc, "body", None))
    codes.update(_error_codes(getattr(exc, "response_json", None)))
    codes.update(_error_codes(str(exc)))
    if codes:
        details["error_codes"] = sorted(codes)
    return details

def is_retryable_brs_error(exc: BaseException) -> bool:
    return isinstance(exc, Exception) and is_retryable_failure_record(_failure_details(exc))

def record_request_failure(metadata: dict[str, Any], exc: Exception) -> None:
    details = _failure_details(exc)
    for key in _FAILURE_FIELDS:
        metadata.pop(key, None)
    metadata.update(details)
    if metadata.get("requests"):
        metadata["requests"][-1].update(details)

def _before_sleep(retry_state: Any) -> None:
    metadata = retry_state.kwargs["metadata"]
    phase = retry_state.kwargs["phase"]
    delay = float(retry_state.next_action.sleep)
    if metadata.get("requests"):
        metadata["requests"][-1]["retry_delay_seconds"] = delay
    _log.warning("BRS %s transient request failure; retrying after %.0fs (attempt %d/8)",
                 phase, delay, retry_state.attempt_number)

@retry(
    retry=retry_if_exception(is_retryable_brs_error),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    reraise=True,
    before_sleep=_before_sleep,
)
def call_with_brs_retries(
    operation: Callable[[], _T], *, metadata: dict[str, Any], phase: str,
) -> _T:
    result = operation()

    for key in _FAILURE_FIELDS:
        metadata.pop(key, None)
    metadata["status"] = "completed"
    return result

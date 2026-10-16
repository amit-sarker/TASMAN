from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

from .brs import normalize_counts, serialize_observer_state, scores_from_observer_state, validate_agent_ids
from .models import AgentFeedback, FeedbackPair
from .recovery import (output_correction_error, parse_saved_blank_scratchpad_feedback,
                       parse_saved_feedback, saved_feedback_is_blocked, transport_retry_error)
from .tool import counterpart_scores
from .response_policy import build_feedback_correction_request

_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
_PHASES = {"prepared", "collecting", "feedback_ready", "staged", "delivering", "delivered", "committed"}
_OBSERVER_PHASES = {"prepared", "requested", "validated", "failed", "fallback_no_update"}
_RESOLVED_OBSERVER_PHASES = {"validated", "fallback_no_update"}

def _valid_correction_request(original: dict, corrected: dict, attempt_idx: int) -> bool:
    return any(corrected == build_feedback_correction_request(
        original, native_structured=native, attempt_idx=attempt_idx,
    ) for native in (False, True))

class StateStoreError(RuntimeError):
    pass

class BRSStateStore:
    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir) / "brs_state"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @contextmanager
    def run_lock(self):
        import fcntl
        with (self.root / ".run.lock").open("a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StateStoreError("Another BRS process is using this experiment directory") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def key(trajectory_idx: int, episode_idx: int) -> str:
        if trajectory_idx < 1 or episode_idx < 1:
            raise StateStoreError("Trajectory and episode indices must be positive")
        return f"t{trajectory_idx:03d}-e{episode_idx:03d}"

    def _path(self, key: str) -> Path:
        import re
        if not re.fullmatch(r"t[0-9]+-e[0-9]+", key):
            raise StateStoreError("Invalid BRS journal key")
        return self.root / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            path = self._path(key)
            if not path.exists():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("schema_version") != 3 or value.get("key") != key:
                    raise ValueError("Journal identity/schema mismatch")
                observers = value["observers"]
                if value["phase"] not in _PHASES or not isinstance(observers, dict) or len(observers) < 2:
                    raise ValueError("Invalid observer journal structure")
                if any(not isinstance(row, dict) or row.get("phase") not in _OBSERVER_PHASES for row in observers.values()):
                    raise ValueError("Invalid observer feedback phase")
                self._validate_history(value.get("public_history"), value.get("display_names"), observers)
                return value
            except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
                raise StateStoreError(f"Cannot safely read BRS journal {key}: {exc}") from exc

    def _write(self, value: dict[str, Any]) -> None:
        self._atomic_write(self._path(value["key"]), value)

    def _atomic_write(self, path: Path, value: dict[str, Any]) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                             prefix=f".{path.stem}-", delete=False) as handle:
                temporary = handle.name
                json.dump(value, handle, allow_nan=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _usage_segments(self) -> dict:
        path = self.root / "usage.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected a segment mapping")
            return value
        except (OSError, ValueError) as exc:
            raise StateStoreError("Cannot safely recover BRS request usage") from exc

    @staticmethod
    def _validated_usage(values: Any) -> dict[str, int]:
        if not isinstance(values, Mapping) or (values and set(values) != set(_USAGE_KEYS)):
            raise StateStoreError("Invalid persisted BRS usage counter")
        result = {}
        for key in _USAGE_KEYS:
            count = values.get(key, 0)
            if type(count) is not int or count < 0:
                raise StateStoreError("Invalid persisted BRS usage counter")
            result[key] = count
        return result

    def save_usage(self, segment: str, negotiator: dict) -> None:
        with self._lock:
            segments = self._usage_segments()
            if not isinstance(segment, str) or not segment:
                raise StateStoreError("Usage segment must be a nonempty string")
            previous = self._validated_usage(segments.get(segment, {}))
            current = self._validated_usage(negotiator)

            segments[segment] = {key: max(current[key], previous[key]) for key in _USAGE_KEYS}
            self._atomic_write(self.root / "usage.json", segments)

    def prior_usage(self, excluding_segment: str) -> dict[str, int]:
        with self._lock:
            total = dict.fromkeys(_USAGE_KEYS, 0)
            for segment, usage in self._usage_segments().items():
                if segment == excluding_segment:
                    continue
                values = self._validated_usage(usage)
                for key in total:
                    count = values[key]
                    total[key] += count
            return total

    @staticmethod
    def _validate_history(public_history: Any, display_names: Any, observer_ids: Iterable[str]) -> None:
        if not isinstance(public_history, str):
            raise ValueError("BRS public history must be saved formatted text")
        if (not isinstance(display_names, dict) or set(display_names) != set(observer_ids)
                or any(not isinstance(name, str) or not name.strip() for name in display_names.values())):
            raise ValueError("Display names must cover exactly the episode participants")

    def prepare(self, key: str, *, record: dict, public_history: str, display_names: dict[str, str],
                evidence_before: dict, observer_ids: Iterable[str]) -> dict:
        with self._lock:
            try:
                ids = validate_agent_ids(observer_ids)
                before = serialize_observer_state(evidence_before)
                if set(ids) != set(before):
                    raise ValueError("Evidence must contain exactly the observer IDs")
                self._validate_history(public_history, display_names, ids)
            except (ValueError, TypeError) as exc:
                raise StateStoreError(f"Invalid BRS episode preparation: {exc}") from exc
            existing = self.load(key)
            if existing is not None:
                if (existing["record"] != record or existing["public_history"] != public_history
                        or existing["display_names"] != display_names
                        or existing["evidence_before"] != before or set(existing["observers"]) != set(ids)):
                    raise StateStoreError("Cannot replace an existing completed episode")
                return existing
            value = {"schema_version": 3, "key": key, "phase": "prepared",
                     "record": record, "public_history": public_history, "display_names": dict(display_names),
                     "evidence_before": before,
                     "observers": {observer: {"phase": "prepared", "request": None,
                                                "scratchpad": None, "counts": None,
                                                "feedback": None, "metadata": {}}
                                   for observer in ids},
                     "evidence_after": None, "scores": None, "delivered": [],
                     "delivery_metadata": {}}
            self._write(value)
            return value

    def _transition(self, key: str, expected: set[str], phase: str, **fields) -> dict:
        with self._lock:
            value = self.load(key)
            if value is None or value["phase"] not in expected:
                raise StateStoreError(f"Unsafe BRS transition to {phase}: {key}")
            value.update(fields)
            value["phase"] = phase
            self._write(value)
            return value

    def _observer_transition(self, key: str, observer_id: str, expected: str, phase: str, **fields) -> dict:
        with self._lock:
            value = self.load(key)
            if value is None or value["phase"] not in {"prepared", "collecting"}:
                raise StateStoreError("Observer feedback requires an un-staged episode")
            observer = value["observers"].get(observer_id)
            if observer is None or observer["phase"] != expected:
                raise StateStoreError(f"Unsafe BRS feedback transition for {observer_id}: {expected} to {phase}")
            observer.update(fields)
            observer["phase"] = phase
            value["phase"] = (
                "feedback_ready" if all(row["phase"] in _RESOLVED_OBSERVER_PHASES
                                        for row in value["observers"].values())
                else "collecting"
            )
            self._write(value)
            return value

    def mark_requested(self, key: str, observer_id: str, request: dict) -> dict:
        if not isinstance(request, dict):
            raise StateStoreError("Feedback dispatch requires the exact request")
        return self._observer_transition(key, observer_id, "prepared", "requested",
                                         request=request, feedback_request_attempt=1)

    def validate_feedback_retry(self, key: str, observer_id: str) -> dict:
        with self._lock:
            value = self.load(key)
            if (value is None or value["phase"] not in {"prepared", "collecting"}
                    or value.get("evidence_after") is not None or value.get("scores") is not None
                    or value.get("delivered") or value.get("delivery_metadata")):
                raise StateStoreError("Feedback retry requires an un-staged episode")
            observer = value["observers"].get(observer_id)
            error = transport_retry_error(observer)
            if error is not None:
                raise StateStoreError(f"Cannot retry BRS feedback for {observer_id}: {error}")
            prior = observer.get("prior_attempts", [])
            attempt = observer.get("feedback_request_attempt", 1)
            if (not isinstance(prior, list) or any(not isinstance(row, dict) for row in prior)
                    or type(attempt) is not int or attempt != len(prior) + 1):
                raise StateStoreError("Invalid BRS feedback retry attempt history")
            for number, archived in enumerate(prior, 1):
                if ("prior_attempts" in archived
                        or type(archived.get("feedback_request_attempt")) is not int
                        or archived["feedback_request_attempt"] != number
                        or (transport_retry_error(archived) is not None
                            and output_correction_error(archived) is not None)):
                    raise StateStoreError("Invalid BRS feedback retry attempt history")
                following = prior[number] if number < len(prior) else observer
                if archived.get("request") != following.get("request"):
                    if (following.get("retry_reason") != "feedback_output_correction"
                            or output_correction_error(archived) is not None
                            or following.get("feedback_correction_attempt") != 1
                            or not _valid_correction_request(archived["request"], following["request"],
                                                             value["record"].get("attempt_idx", 1))):
                        raise StateStoreError("Invalid BRS feedback retry attempt history: changed request")
            return value

    def retry_feedback(self, key: str, observer_id: str) -> dict:
        with self._lock:
            value = self.validate_feedback_retry(key, observer_id)
            observer = value["observers"][observer_id]
            prior = observer.get("prior_attempts", [])
            attempt = observer.get("feedback_request_attempt", 1)
            previous = deepcopy({field: item for field, item in observer.items() if field != "prior_attempts"})
            previous["feedback_request_attempt"] = attempt
            observer.update(
                phase="requested", metadata={}, prior_attempts=[*prior, previous],
                feedback_request_attempt=attempt + 1, retry_reason="explicit_transport_retry",
            )
            observer.pop("error", None)
            value["phase"] = "collecting"
            self._write(value)
            return value

    def retry_feedback_output(self, key: str, observer_id: str, request: dict) -> dict:
        with self._lock:
            value = self.load(key)
            if (value is None or value["phase"] not in {"prepared", "collecting"}
                    or value.get("evidence_after") is not None or value.get("scores") is not None
                    or value.get("delivered") or value.get("delivery_metadata")):
                raise StateStoreError("Feedback output correction requires an un-staged episode")
            observer = value["observers"].get(observer_id)
            reason = output_correction_error(observer)
            if reason is not None:
                raise StateStoreError(f"Cannot correct BRS feedback for {observer_id}: {reason}")
            original = observer["request"]
            if (not isinstance(request, dict)
                    or not _valid_correction_request(original, request, value["record"].get("attempt_idx", 1))):
                raise StateStoreError("Correction must use the saved request and standard response policy")
            prior = observer.get("prior_attempts", [])
            attempt = observer.get("feedback_request_attempt", 1)
            if (not isinstance(prior, list) or type(attempt) is not int
                    or attempt != len(prior) + 1):
                raise StateStoreError("Invalid BRS feedback correction attempt history")
            archived = deepcopy({field: item for field, item in observer.items()
                                 if field != "prior_attempts"})
            archived["feedback_request_attempt"] = attempt
            observer.update(phase="requested", request=deepcopy(request), metadata={},
                            prior_attempts=[*prior, archived],
                            feedback_request_attempt=attempt + 1,
                            feedback_correction_attempt=1,
                            retry_reason="feedback_output_correction")
            observer.pop("error", None)
            value["phase"] = "collecting"
            self._write(value)
            return value

    def fallback_feedback(self, key: str, observer_id: str) -> dict:
        with self._lock:
            value = self.load(key)
            if value is None or observer_id not in value["observers"]:
                raise StateStoreError("Unknown feedback observer")
            observer = value["observers"][observer_id]
            candidate = {**observer, "feedback_correction_attempt": 0}
            blocked = saved_feedback_is_blocked(observer)
            if (not blocked and (observer.get("feedback_correction_attempt") != 1
                    or output_correction_error(candidate) is not None)):
                raise StateStoreError("Fallback requires an invalid received corrective answer")
            reason = ("content_filter" if blocked else
                      str(observer.get("error") or "Invalid corrective feedback response"))
            return self._observer_transition(
                key, observer_id, "failed", "fallback_no_update",
                fallback_reason=reason, update_skipped=True,
                counts=None, feedback=None, scratchpad=None,
            )

    def recover_blank_scratchpad_feedback(self, key: str, observer_id: str) -> dict:
        return self.recover_saved_feedback(key, observer_id, blank_only=True)

    def recover_saved_feedback(self, key: str, observer_id: str, *, blank_only: bool = False) -> dict:
        with self._lock:
            value = self.load(key)
            if value is None or observer_id not in value["observers"]:
                raise StateStoreError("Unknown feedback observer")
            observer = value["observers"][observer_id]
            targets = [target for target in value["observers"] if target != observer_id]
            parser = parse_saved_blank_scratchpad_feedback if blank_only else parse_saved_feedback
            assessment = parser(observer, targets)
            if assessment is None:
                return value
            if (value["phase"] not in {"prepared", "collecting"}
                    or value.get("evidence_after") is not None or value.get("scores") is not None
                    or value.get("delivered") or value.get("delivery_metadata")):
                raise StateStoreError("Local feedback validation recovery requires an un-staged episode")
            original = deepcopy(observer)
            empty_targets = sorted(target for target, summary in assessment.scratchpad.items()
                                   if not summary.strip())
            recovery = {"reason": ("blank_scratchpad_is_diagnostic" if empty_targets
                                   else "saved_feedback_local_revalidation"),
                        "source": "saved_provider_response", "new_feedback_requests": 0,
                        "empty_scratchpad_targets": empty_targets}
            metadata = deepcopy(observer["metadata"])
            metadata.update(empty_scratchpad_targets=empty_targets,
                            local_validation_recovery=recovery)
            feedback = normalize_counts(assessment.feedback)
            observer.update(
                phase="validated", scratchpad=dict(assessment.scratchpad),
                counts={target: pair.model_dump(mode="json")
                        for target, pair in assessment.feedback.items()},
                feedback={target: pair.model_dump(mode="json") for target, pair in feedback.items()},
                metadata=metadata,
                local_validation_recovery={**recovery, "original_failure": original},
            )
            observer.pop("error", None)
            value["phase"] = (
                "feedback_ready" if all(row["phase"] in _RESOLVED_OBSERVER_PHASES
                                        for row in value["observers"].values())
                else "collecting"
            )
            self._write(value)
            return value

    def save_feedback(self, key: str, observer_id: str, feedback: dict, counts: dict,
                      *, scratchpad: dict[str, str],
                      metadata: dict | None = None, usage_snapshot: dict | None = None) -> dict:
        with self._lock:
            value = self.load(key)
            if value is None or observer_id not in value["observers"]:
                raise StateStoreError("Unknown feedback observer")
            try:
                targets = set(value["observers"]) - {observer_id}
                if not isinstance(counts, Mapping) or set(counts) != targets:
                    raise ValueError("Counts must cover exactly the observer's counterparts")
                if not isinstance(feedback, Mapping) or set(feedback) != targets:
                    raise ValueError("Feedback must cover exactly the observer's counterparts")
                assessment = AgentFeedback.model_validate({
                    "scratchpad": scratchpad, "feedback": counts,
                })
                normalized = normalize_counts(counts)
                validated = {target: FeedbackPair.model_validate(feedback[target]) for target in targets}
                if validated != normalized:
                    raise ValueError("Feedback must equal the code-normalized counts")
                saved_counts = {target: pair.model_dump(mode="json") for target, pair in assessment.feedback.items()}
                saved_feedback = {target: pair.model_dump(mode="json") for target, pair in normalized.items()}
                if usage_snapshot is not None:
                    if not isinstance(usage_snapshot, dict) or not isinstance(usage_snapshot.get("segment"), str) or not usage_snapshot["segment"]:
                        raise ValueError("Usage snapshot requires a process segment")
                    self._validated_usage(usage_snapshot["negotiator"])
            except (ValueError, TypeError, KeyError) as exc:
                raise StateStoreError(f"Invalid observer feedback: {exc}") from exc
            return self._observer_transition(key, observer_id, "requested", "validated",
                                             scratchpad=dict(assessment.scratchpad),
                                             feedback=saved_feedback, counts=saved_counts,
                                             metadata=metadata or {}, usage_snapshot=usage_snapshot)

    def recover_usage(self) -> None:
        with self._lock:
            for path in self.root.glob("t*-e*.json"):
                value = self.load(path.stem)
                for observer in value["observers"].values():
                    snapshot = observer.get("usage_snapshot")
                    if snapshot:
                        self.save_usage(snapshot["segment"], snapshot["negotiator"])

    def stage(self, key: str, *, evidence_after: dict, scores: dict) -> dict:
        with self._lock:
            value = self.load(key)
            try:
                after = serialize_observer_state(evidence_after)
                if value is None or set(after) != set(value["observers"]):
                    raise ValueError("Staged evidence must match the episode observers")
                expected = scores_from_observer_state(after)
                if not isinstance(scores, Mapping) or set(scores) != set(expected):
                    raise ValueError("Staged scores must match the episode observers")
                actual = {observer: counterpart_scores(scores, observer) for observer in expected}
                if actual != expected:
                    raise ValueError("Staged scores must equal the BRS evidence scores")
            except (ValueError, TypeError) as exc:
                raise StateStoreError(f"Invalid staged BRS evidence: {exc}") from exc
            return self._transition(key, {"feedback_ready"}, "staged", evidence_after=after, scores=actual)

    def mark_delivered(self, key: str, agent_id: str, metadata: dict | None = None) -> dict:
        with self._lock:
            value = self.load(key)
            if value is None or value["phase"] not in {"staged", "delivering", "delivered"}:
                raise StateStoreError("Delivery requires saved, staged evidence")
            if agent_id not in value["scores"]:
                raise StateStoreError("Unknown delivery recipient")
            delivered = sorted(set(value["delivered"]) | {agent_id})
            delivery_metadata = dict(value.get("delivery_metadata", {}))
            if agent_id not in delivery_metadata:
                delivery_metadata[agent_id] = metadata or {}
            phase = "delivered" if set(delivered) == set(value["scores"]) else "delivering"
            return self._transition(key, {value["phase"]}, phase, delivered=delivered,
                                    delivery_metadata=delivery_metadata)

    def mark_committed(self, key: str) -> dict:
        return self._transition(key, {"delivered", "committed"}, "committed")

    def fail_feedback(self, key: str, observer_id: str, error: str, metadata: dict | None = None) -> dict:
        return self._observer_transition(key, observer_id, "requested", "failed", error=error,
                                         metadata=metadata or {})

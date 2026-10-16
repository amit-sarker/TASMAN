"""Build a standalone observer-input/feedback-output report without making model calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .models import AccumulatedEvidence, FeedbackPair, ObservationCounts

TEMPLATE_PATH = Path(__file__).with_name("feedback_report.html")
DATA_TOKEN = "__FEEDBACK_REPORT_DATA__"
_KEY = re.compile(r"t([0-9]+)-e([0-9]+)")
_REQUEST_FIELDS = ("system", "user", "max_tokens", "temperature", "schema", "seed", "role")
_CALL_FIELDS = (
    "phase", "request_seed", "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "total_tokens", "finish_reason", "status", "error", "model", "provider",
    "configured_model", "response_id", "response_model", "system_fingerprint",
    "latency_seconds", "error_type", "refusal", "block_reason",
    "feedback_request_attempt", "retry_reason",
    "feedback_correction_attempt", "response_recovery_policy", "validation_error",
    "feedback_output_invalid", "output_attempt", "output_recovery_exhausted",
    "delivery_phase", "delivery_blocked", "acknowledgment_status",
)

def _json_file(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    except (OSError, ValueError):
        warnings.append(f"Could not read a JSON object from {path.name}.")
        return None

def _feedback_rows(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("expected object")
                    rows.append(value)
                except ValueError:
                    warnings.append(f"Ignored malformed {path.name} line {number}.")
    except OSError:
        warnings.append(f"Could not read {path.name}.")
    return rows

def _cell(key: Any) -> tuple[int, int] | None:
    matched = _KEY.fullmatch(key) if isinstance(key, str) else None
    if matched and int(matched[1]) > 0 and int(matched[2]) > 0:
        return int(matched[1]), int(matched[2])
    return None

def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None

def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False

def _scores(value: Any, agent_ids: set[str] | None = None) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    if agent_ids is not None and set(value) != agent_ids:
        return None
    if any(not isinstance(agent, str) or not _number(score) or not 0 <= score <= 1
           for agent, score in value.items()):
        return None
    return dict(value)

def _validated_pairs(value: Any, model: Any, agent_ids: set[str] | None = None) -> dict | None:
    if not isinstance(value, dict) or not value:
        return None
    if any(not isinstance(agent, str) or not agent.strip() for agent in value):
        return None
    if agent_ids is not None and set(value) != agent_ids:
        return None
    try:
        return {agent: model.model_validate(pair).model_dump(mode="json") for agent, pair in value.items()}
    except (ValueError, TypeError):
        return None

def _scratchpad(value: Any, agent_ids: set[str] | None = None) -> dict[str, str] | None:
    if not isinstance(value, dict) or not value:
        return None
    if agent_ids is not None and set(value) != agent_ids:
        return None
    if any(not isinstance(agent, str) or not agent.strip()
           or not isinstance(summary, str)
           for agent, summary in value.items()):
        return None
    return dict(value)

def _metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        key: value[key] for key in _CALL_FIELDS
        if key in value and (isinstance(value[key], (str, bool)) or _number(value[key]))
    }
    requests = value.get("requests")
    if isinstance(requests, list):
        result["requests"] = [_metadata(row) for row in requests if isinstance(row, dict)]
    repairs = value.get("local_parse_repairs")
    if isinstance(repairs, list) and all(isinstance(item, str) for item in repairs):
        result["local_parse_repairs"] = list(repairs)
    corrections = value.get("output_recovery")
    if isinstance(corrections, list):
        result["output_recovery"] = [_metadata(row) for row in corrections if isinstance(row, dict)]
    settings = value.get("inference_settings")
    if isinstance(settings, dict):
        result["inference_settings"] = {
            key: settings[key] for key in (
                "max_tokens", "max_completion_tokens", "max_output_tokens", "temperature",
                "seed", "reasoning_effort",
            ) if key in settings and (isinstance(settings[key], str) or _number(settings[key]))
        }
        thinking = settings.get("thinking_config")
        if isinstance(thinking, dict):
            result["inference_settings"]["thinking_config"] = {
                key: thinking[key] for key in ("thinking_budget", "thinking_level", "include_thoughts")
                if key in thinking and (isinstance(thinking[key], (str, bool)) or _number(thinking[key]))
            }
    return result

def _request(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or any(key not in value for key in _REQUEST_FIELDS):
        return None
    if not all(isinstance(value[key], str) for key in ("system", "user", "role")):
        return None
    if not isinstance(value["schema"], dict):
        return None
    result = {key: value[key] for key in _REQUEST_FIELDS}
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return result

def _request_attempt(value: Any) -> int:
    return value if type(value) is int and value > 0 else 1

def _prior_attempts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    attempts = []
    for saved in value:
        if not isinstance(saved, dict):
            continue
        metadata = saved.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        request = _request(saved.get("request"))
        attempts.append({
            "feedback_request_attempt": _request_attempt(saved.get("feedback_request_attempt")),
            "phase": _text(saved.get("phase")) or "unavailable",
            "error": _text(saved.get("error")),
            "request": request,
            "request_is_exact": request is not None,
            "raw_output": _text(metadata.get("raw_completion")),
            "model": _text(metadata.get("model")) or _text(metadata.get("configured_model")),
            "metadata": _metadata(metadata),
            "retry_reason": _text(saved.get("retry_reason")),
        })
    return attempts

def collect_feedback_report_data(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {root}")
    warnings: list[str] = []
    grouped_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in _feedback_rows(root / "brs_feedback.jsonl", warnings):
        cell = _cell(row.get("feedback_key"))
        if cell is None:
            trajectory = row.get("trajectory_idx", row.get("run_idx"))
            episode = row.get("episode_position", row.get("episode_idx"))
            if type(trajectory) is int and type(episode) is int and trajectory > 0 and episode > 0:
                cell = trajectory, episode
        if cell is not None:
            grouped_rows.setdefault(cell, []).append(row)

    journals: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted((root / "brs_state").glob("t*-e*.json")):
        cell = _cell(path.stem)
        if cell is None:
            continue
        journal = _json_file(path, warnings)
        if journal is None:
            journals.setdefault(cell, {"phase": "unreadable", "error": "Journal could not be read."})
            continue
        if journal.get("schema_version") not in (2, 3):
            warnings.append(f"{path.name}: unsupported historical feedback schema; existing reports were left unchanged.")
            continue
        if _cell(journal.get("key")) != cell:
            warnings.append(f"Ignored identity-mismatched journal {path.name}.")
            continue
        if cell in journals:
            warnings.append(f"Duplicate journal identity for trajectory {cell[0]}, episode {cell[1]}; kept one row.")
            if path.stem != f"t{cell[0]:03d}-e{cell[1]:03d}":
                continue
        journals[cell] = journal

    episodes = []
    for cell in sorted(set(journals) | set(grouped_rows)):
        journal = journals.get(cell, {})
        key = f"t{cell[0]:03d}-e{cell[1]:03d}"
        record = journal.get("record")
        attempt = _text(record.get("attempt_id")) if isinstance(record, dict) else None
        rows = grouped_rows.get(cell, [])
        if not attempt:
            attempt = next(
                (_text(row.get("attempt_id")) for row in reversed(rows) if _text(row.get("attempt_id"))),
                None,
            )
        if attempt:
            rows = [row for row in rows if row.get("attempt_id") == attempt]
        public_history = _text(journal.get("public_history")) if journal.get("schema_version") == 3 else None
        if journal.get("schema_version") == 3 and public_history is None:
            warnings.append(f"{key}: saved public history unavailable; private recovery record was not used.")
        saved_observers = journal.get("observers")
        saved_observers = saved_observers if isinstance(saved_observers, dict) else {}
        parties = {agent for agent in saved_observers if isinstance(agent, str) and agent.strip()} or None
        raw_names = journal.get("display_names")
        display_names = {
            agent: name for agent, name in raw_names.items()
            if parties is not None and agent in parties and isinstance(name, str) and name.strip()
        } if isinstance(raw_names, dict) else {}
        observer_ids = set(saved_observers) | {
            row["agent_id"] for row in rows if isinstance(row.get("agent_id"), str)
        }
        if parties is not None:
            observer_ids |= parties
        observers = []
        for observer_id in sorted(value for value in observer_ids if isinstance(value, str) and value.strip()):
            if parties is not None and observer_id not in parties:
                warnings.append(f"{key}: unknown observer {observer_id} omitted.")
                continue
            saved = saved_observers.get(observer_id)
            saved = saved if isinstance(saved, dict) else {}
            observer_rows = [row for row in rows if row.get("agent_id") == observer_id]
            request_attempt = (
                _request_attempt(saved.get("feedback_request_attempt")) if saved else
                max((_request_attempt(row.get("feedback_request_attempt")) for row in observer_rows), default=1)
            )

            observer_rows = [row for row in observer_rows
                             if _request_attempt(row.get("feedback_request_attempt")) == request_attempt]
            last = observer_rows[-1] if observer_rows else {}
            targets = parties - {observer_id} if parties is not None else None
            request = _request(saved.get("request"))
            if request is None:
                request = next((_request(row.get("request")) for row in reversed(observer_rows)
                                if _request(row.get("request")) is not None), None)
            phase = _text(saved.get("phase")) or _text(last.get("status")) or "prepared"
            if request is None and phase != "prepared":
                warnings.append(f"{key}/{observer_id}: exact observer request was not saved or is unavailable.")
            metadata = saved.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            raw_output = _text(metadata.get("raw_completion"))
            for row in reversed(observer_rows):
                candidate = row.get("metadata")
                if isinstance(candidate, dict):
                    if not metadata:
                        metadata = candidate
                    if raw_output is None and isinstance(candidate.get("raw_completion"), str):
                        raw_output = candidate["raw_completion"]
                        break
            counts_raw = saved.get("counts")
            feedback_raw = saved.get("feedback")
            scratchpad_raw = saved.get("scratchpad")
            for row in reversed(observer_rows):
                if row.get("status") == "validated":
                    if counts_raw is None:
                        counts_raw = row.get("counts")
                    if feedback_raw is None:
                        feedback_raw = row.get("feedback")
                    if scratchpad_raw is None:
                        scratchpad_raw = row.get("scratchpad")
                    break
            if phase == "fallback_no_update":
                counts_raw = feedback_raw = scratchpad_raw = None
                warnings.append(f"{key}/{observer_id}: feedback update skipped; previous trust retained.")
            counts = _validated_pairs(counts_raw, ObservationCounts, targets)
            feedback = _validated_pairs(feedback_raw, FeedbackPair, targets)
            scratchpad = _scratchpad(scratchpad_raw, targets if targets is not None else set(counts) if counts else None)
            if counts is not None and observer_id in counts:
                counts = None
            if feedback is not None and observer_id in feedback:
                feedback = None
            if scratchpad is not None and observer_id in scratchpad:
                scratchpad = None
            if counts_raw is not None and counts is None:
                warnings.append(f"{key}/{observer_id}: invalid or incomplete observation counts omitted.")
            if feedback_raw is not None and feedback is None:
                warnings.append(f"{key}/{observer_id}: invalid or incomplete normalized feedback omitted.")
            if scratchpad_raw is not None and scratchpad is None:
                warnings.append(f"{key}/{observer_id}: invalid or incomplete feedback scratchpad omitted.")
            empty_scratchpad_targets = sorted(
                target for target, summary in (scratchpad or {}).items() if not summary.strip()
            )
            if empty_scratchpad_targets:
                warnings.append(
                    f"{key}/{observer_id}: model returned blank feedback explanations for "
                    f"{', '.join(empty_scratchpad_targets)}; reported counts and scores are retained."
                )
            before = journal.get("evidence_before")
            after = journal.get("evidence_after")
            score_matrix = journal.get("scores")
            before = before.get(observer_id) if isinstance(before, dict) else None
            after = after.get(observer_id) if isinstance(after, dict) else None
            scores = score_matrix.get(observer_id) if isinstance(score_matrix, dict) else None
            error = None
            if phase in {"failed", "requested", "fallback_no_update"}:
                error = _text(saved.get("error")) or next(
                    (_text(row.get("error")) for row in reversed(observer_rows) if _text(row.get("error"))), None,
                )
            observers.append({
                "agent_id": observer_id, "phase": phase, "request": request,
                "feedback_request_attempt": request_attempt,
                "retry_reason": _text(saved.get("retry_reason")) or _text(last.get("retry_reason")),
                "feedback_correction_attempt": saved.get("feedback_correction_attempt", 0),
                "update_skipped": phase == "fallback_no_update",
                "fallback_reason": _text(saved.get("fallback_reason")) or _text(last.get("fallback_reason")),
                "prior_attempts": _prior_attempts(saved.get("prior_attempts")),
                "request_is_exact": request is not None,
                "request_label": "Exact saved observer request (includes own private context and prior trust)"
                    if request is not None else "Unavailable: exact observer request not saved",
                "counts": counts, "feedback": feedback, "scratchpad": scratchpad,
                "empty_scratchpad_targets": empty_scratchpad_targets,
                "raw_output": raw_output,
                "error": error, "model": _text(metadata.get("model")) or _text(metadata.get("configured_model")),
                "metadata": _metadata(metadata),
                "evidence_before": _validated_pairs(before, AccumulatedEvidence, targets),
                "evidence_after": _validated_pairs(after, AccumulatedEvidence, targets),
                "scores": _scores(scores, targets) if not isinstance(scores, dict) or observer_id not in scores else None,
                "delivered": observer_id in (journal.get("delivered") or []),
            })
        episodes.append({
            "key": key, "trajectory_idx": cell[0], "episode_idx": cell[1],
            "attempt_id": attempt, "phase": _text(journal.get("phase")) or "collecting",
            "public_history": public_history, "display_names": display_names,
            "history_label": "Exact saved public history (same format as freeform and metrics)"
                if public_history is not None else
                "Unavailable: formatted public history was not saved; exact observer requests remain below when available.",
            "observers": observers,
            "error": _text(journal.get("error")),
        })
    if not episodes:
        warnings.append("No per-observer BRS feedback journals or events were found.")
    return {
        "run_name": root.name, "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_scope": "Exact requests include each observer's own private negotiation context and prior trust.",
        "episodes": episodes, "warnings": list(dict.fromkeys(warnings)),
    }

def _script_json(data: dict[str, Any]) -> str:
    value = json.dumps(data, ensure_ascii=False, allow_nan=False)
    for character, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                               ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        value = value.replace(character, escaped)
    return value

def write_feedback_report(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    data = collect_feedback_report_data(root)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count(DATA_TOKEN) != 1:
        raise ValueError("Feedback report template must contain exactly one data token")
    document = template.replace(DATA_TOKEN, _script_json(data))
    output = root / "feedback_report.html"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".feedback-report-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    print(write_feedback_report(args.run_dir))

if __name__ == "__main__":
    main()

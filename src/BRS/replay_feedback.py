"""Reassess saved BRS histories with the current rubric, without negotiation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from src.BRS.brs import normalize_counts, scores_from_state, update_brs
from src.BRS.feedback import (build_agent_feedback_schema, parse_agent_feedback_output,
                              parse_agent_feedback_output_with_diagnostics)
from src.BRS.recovery import output_correction_error, parse_saved_feedback, saved_feedback_is_blocked
from src.prompts.feedback_prompt import build_feedback_user_message
from src.BRS.prompt_support import brs_prompt_fingerprint, build_system_trust_block
from src.agents.prompt_adapter import SUPPLEMENT_PROMPT_PROTOCOL

SEMANTICS = (
    "Frozen public histories and original observer priors. Each new score is a single "
    "BRS update from that source episode's evidence_before; new scores never feed into "
    "later replay requests. This compares feedback, not new negotiation outcomes or "
    "a new longitudinal trust trajectory."
)

class ReplayError(RuntimeError):
    pass

def replace_system_trust_block(
    system_prompt: str,
    trust_lines: str,
    *,
    initial_score: float = 0.5,
    hint_level: str = "prescriptive",
    is_malicious: bool = False,
    min_blocking_rejects: int = 2,
) -> str:
    start_marker = "\nTrust:\n"
    end_marker = "\n\nIn your public messages,"
    if system_prompt.count(start_marker) != 1 or system_prompt.count(end_marker) != 1:
        raise ValueError("Saved V5 system prompt must contain one unambiguous trust block")
    start = system_prompt.index(start_marker) + 1
    end = system_prompt.index(end_marker)
    if end <= start:
        raise ValueError("Saved V5 system prompt has invalid trust block boundaries")
    replacement = build_system_trust_block(
        trust_lines,
        initial_score=initial_score,
        hint_level=hint_level,
        is_malicious=is_malicious,
        min_blocking_rejects=min_blocking_rejects,
    )
    return system_prompt[:start] + replacement + system_prompt[end:]

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)

def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()

def _atomic(path: Path, value: Any, *, text: bool = False) -> None:
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".replay-", delete=False) as handle:
            name = handle.name
            handle.write(value if text else _json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)

def _safe_paths(source_dir: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    source = Path(source_dir).expanduser().resolve(strict=True)
    output_arg = Path(out_dir).expanduser().absolute()
    if any(path.is_symlink() for path in [output_arg, *output_arg.parents]):
        raise ReplayError("Output path must not contain symlinks")
    output = output_arg.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ReplayError("Source and output directories must be separate, without overlap")
    if not source.is_dir() or (output.exists() and not output.is_dir()):
        raise ReplayError("Source and output must be directories")
    if output.exists() and any(path.is_symlink() for path in output.rglob("*")):
        raise ReplayError("Output directory must not contain symlinks")
    return source, output

@contextmanager
def _lock(output: Path):
    import fcntl
    with (output / ".replay.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReplayError("Another replay process is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def _source_data(source: Path) -> tuple[dict, list[dict], list[dict]]:
    paths = [source / "experiment.json", *sorted((source / "brs_state").glob("t*-e*.json"))]
    if len(paths) < 2:
        raise ReplayError("Source requires experiment.json and completed BRS episode journals")
    manifest, values = [], []
    for path in paths:
        if path.is_symlink() or source not in path.resolve().parents:
            raise ReplayError(f"Source input cannot escape its directory: {path}")
        raw = path.read_bytes()
        manifest.append({"path": str(path.relative_to(source)), "sha256": hashlib.sha256(raw).hexdigest()})
        values.append(json.loads(raw))
    experiment, journals = values[0], values[1:]
    config = experiment["run_config"]
    if config.get("supplement_prompt_protocol") != SUPPLEMENT_PROMPT_PROTOCOL:
        raise ReplayError(
            "Saved-history reassessment requires a source run from this supplement's "
            "strategic S/S+P prompt protocol; missing or incompatible "
            "supplement_prompt_protocol"
        )
    if config.get("trust_mode") != "brs" or config.get("prompt_version") != "v5":
        raise ReplayError("Saved-history reassessment currently requires a BRS V5 source run")
    for path, journal in zip(paths[1:], journals):
        if journal.get("schema_version") != 3 or journal.get("phase") != "committed":
            raise ReplayError("Every source episode journal must be schema-v3 and committed")
        if journal.get("key") != path.stem:
            raise ReplayError("Source journal key must match its filename")
    return experiment, journals, manifest

def _source_trust_lines(request: dict, observer: str, targets: list[str], history: str) -> str:
    user = request["user"]
    begin = f"Your agent ID: {observer}\n\nCurrent trust scores:\n"
    end = f"\n\nParties to assess: {', '.join(targets)}\n\nPublic episode history:\n<HISTORY>\n"
    if user.count(begin) != 1 or user.count(end) != 1:
        raise ReplayError("Cannot unambiguously recover saved observer/trust context")
    prefix, saved_history = user.split(end, 1)
    if not saved_history.startswith(history + "\n</HISTORY>"):
        raise ReplayError("Saved feedback input does not contain the exact journal public history")
    return prefix.split(begin, 1)[1]

def prepare_replay(source: Path, *, registry_path: Path, dry_run: bool) -> tuple[dict, dict, dict]:
    experiment, journals, source_manifest = _source_data(source)
    config = experiment["run_config"]
    aliases = config.get("agent_models") or experiment.get("agent_models", {})
    factor = config["brs_config"]["forgetting_factor"]
    jobs = {}
    for journal in journals:
        observers = journal["observers"]
        for observer, row in observers.items():
            source_fallback = row.get("phase") == "fallback_no_update"
            if row.get("phase") not in {"validated", "fallback_no_update"} or not aliases.get(observer):
                raise ReplayError("Source must have resolved feedback and model aliases for every observer")
            request = deepcopy(row["request"])
            if set(request) != {"system", "user", "schema", "max_tokens", "temperature", "seed", "role"}:
                raise ReplayError("Unsupported saved feedback request fields")
            targets = request["schema"]["properties"]["feedback"]["required"]
            if set(targets) != set(observers) - {observer}:
                raise ReplayError("Source request targets do not match episode participants")
            if request["schema"] != build_agent_feedback_schema(targets) or request["role"] != "trust_feedback":
                raise ReplayError("Saved feedback contract is incompatible with this replay")
            history = journal["public_history"]
            trust_lines = _source_trust_lines(request, observer, targets, history)
            before = journal["evidence_before"][observer]
            if source_fallback:
                if (not row.get("update_skipped") or any(row.get(field) is not None
                        for field in ("scratchpad", "counts", "feedback"))):
                    raise ReplayError("Source fallback must explicitly omit feedback and hold its evidence")
                source_feedback = None
                source_after = deepcopy(before)
            else:
                source_feedback = parse_agent_feedback_output(
                    _json({"scratchpad": row["scratchpad"], "feedback": row["counts"]}), targets,
                )
                source_after = update_brs(before, normalize_counts(source_feedback.feedback), factor)
            source_scores = scores_from_state(source_after)
            if any(not math.isclose(score, journal["scores"][observer][target], abs_tol=1e-12)
                   for target, score in source_scores.items()):
                raise ReplayError("Source scores disagree with saved feedback/evidence")
            request["system"] = replace_system_trust_block(
                request["system"], trust_lines,
                initial_score=config["trust_initial_score"],
                hint_level=config.get("trust_decision_hint_level", "prescriptive"),
                is_malicious=journal["record"].get("malicious_agent_id") == observer,
                min_blocking_rejects=config.get("trust_min_blocking_rejects", 2),
            )
            request["user"] = build_feedback_user_message(history, observer, targets, trust_lines)
            key = f"{journal['key']}:{observer}"
            if key in jobs:
                raise ReplayError(f"Duplicate saved observer episode: {key}")
            jobs[key] = {
                "episode_key": journal["key"], "observer_id": observer,
                "targets": targets, "model_alias": aliases[observer],
                "public_history": history, "source_request": row["request"],
                "request": request, "phase": "prepared", "synthetic": dry_run,
                "source_prior_scores": scores_from_state(before), "evidence_before": before,
                "source": {
                    "scratchpad": row.get("scratchpad"), "counts": row.get("counts"),
                    "feedback": ({key: value.model_dump() for key, value in normalize_counts(source_feedback.feedback).items()}
                                 if source_feedback is not None else None),
                    "evidence_after": {key: value.model_dump() if hasattr(value, "model_dump") else value
                                       for key, value in source_after.items()},
                    "scores": source_scores,
                },
            }
            if source_fallback:
                jobs[key]["source"]["update_skipped"] = True
    registry_bytes = registry_path.read_bytes() if registry_path.is_file() else None
    manifest = {
        "schema_version": 1, "source_dir": str(source), "source_files": source_manifest,
        "semantics": SEMANTICS, "dry_run": dry_run,
        "synthetic_warning": "SYNTHETIC DRY RUN — no model judgments or scientific results" if dry_run else None,
        "prompt_sha256": brs_prompt_fingerprint(), "requests_sha256": _hash(jobs),
        "model_registry_path": str(registry_path),
        "model_registry_sha256": hashlib.sha256(registry_bytes).hexdigest() if registry_bytes is not None else None,
        "source_client_provenance": config.get("resolved_client_provenance", {}),
        "saved_llm_config": config["llm_config"], "forgetting_factor": factor,
        "source_feedback_prompt_sha256": config["brs_config"].get("feedback_prompt_sha256"),
        "expected_feedback_calls": len(jobs),
    }
    state = {"schema_version": 1, "manifest_sha256": _hash(manifest), "jobs": jobs}
    return manifest, state, config

def _client_factory(config: dict, registry_path: Path, output: Path, dry_run: bool):
    from src.main import build_llm_client_factory
    from src.utils.config_loader import load_model_registry
    from src.utils.types import GameConfig
    game_config = GameConfig.model_validate({
        "game": {"num_episodes": 1, "pass_rule": "n_minus_1_total", "rounds_per_episode_multiplier": 1,
                 "base_seed": config.get("base_seed", 0)},
        "llm": {**config["llm_config"], "dry_run": dry_run},
        "trust": {"mode": "brs", "initial_score": config["trust_initial_score"],
                  "brs": {"forgetting_factor": config["brs_config"]["forgetting_factor"]}},
        "prompts": config["prompt_config"], "malicious": config["malicious_config"],
        "logging": {"out_dir": str(output), "log_full_prompts": True, "log_full_completions": True},

        "scenario_paths": ["saved-history-only"], "agent_models": config["agent_models"],
    })
    return build_llm_client_factory(game_config, load_model_registry(registry_path), dry_run)

def _client_provenance(client: Any, alias: str) -> dict:
    from src.game.engine import GameEngine
    return GameEngine._describe_client(client, configured_model=alias)

def _comparisons(state: dict, manifest: dict) -> dict:
    rows, usage = [], dict.fromkeys(("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"), 0)
    for key, job in state["jobs"].items():
        for attempt in [*job.get("prior_attempts", []), job]:
            for request in attempt.get("metadata", {}).get("requests", []):
                for name in usage:
                    value = request.get(name)
                    if name == "total_tokens" and value is None:
                        value = request.get("prompt_tokens", 0) + request.get("completion_tokens", 0)
                    if isinstance(value, int) and value >= 0:
                        usage[name] += value
        if job["phase"] != "validated" or job["source"].get("update_skipped"):
            continue
        for target in job["targets"]:
            old, new = job["source"], job["result"]
            rows.append({"key": key, "target": target,
                         "source_counts": old["counts"][target], "new_counts": new["counts"][target],
                         "source_scratchpad": old["scratchpad"][target], "new_scratchpad": new["scratchpad"][target],
                         "source_score": old["scores"][target], "new_score": new["scores"][target],
                         "source_feedback": old["feedback"][target], "new_feedback": new["feedback"][target]})
    totals = {}
    for prefix in ("source", "new"):
        counts = [row[f"{prefix}_counts"] for row in rows]
        totals[prefix] = {"P": sum(row["P"] for row in counts), "N": sum(row["N"] for row in counts),
                          "zero_pairs": sum(row["P"] == row["N"] == 0 for row in counts)}
    jobs = list(state["jobs"].values())
    valid = sum(job["phase"] == "validated" for job in jobs)
    fallback = sum(job["phase"] == "fallback_no_update" for job in jobs)
    return {"semantics": SEMANTICS, "synthetic": manifest["dry_run"],
            "completed_requests": valid + fallback,
            "valid_feedback_requests": valid, "fallback_requests": fallback,
            "missing_feedback_requests": len(jobs) - valid,
            "missing_feedback_rate": (len(jobs) - valid) / len(jobs) if jobs else 0,
            "source_fallback_requests": sum(bool(job["source"].get("update_skipped")) for job in jobs),
            "held_score_jobs": [key for key, job in state["jobs"].items() if job["phase"] == "fallback_no_update"],
            "expected_requests": len(state["jobs"]), "paired_assessments": len(rows),
            "changed_counts": sum(row["source_counts"] != row["new_counts"] for row in rows),
            "totals_on_completed_pairs": totals, "recorded_usage": usage, "comparisons": rows}

def _write_reports(output: Path, state: dict, manifest: dict) -> None:
    comparison = _comparisons(state, manifest)
    _atomic(output / "comparisons.json", comparison)
    esc = lambda value: html.escape(str(value))
    body = ["<!doctype html><html><head><meta charset='utf-8'><title>BRS feedback replay</title>",
            "<style>body{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#172231}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:1rem}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{padding:.6rem;border:1px solid #ccd3dd;text-align:left}"
            "summary{cursor:pointer;padding:.6rem;font-weight:600}details{margin:1rem 0;border:1px solid #ccd3dd;padding:.5rem}"
            ".warning{background:#fff0bd;padding:1rem}nav a{margin-right:1rem}</style></head><body>",
            "<h1>BRS feedback on saved histories</h1>", f"<p>{esc(SEMANTICS)}</p>"]
    if manifest["dry_run"]:
        body.append(f"<p class='warning'><strong>{esc(manifest['synthetic_warning'])}</strong></p>")
    body.append("<p>Local private artifact: request details include each observer's own private briefing. "
                "Counts describe feedback changes, not human-validated accuracy.</p>")
    body.append(f"<p>Completed {comparison['completed_requests']}/{comparison['expected_requests']} requests; "
                f"{comparison['changed_counts']}/{comparison['paired_assessments']} paired counts changed.</p>")
    body.append("<pre>" + esc(_json({key: value for key, value in comparison.items() if key != "comparisons"})) + "</pre>")
    episodes = sorted({job["episode_key"] for job in state["jobs"].values()})
    body.append("<nav>" + " ".join(f"<a href='#{esc(key)}'>{esc(key)}</a>" for key in episodes) + "</nav>")
    for episode in episodes:
        jobs = [job for job in state["jobs"].values() if job["episode_key"] == episode]
        body.append(f"<h2 id='{esc(episode)}'>{esc(episode)}</h2><details><summary>Saved public history</summary><pre>" + esc(jobs[0]["public_history"]) + "</pre></details>")
        for job in jobs:
            body.append(f"<details><summary>{esc(job['observer_id'])} — {esc(job['phase'])}</summary>")
            if job["phase"] in {"validated", "fallback_no_update"}:
                body.append("<table><tr><th>Counterpart</th><th>Source P,N → new P,N</th><th>Source score → one-step new score</th></tr>")
                for target in job["targets"]:
                    old, new = job["source"], job["result"]
                    def count_text(value):
                        pair = (value.get("counts") or {}).get(target)
                        return f"{pair['P']},{pair['N']}" if pair is not None else "Missing feedback (scores held)"
                    body.append(f"<tr><td>{esc(target)}</td><td>{esc(count_text(old))} → "
                                f"{esc(count_text(new))}</td><td>"
                                f"{old['scores'][target]:.6f} → {new['scores'][target]:.6f}</td></tr>")
                    body.append(f"<tr><td colspan='3'><b>Source:</b> {esc((old.get('scratchpad') or {}).get(target, 'Unavailable'))}<br>"
                                f"<b>New:</b> {esc((new.get('scratchpad') or {}).get(target, 'Unavailable'))}</td></tr>")
                body.append("</table>")
            if job.get("error"):
                body.append(f"<p class='warning'>{esc(job['error'])}</p>")
            for label, data in (("New request (observer's own private context)", job["request"]),
                                ("Saved source request", job["source_request"]),
                                ("Response, usage, evidence and provenance", {k: v for k, v in job.items()
                                 if k not in {"request", "source_request", "public_history"}})):
                body.append(f"<details><summary>{esc(label)}</summary><pre>{esc(_json(data))}</pre></details>")
            body.append("</details>")
    body.append("</body></html>")
    _atomic(output / "feedback_replay.html", "".join(body), text=True)

def _result_from_raw(job: dict, factor: float) -> dict:
    feedback, diagnostics = parse_agent_feedback_output_with_diagnostics(
        job["raw_completion"], job["targets"], allow_wrapping=True,
    )
    job["parse_diagnostics"] = diagnostics
    normalized = normalize_counts(feedback.feedback)
    after = update_brs(job["evidence_before"], normalized, factor)
    return {
        "scratchpad": feedback.scratchpad,
        "counts": {key: value.model_dump() for key, value in feedback.feedback.items()},
        "feedback": {key: value.model_dump() for key, value in normalized.items()},
        "evidence_after": {key: value.model_dump() for key, value in after.items()},
        "scores": scores_from_state(after),
    }

def _recovery_observer(job: dict) -> dict:
    return {"phase": job["phase"], "error": job.get("error", "Invalid saved feedback"),
            "request": job.get("active_request", job["request"]),
            "metadata": job.get("metadata", {}),
            "feedback_correction_attempt": job.get("feedback_correction_attempt", 0)}

def _hold_result(job: dict) -> dict:
    return {"scratchpad": None, "counts": None, "feedback": None,
            "evidence_after": deepcopy(job["evidence_before"]),
            "scores": scores_from_state(job["evidence_before"]), "update_skipped": True}

def _resolve_saved_response(job: dict, factor: float) -> bool:
    phase = job["phase"]
    if phase == "fallback_no_update":
        candidate = {**_recovery_observer(job), "phase": "failed", "feedback_correction_attempt": 0}
        if ((not saved_feedback_is_blocked(candidate)
                and (job.get("feedback_correction_attempt") != 1
                     or output_correction_error(candidate) is not None))
                or job.get("result") != _hold_result(job)):
            raise ReplayError("Saved fallback disagrees with its failed response or held evidence")
        return True
    if phase == "validated":
        if job.get("result") != _result_from_raw(job, factor):
            raise ReplayError("Saved replay result disagrees with its raw response")
        return True
    if phase == "responded":
        try:
            result = _result_from_raw(job, factor)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            job.update(phase="failed", error=f"{type(exc).__name__}: {exc}")
        else:
            job.update(result=result, phase="validated")
            job.pop("error", None)
            return True
    if job["phase"] == "failed":
        if saved_feedback_is_blocked(_recovery_observer(job)):
            job.update(result=_hold_result(job), phase="fallback_no_update",
                       update_skipped=True, fallback_reason="content_filter")
            return True

        assessment = parse_saved_feedback(_recovery_observer(job), job["targets"])
        if assessment is not None:
            job["raw_completion"] = job["metadata"]["raw_completion"]
            job.update(result=_result_from_raw(job, factor), phase="validated",
                       local_validation_recovery=True)
            job.pop("error", None)
            return True
        candidate = {**_recovery_observer(job), "feedback_correction_attempt": 0}
        if job.get("feedback_correction_attempt") == 1 and output_correction_error(candidate) is None:
            job.update(result=_hold_result(job), phase="fallback_no_update",
                       update_skipped=True, fallback_reason=job.get("error"))
            return True
    return False

def _validate_correction_history(job: dict) -> None:
    from src.BRS.response_policy import build_feedback_correction_request

    attempt = job.get("feedback_correction_attempt", 0)
    prior = job.get("prior_attempts", [])
    if type(attempt) is not int or attempt not in (0, 1) or not isinstance(prior, list) or len(prior) != attempt:
        raise ReplayError("Invalid saved feedback correction budget")
    if not attempt:
        if job.get("active_request", job["request"]) != job["request"]:
            raise ReplayError("Saved replay request changed without a correction")
        return
    original = prior[0]
    if (not isinstance(original, dict) or original.get("request") != job["request"]
            or output_correction_error(original) is not None):
        raise ReplayError("Invalid saved feedback correction history")
    expected = build_feedback_correction_request(
        job["request"], native_structured=job.get("correction_native_structured", False), attempt_idx=1,
    )
    if job.get("active_request") != expected:
        raise ReplayError("Saved corrective request does not match its original settings")

def _prepare_correction(job: dict, client: Any) -> None:
    from src.BRS.response_policy import build_feedback_correction_request

    observer = _recovery_observer(job)
    reason = output_correction_error(observer)
    if reason is not None:
        raise ReplayError(f"Saved feedback cannot receive a corrective request: {reason}")
    native = bool(getattr(client, "native_structured_output_enabled", False))
    archived = deepcopy(observer)
    archived["raw_completion"] = job.get("raw_completion")
    job.update(active_request=build_feedback_correction_request(job["request"], native_structured=native, attempt_idx=1),
               prior_attempts=[archived], feedback_correction_attempt=1,
               correction_native_structured=native, retry_reason="feedback_output_correction")
    job.pop("raw_completion", None)
    job.pop("metadata", None)
    job.pop("error", None)

def replay_feedback(source_dir: str | Path, out_dir: str | Path, *, model_registry: str | Path,
                    dry_run: bool = False, resume: bool = False, prepare_only: bool = False,
                    client_factory: Callable | None = None) -> dict:
    source, output = _safe_paths(source_dir, out_dir)
    registry = Path(model_registry).expanduser().resolve()
    manifest, prepared, config = prepare_replay(source, registry_path=registry, dry_run=dry_run)
    output.mkdir(parents=True, exist_ok=True)
    with _lock(output):
        state_path, manifest_path = output / "replay_state.json", output / "manifest.json"
        if resume:
            if not state_path.is_file() or not manifest_path.is_file():
                raise ReplayError("--resume requires a replay_state.json and manifest.json")
            old_manifest = json.loads(manifest_path.read_text())
            state = json.loads(state_path.read_text())
            if old_manifest != manifest or state.get("manifest_sha256") != _hash(manifest):
                raise ReplayError("Source files, prompt, registry, or dry-run mode changed; use a fresh output directory")
            if set(state.get("jobs", {})) != set(prepared["jobs"]):
                raise ReplayError("Saved replay job coverage changed")
            for key, job in state["jobs"].items():
                for field, value in prepared["jobs"][key].items():
                    if field != "phase" and job.get(field) != value:
                        raise ReplayError(f"Saved replay request/context changed for {key}")
        else:
            if any(path.name != ".replay.lock" for path in output.iterdir()):
                raise ReplayError("Fresh replay requires an empty output directory; use --resume for this replay")
            state = prepared
            _atomic(manifest_path, manifest)
            _atomic(state_path, state)
        try:
            for job in state["jobs"].values():
                _validate_correction_history(job)
                _resolve_saved_response(job, manifest["forgetting_factor"])
            _atomic(state_path, state)
            pending = [key for key, job in state["jobs"].items()
                       if job["phase"] not in {"prepared", "validated", "fallback_no_update"}
                       and output_correction_error(_recovery_observer(job)) is not None]
            if pending:
                raise ReplayError(f"Unanswered or ineligible feedback is never automatically reissued: {', '.join(pending)}. Inspect replay_state.json.")
            if prepare_only or all(job["phase"] in {"validated", "fallback_no_update"} for job in state["jobs"].values()):
                return _comparisons(state, manifest)
            factory = client_factory or _client_factory(config, registry, output, dry_run)
            clients = {}
            for key, job in state["jobs"].items():
                if job["phase"] in {"validated", "fallback_no_update"}:
                    continue
                alias = job["model_alias"]
                if alias not in clients:
                    clients[alias] = factory(model_override=alias)
                client = clients[alias]
                provenance = _client_provenance(client, alias)
                expected = manifest["source_client_provenance"].get(job["observer_id"])
                if not dry_run and client_factory is None:
                    if not expected or provenance != expected:
                        raise ReplayError(f"Client provenance differs from the source for {job['observer_id']}; restore its model registry settings")
                if dry_run:
                    client.brs_feedback_output = _json({
                        "scratchpad": {target: "SYNTHETIC DRY RUN: no evidence was assessed." for target in job["targets"]},
                        "feedback": {target: {"P": 0, "N": 0} for target in job["targets"]},
                    })
                while job["phase"] not in {"validated", "fallback_no_update"}:
                    if job["phase"] == "failed":
                        _prepare_correction(job, client)
                    job.update(phase="requested", client_provenance=provenance)
                    _atomic(state_path, state)
                    try:
                        raw = client.complete_feedback_once(**job.get("active_request", job["request"]))
                    except Exception as exc:
                        job.update(phase="failed", error=f"{type(exc).__name__}: {exc}",
                                   metadata=deepcopy(getattr(client, "last_call_metadata", {}) or {}))
                    else:
                        job.update(raw_completion=raw,
                                   metadata=deepcopy(getattr(client, "last_call_metadata", {}) or {}), phase="responded")
                    _atomic(state_path, state)
                    _resolve_saved_response(job, manifest["forgetting_factor"])
                    _atomic(state_path, state)
                    if job["phase"] == "failed" and output_correction_error(_recovery_observer(job)) is not None:
                        raise ReplayError(f"Feedback failed for {key}; unanswered or ineligible feedback will not be automatically reissued: {job.get('error')}")
                print(f"Completed {key} ({'synthetic' if dry_run else alias})", flush=True)
            return _comparisons(state, manifest)
        finally:
            _write_reports(output, state, manifest)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Completed BRS V5 log directory (read only)")
    parser.add_argument("--out-dir", required=True, help="Separate fresh output directory, or existing replay with --resume")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--model-registry", help="Defaults to PROJECT_ROOT/config/model_registry.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Offline synthetic zero feedback; never use as experimental results")
    parser.add_argument("--resume", action="store_true", help="Continue prepared requests; reuse validated responses without new calls")
    parser.add_argument("--prepare-only", action="store_true", help="Save exact requests without calling a model; continue with --resume")
    args = parser.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()

    if not args.dry_run and not args.prepare_only:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    try:
        comparison = replay_feedback(args.source_dir, args.out_dir,
                                     model_registry=args.model_registry or root / "config/model_registry.yaml",
                                     dry_run=args.dry_run, resume=args.resume, prepare_only=args.prepare_only)
    except (ReplayError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Feedback replay stopped: {exc}\n")
    print(f"Feedback replay: {comparison['completed_requests']}/{comparison['expected_requests']} requests; "
          f"report: {Path(args.out_dir).resolve() / 'feedback_replay.html'}")

if __name__ == "__main__":
    main()

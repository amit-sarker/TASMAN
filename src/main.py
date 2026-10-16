from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from pydantic import ValidationError

from src.agents.gemini_client import GeminiClient
from src.agents.llm_client import LLMClient
from src.agents.stub_client import StubLLMClient
from src.logger import NegotiationLogger
from src.game.engine import GameEngine
from src.utils.config_loader import load_game_config, load_model_registry, load_scenario
from src.utils.types import GameConfig, TrustConfig

def _fallback_proposer_response() -> str:
    return """{
  "scratchpad": "Starting from a moderate package that is broadly acceptable.",
  "public_answer": "I propose a compromise package that should be workable for most parties while keeping the project viable. This gives everyone something material to work with.",
  "deal": {"A": "A2", "B": "B2", "C": "C2", "D": "D3", "E": "E3"}
}"""

def _fallback_voter_response() -> str:
    return """{
  "scratchpad": "The proposal looks workable enough to keep negotiations moving.",
  "public_answer": "This looks like a plausible compromise and I can support it for now.",
  "vote": true
}"""

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the repeated negotiation game.")
    parser.add_argument("--config", default="config/game_config.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--model-registry", default=None)
    trust_group = parser.add_mutually_exclusive_group()
    trust_group.add_argument("--use-trust", dest="use_trust", action="store_true")
    trust_group.add_argument("--no-use-trust", dest="use_trust", action="store_false")
    parser.set_defaults(use_trust=None)
    parser.add_argument("--episodes", type=int, default=None)
    trajectory_group = parser.add_mutually_exclusive_group()
    trajectory_group.add_argument(
        "--trajectories",
        type=int,
        default=None,
        help="Number of independent longitudinal trajectories",
    )
    trajectory_group.add_argument(
        "--runs",
        dest="legacy_runs",
        type=int,
        default=None,
        help="Deprecated alias for --trajectories",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--initial-score",
        type=float,
        default=None,
        dest="initial_score",
        help="Override trust.initial_score (initial trust prior in [0, 1]) without editing the config.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-console", action="store_true")
    parser.add_argument("--no-console-private", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from checkpoint.json in the output directory. "
            "Skips committed work and restores each trajectory's trust state. "
            "Requires --out-dir to point to an existing run directory."
        ),
    )
    parser.add_argument(
        "--resume-force",
        action="store_true",
        dest="resume_force",
        help=(
            "Skip config-mismatch validation when resuming. "
            "Use only when you are sure the config change is harmless "
            "(e.g. changing parallel_workers or logging settings). "
            "Has no effect without --resume."
        ),
    )
    parser.add_argument(
        "--retry-brs-feedback", action="append", default=[],
        metavar="JOURNAL:AGENT",
        help=(
            "With --resume and trust.mode=brs, explicitly retry one saved transient-failed "
            "feedback request, e.g. t002-e001:aviation_safety. Reuses its exact input; "
            "never replaces valid feedback or repeats negotiation. Repeat for multiple "
            "requests. Invalid model responses and uncertain crash states remain blocked."
        ),
    )
    return parser

def _set_trajectory_count(game_config: GameConfig, count: int) -> None:
    game_config.game.num_trajectories = count
    game_config.game.num_runs = count

def build_llm_client_factory(
    game_config: GameConfig,
    model_registry: dict,
    use_stub: bool = False,
):
    if use_stub:
        def llm_client_factory(model_override=None, endpoint_cfg=None) -> StubLLMClient:
            return StubLLMClient(
                canned={},
                fallback_proposer=_fallback_proposer_response(),
                fallback_voter=_fallback_voter_response(),
            )
    else:
        def llm_client_factory(model_override=None, endpoint_cfg=None) -> LLMClient:
            reasoning_prefixes = tuple(game_config.llm.reasoning_model_prefixes)

            registry_cfg = model_registry.get(model_override or "")
            resolved_cfg = registry_cfg or endpoint_cfg

            if resolved_cfg is not None:
                api_key = os.getenv(resolved_cfg.api_key_env)
                structured_protocol = getattr(
                    resolved_cfg, "structured_output_protocol", None
                )
                requires_registry_key = structured_protocol in {
                    "gemini_native_v1",
                    "kimi_text_json_v1",
                    "deepseek_text_json_v1",
                }
                if requires_registry_key and not api_key:
                    raise ValueError(
                        "Missing API key environment variable "
                        f"{resolved_cfg.api_key_env!r} for model "
                        f"{model_override!r}. Registry-routed clients must not "
                        "silently fall back to a different provider key."
                    )

                api_model = getattr(resolved_cfg, "model_name", None) or model_override

                if (model_override or "").lower().startswith("gemini"):
                    return GeminiClient(
                        api_key=api_key,
                        model=api_model,
                        thinking_budget=resolved_cfg.thinking_budget,
                        supports_seed=getattr(resolved_cfg, "supports_seed", True),
                        structured_output_protocol=getattr(
                            resolved_cfg, "structured_output_protocol", None
                        ),
                        structured_output_regeneration_attempts=getattr(
                            resolved_cfg,
                            "structured_output_regeneration_attempts",
                            0,
                        ),
                        structured_output_reformat_attempts=getattr(
                            resolved_cfg, "structured_output_reformat_attempts", 0
                        ),
                        trajectory_cell_retry_attempts=getattr(
                            resolved_cfg,
                            "trajectory_cell_retry_attempts",
                            0,
                        ),
                        scratchpad_limits=dict(
                            getattr(resolved_cfg, "scratchpad_limits", {}) or {}
                        ),
                        scratchpad_recovery_targets={
                            str(key): list(values)
                            for key, values in dict(
                                getattr(
                                    resolved_cfg,
                                    "scratchpad_recovery_targets",
                                    {},
                                )
                                or {}
                            ).items()
                        },
                    )
                return LLMClient(
                    api_version=resolved_cfg.api_version,
                    reasoning_prefixes=reasoning_prefixes,
                    model_override=api_model,
                    base_url_override=resolved_cfg.base_url,
                    api_key_override=api_key,
                    thinking_budget=resolved_cfg.thinking_budget,
                    reasoning_effort=getattr(
                        resolved_cfg, "reasoning_effort", None
                    ),
                    is_reasoning_model=getattr(
                        resolved_cfg, "is_reasoning_model", None
                    ),
                    merge_system_into_user=getattr(
                        resolved_cfg, "merge_system_into_user", None
                    ),
                    supports_response_format=getattr(
                        resolved_cfg, "supports_response_format", None
                    ),
                    openai_response_format_protocol=getattr(
                        resolved_cfg, "openai_response_format_protocol", None
                    ),
                    supports_temperature=getattr(resolved_cfg, "supports_temperature", False),
                    supports_seed=getattr(resolved_cfg, "supports_seed", True),
                    require_seed_support=getattr(
                        resolved_cfg, "require_seed_support", False
                    ),
                    max_token_parameter=getattr(
                        resolved_cfg,
                        "max_token_parameter",
                        "max_completion_tokens",
                    ),
                    thinking_mode=getattr(
                        resolved_cfg, "thinking_mode", None
                    ),
                    thinking_keep=getattr(
                        resolved_cfg, "thinking_keep", None
                    ),
                    structured_output_protocol=getattr(
                        resolved_cfg, "structured_output_protocol", None
                    ),
                    structured_output_regeneration_attempts=getattr(
                        resolved_cfg,
                        "structured_output_regeneration_attempts",
                        0,
                    ),
                    structured_output_reformat_attempts=getattr(
                        resolved_cfg,
                        "structured_output_reformat_attempts",
                        0,
                    ),
                    trajectory_cell_retry_attempts=getattr(
                        resolved_cfg, "trajectory_cell_retry_attempts", 0
                    ),
                    scratchpad_limits=dict(
                        getattr(resolved_cfg, "scratchpad_limits", {}) or {}
                    ),
                    scratchpad_recovery_targets={
                        str(key): list(values)
                        for key, values in dict(
                            getattr(
                                resolved_cfg,
                                "scratchpad_recovery_targets",
                                {},
                            )
                            or {}
                        ).items()
                    },
                )

            return LLMClient(
                reasoning_prefixes=reasoning_prefixes,
                model_override=model_override,
            )

    return llm_client_factory

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    config_path = Path(args.config)
    game_config = load_game_config(config_path)

    repo_root = Path(args.project_root).resolve() if args.project_root else config_path.resolve().parent.parent
    load_dotenv(repo_root / ".env") or load_dotenv(repo_root / ".env.example")

    model_registry = load_model_registry(
        Path(args.model_registry) if args.model_registry else config_path.parent / "model_registry.yaml"
    )

    if args.use_trust is not None:
        if not args.use_trust:
            game_config.trust.mode = "none"
        elif game_config.trust.mode == "none":
            game_config.trust.mode = "metrics"

    if args.initial_score is not None:
        if not 0.0 <= args.initial_score <= 1.0:
            build_arg_parser().error("--initial-score must be in [0, 1]")
        game_config.trust.initial_score = args.initial_score
    if (game_config.trust.mode == "brs" and not game_config.trust.is_frozen
            and not game_config.trust.random_each_episode
            and game_config.trust.initial_score != 0.5):
        build_arg_parser().error("BRS requires --initial-score 0.5 (fixed Beta(1,1) prior)")
    if args.episodes is not None:
        if args.episodes <= 0:
            build_arg_parser().error("--episodes must be positive")
        game_config.game.num_episodes = args.episodes
    num_trajectories = args.trajectories
    if args.legacy_runs is not None:
        print(
            "Warning: --runs is deprecated; use --trajectories instead.",
            file=sys.stderr,
        )
        num_trajectories = args.legacy_runs
    if num_trajectories is not None:
        if num_trajectories <= 0:
            build_arg_parser().error("--trajectories must be positive")
        _set_trajectory_count(game_config, num_trajectories)
    if args.seed is not None:
        game_config.game.base_seed = args.seed
    if args.out_dir is not None:
        game_config.logging.out_dir = args.out_dir
    if args.no_console:
        game_config.logging.console = False
    if args.no_console_private:
        game_config.logging.console_private = False
    if game_config.trust.replay is not None or game_config.trust.random_each_episode:
        try:
            game_config.trust = TrustConfig.model_validate(game_config.trust.model_dump())
        except ValidationError as exc:
            build_arg_parser().error(str(exc))
    if args.resume and not args.out_dir:
        build_arg_parser().error(
            "--resume requires --out-dir pointing to an existing run directory. "
            "Without --out-dir the logger creates a fresh timestamped directory "
            "and no checkpoint.json will be found."
        )
    if args.retry_brs_feedback and (
        not args.resume or game_config.trust.mode != "brs" or game_config.trust.is_frozen
        or game_config.trust.replay is not None or game_config.trust.random_each_episode
    ):
        build_arg_parser().error(
            "--retry-brs-feedback requires --resume and unfrozen trust.mode=brs "
            "with trust.replay disabled and trust.random_each_episode disabled"
        )

    def _resolve(p: str) -> Path:
        resolved = Path(p)
        if not resolved.is_absolute():
            resolved = (repo_root / resolved).resolve()
        return resolved

    if game_config.trust.replay is not None:
        game_config.trust.replay.source_run = str(_resolve(game_config.trust.replay.source_run))

    scenarios = [load_scenario(_resolve(p)) for p in game_config.scenario_paths]

    logger = NegotiationLogger(
        game_config.logging,
        out_dir_override=args.out_dir,
    )

    use_stub = args.dry_run or getattr(game_config.llm, "dry_run", False)

    game_config.llm.dry_run = bool(use_stub)
    llm_client_factory = build_llm_client_factory(game_config, model_registry, use_stub)

    engine = GameEngine(
        game_config=game_config,
        scenarios=scenarios,
        llm_client_factory=llm_client_factory,
        logger=logger,
        resume=args.resume,
        resume_force=getattr(args, "resume_force", False),
        retry_brs_feedback=args.retry_brs_feedback,
    )
    engine.run()

if __name__ == "__main__":
    main()

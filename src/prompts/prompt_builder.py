from __future__ import annotations

import html
import json
import re
from typing import Any

from src.prompts.freeform_trust_template import FREEFORM_TRUST_UPDATE_BLOCK
from src.prompts.round_template import (
    EMPTY_HISTORY_BLOCK,
    PROPOSER_ROUND_BLOCK,
    TRUST_UPDATE_BLOCK,
    VOTER_ROUND_BLOCK,
)
from src.prompts.system_template import (
    HONEST_TYPE_DISPOSITIONS,
    PRIVATE_OBJECTIVE_HONEST,
    PRIVATE_OBJECTIVE_MALICIOUS_BLOCKER,
    SYSTEM_TEMPLATE,
    TRUST_DECISION_HINT,
    TRUST_DECISION_HINT_SEMANTIC,
    TRUST_VISIBILITY_BLOCK,
)

from src.game.scenario import Scenario
from src.utils.types import AgentConfig, Party, RoundRecord

class PromptLeakError(ValueError):
    pass

def _issues_block(scenario: Scenario) -> str:
    lines: list[str] = []
    for issue in scenario.issues:
        lines.append(f"{issue.id}. {issue.name}: {issue.description.strip()}")
        for option in issue.options:
            lines.append(f"- {option.id}: {option.description}")
    return "\n".join(lines)

def format_history(
    rounds: list[RoundRecord],
    window_size: int,
    parties: list[Party],
) -> str:
    display_names = {party.id: party.display_name for party in parties}
    visible_rounds = rounds[-window_size:]
    parts: list[str] = []
    if len(visible_rounds) < len(rounds):
        parts.append("<omitted/>")
    for round_record in visible_rounds:
        deal_attr = " ".join(
            f"{issue_id}:{option_id}"
            for issue_id, option_id in round_record.deal.choices.items()
        )
        proposer_name = display_names[round_record.proposer_id]
        rationale = html.escape(
            round_record.rationale.strip().replace("\n", " ")
        )
        vote_lines: list[str] = []
        for vote in round_record.votes:
            decision = "accept" if vote.accept else "reject"
            reason = html.escape((vote.reason or "").strip().replace("\n", " "))
            vote_lines.append(
                f'<vote by="{display_names[vote.agent_id]}" '
                f'decision="{decision}">{reason}</vote>'
            )
        outcome = "PASSED" if round_record.passed else "FAILED"
        round_block = (
            f'<round id="{round_record.round_idx}">\n'
            f'<proposal by="{proposer_name}" deal="{deal_attr}" '
            f'accept="implicit">{rationale}</proposal>\n'
            + "\n".join(vote_lines)
            + "\n"
            f"<outcome>{outcome}</outcome>\n"
            f"</round>"
        )
        parts.append(round_block)
    return "\n".join(parts).strip()

def format_trust_lines(
    trust_for_focal: dict[str, float],
    display_names: dict[str, str],
) -> str:
    return "\n".join(
        f"- {display_names[other_id]}: {score:.2f}"
        for other_id, score in trust_for_focal.items()
    )

def assert_no_private_leak(
    prompt_text: str,
    agent_id: str,
    all_agent_configs: dict[str, AgentConfig],
) -> None:
    normalized_prompt = re.sub(r"\s+", " ", prompt_text).strip()
    own = all_agent_configs[agent_id]
    for other_id, other in all_agent_configs.items():
        if other_id == agent_id:
            continue
        normalized_goal = re.sub(r"\s+", " ", other.goal_description).strip()
        normalized_priority = re.sub(
            r"\s+", " ", other.priority_narrative
        ).strip()
        threshold = re.escape(str(other.min_threshold))
        threshold_terms = (
            r"(?:threshold|BATNA|walk.?away|no.deal payoff|minimum acceptable)"
        )
        threshold_pattern = re.compile(
            rf"(?:{threshold_terms}.{{0,60}}(?<![\d.]){threshold}(?![\d.])"
            rf"|(?<![\d.]){threshold}(?![\d.]).{{0,60}}{threshold_terms})",
            re.IGNORECASE,
        )
        if (
            normalized_goal
            and normalized_goal
            != re.sub(r"\s+", " ", own.goal_description).strip()
            and normalized_goal in normalized_prompt
        ):
            raise PromptLeakError(
                f"Prompt for agent '{agent_id}' contains goal_description of "
                f"'{other_id}'"
            )
        if (
            normalized_priority
            and normalized_priority
            != re.sub(r"\s+", " ", own.priority_narrative).strip()
            and normalized_priority in normalized_prompt
        ):
            raise PromptLeakError(
                f"Prompt for agent '{agent_id}' contains priority_narrative of "
                f"'{other_id}'"
            )
        other_description = re.sub(
            r"\s+", " ", getattr(other, "priority_description", "") or ""
        ).strip()
        own_description = re.sub(
            r"\s+", " ", getattr(own, "priority_description", "") or ""
        ).strip()
        if (
            other_description
            and other_description != own_description
            and other_description in normalized_prompt
        ):
            raise PromptLeakError(
                f"Prompt for agent '{agent_id}' contains priority_description of "
                f"'{other_id}'"
            )
        if (
            other.min_threshold != own.min_threshold
            and threshold_pattern.search(normalized_prompt)
        ):
            raise PromptLeakError(
                f"Prompt for agent '{agent_id}' contains threshold of '{other_id}'"
            )

TRUST_DECISION_HINT_LEVELS = frozenset({
    "visible_only",
    "scores_only",
    "semantic",
    "prescriptive",
})

PROPOSER_TRUST_CONSIDER = (
    "; what each party's trust score implies about how far its stated demands "
    "can be taken at face value, and whose support a package can realistically count on"
)
VOTER_TRUST_CONSIDER = (
    "; what your trust score for the proposer implies about how much weight to give "
    "their claims about this package"
)

def format_trust_lines_compact(
    trust_for_focal: dict[str, float],
    display_names: dict[str, str],
) -> str:
    return ", ".join(
        f"{display_names[agent_id]} {trust_for_focal[agent_id]:.2f}"
        for agent_id in display_names
        if agent_id in trust_for_focal
    )

def format_decision_history(
    rounds: list[RoundRecord],
    window_size: int,
    parties: list[Party],
    issue_order: list[str] | None = None,
) -> str:
    display_names = {party.id: party.display_name for party in parties}
    visible_rounds = rounds[-window_size:]
    parts: list[str] = []
    if len(visible_rounds) < len(rounds):
        parts.append("<omitted/>")
    for round_record in visible_rounds:
        ordered_issue_ids = issue_order or list(round_record.deal.choices)
        ordered_choices = {
            issue_id: round_record.deal.choices[issue_id]
            for issue_id in ordered_issue_ids
        }
        deal_json = html.escape(
            json.dumps(ordered_choices, ensure_ascii=True),
            quote=False,
        )
        proposer_name = html.escape(
            display_names[round_record.proposer_id],
            quote=True,
        )
        rationale = html.escape(
            round_record.rationale.strip().replace("\n", " "),
            quote=False,
        )
        vote_lines: list[str] = []
        for vote in round_record.votes:
            decision = "accept" if vote.accept else "reject"
            voter_name = html.escape(display_names[vote.agent_id], quote=True)
            reason = html.escape(
                (vote.reason or "").strip().replace("\n", " "),
                quote=False,
            )
            vote_lines.append(
                f'<vote by="{voter_name}" decision="{decision}">{reason}</vote>'
            )
        outcome = "PASSED" if round_record.passed else "FAILED"
        round_block = (
            f'<round id="{round_record.round_idx}">\n'
            f'<proposal by="{proposer_name}" accept="implicit">\n'
            f"<deal>{deal_json}</deal>\n"
            f"<rationale>{rationale}</rationale>\n"
            f"</proposal>\n"
            + "\n".join(vote_lines)
            + "\n"
            f"<outcome>{outcome}</outcome>\n"
            f"</round>"
        )
        parts.append(round_block)
    return "\n".join(parts).strip()

def _decision_hint(level: str) -> str:
    if level not in TRUST_DECISION_HINT_LEVELS:
        raise ValueError(
            "trust_decision_hint_level must be one of "
            f"{sorted(TRUST_DECISION_HINT_LEVELS)}, got {level!r}"
        )
    if level in {"visible_only", "scores_only"}:
        return ""
    if level == "semantic":
        return TRUST_DECISION_HINT_SEMANTIC
    return (
        TRUST_DECISION_HINT_SEMANTIC.strip()
        + "\n\n"
        + TRUST_DECISION_HINT.strip()
    )

def _round_trust_data_enabled(
    *,
    use_trust_scores: bool,
    level: str,
    is_malicious: bool = False,
) -> bool:
    _decision_hint(level)
    return use_trust_scores and not is_malicious

def _round_hint_enabled(
    *,
    use_trust_scores: bool,
    level: str,
    is_malicious: bool = False,
) -> bool:
    return level != "visible_only" and _round_trust_data_enabled(
        use_trust_scores=use_trust_scores,
        level=level,
        is_malicious=is_malicious,
    )

def build_system_prompt(
    agent_config: AgentConfig,
    scenario: Scenario,
    total_rounds: int,
    n_agents: int,
    trust_config: Any,
    use_trust_scores: bool = False,
    trust_lines: str = "",
    trust_decision_hint_level: str = "prescriptive",
) -> str:
    selected_hint = _decision_hint(trust_decision_hint_level).format(
        min_blocking_rejects=trust_config.min_blocking_rejects,
    )
    agent_type = getattr(agent_config, "agent_type", "") or ""

    priority_description = (
        getattr(agent_config, "priority_description", "") or ""
    ).strip()
    if agent_config.is_malicious:
        private_objective_block = PRIVATE_OBJECTIVE_MALICIOUS_BLOCKER.format(
            display_name=agent_config.display_name,
            priority_description=priority_description,
            priority_narrative=agent_config.priority_narrative.strip(),
            true_threshold=agent_config.min_threshold,
        ).strip()
    else:
        private_objective_block = PRIVATE_OBJECTIVE_HONEST.format(
            priority_description=priority_description,
            priority_narrative=agent_config.priority_narrative.strip(),
            min_threshold=agent_config.min_threshold,
            type_disposition=HONEST_TYPE_DISPOSITIONS.get(agent_type, "").strip(),
            preference_examples_about_others=agent_config.preference_examples_about_others.strip(),
        ).strip()

    trust_visibility_block = ""
    if use_trust_scores and trust_lines:
        trust_visibility_block = TRUST_VISIBILITY_BLOCK.format(
            trust_lines=trust_lines,
            initial_trust_score=trust_config.initial_score,
        ).strip()
        if not agent_config.is_malicious and selected_hint:
            trust_visibility_block += "\n\n" + selected_hint.strip()

    pass_threshold = n_agents - 1
    values = {
        "display_name": agent_config.display_name,
        "goal_description": agent_config.goal_description.strip(),
        "party_list": ", ".join(p.display_name for p in scenario.parties),
        "num_issues": len(scenario.issues),
        "issues_block": _issues_block(scenario),
        "n": n_agents,
        "pass_threshold": pass_threshold,
        "min_blocking_rejects": trust_config.min_blocking_rejects,
        "total_rounds": total_rounds,
        "private_objective_block": private_objective_block,
        "trust_visibility_block": trust_visibility_block,
    }
    result = SYSTEM_TEMPLATE.format(**values)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()

def build_proposer_user_message(
    *,
    round_idx: int,
    total_rounds: int,
    history_text: str,
    use_trust_scores: bool = False,
    initial_deal: str | None = None,
    needed_voter_accepts: int = 3,
    is_malicious: bool = False,
    num_voters: int = 4,
    min_blocking_rejects: int = 2,
    trust_decision_hint_level: str = "prescriptive",
    trust_lines_compact: str = "",
) -> str:
    history = history_text if history_text else EMPTY_HISTORY_BLOCK
    show_trust_data = _round_trust_data_enabled(
        use_trust_scores=use_trust_scores,
        level=trust_decision_hint_level,
        is_malicious=is_malicious,
    )
    if show_trust_data and not trust_lines_compact.strip():
        raise ValueError(
            "trust_lines_compact is required for an honest trust-enabled proposer"
        )
    trust_data_line = (
        f"Your current trust scores: {trust_lines_compact.strip()}"
        if show_trust_data
        else ""
    )
    trust_consider = (
        PROPOSER_TRUST_CONSIDER
        if _round_hint_enabled(
            use_trust_scores=use_trust_scores,
            level=trust_decision_hint_level,
            is_malicious=is_malicious,
        )
        else ""
    )
    opening_hint = (
        f"\nThis is the opening round. You must propose the following deal "
        f"(do not change any option): {initial_deal}\n"
        if initial_deal
        else ""
    )
    result = PROPOSER_ROUND_BLOCK.format(
        round_num=round_idx + 1,
        total_rounds=total_rounds,
        history=history,
        opening_hint=opening_hint,
        trust_data_line=trust_data_line,
        trust_consider=trust_consider,
    )
    return re.sub(r"\n{3,}", "\n\n", result).strip()

def build_voter_user_message(
    *,
    proposer_display_name: str,
    proposed_deal: str,
    proposer_rationale: str,
    round_idx: int,
    total_rounds: int,
    history_text: str,
    use_trust_scores: bool = False,
    is_malicious: bool = False,
    trust_decision_hint_level: str = "prescriptive",
    proposer_trust: float | None = None,
) -> str:
    history = history_text if history_text else EMPTY_HISTORY_BLOCK
    show_trust_data = _round_trust_data_enabled(
        use_trust_scores=use_trust_scores,
        level=trust_decision_hint_level,
        is_malicious=is_malicious,
    )
    if show_trust_data and proposer_trust is None:
        raise ValueError(
            "proposer_trust is required for an honest trust-enabled voter"
        )
    trust_data_line = (
        f"Your current trust score for {proposer_display_name}: {proposer_trust:.2f}."
        if show_trust_data and proposer_trust is not None
        else ""
    )
    trust_consider = (
        VOTER_TRUST_CONSIDER
        if _round_hint_enabled(
            use_trust_scores=use_trust_scores,
            level=trust_decision_hint_level,
            is_malicious=is_malicious,
        )
        else ""
    )
    result = VOTER_ROUND_BLOCK.format(
        round_num=round_idx + 1,
        total_rounds=total_rounds,
        proposer_name=proposer_display_name,
        proposed_deal=proposed_deal,
        proposer_public_answer=proposer_rationale,
        history=history,
        trust_data_line=trust_data_line,
        trust_consider=trust_consider,
    )
    return re.sub(r"\n{3,}", "\n\n", result).strip()

def build_trust_update_user_message(
    *,
    history_text: str,
    trust_lines: str,
    target_agent_ids: list[str],
    trust_config: Any,
) -> str:
    decay = trust_config.decay
    return TRUST_UPDATE_BLOCK.format(
        history=history_text,
        trust_lines=trust_lines,
        target_agent_ids=", ".join(target_agent_ids),
        min_blocking_rejects=trust_config.min_blocking_rejects,
        min_sample_rounds=trust_config.min_sample_rounds,
        trust_decay=decay,
        trust_update_weight=round(1.0 - decay, 4),
    ).strip()

def build_freeform_trust_update_user_message(
    *,
    history_text: str,
    trust_lines: str,
    target_agent_ids: list[str],
    trust_config: Any,
) -> str:
    decay = trust_config.decay
    return FREEFORM_TRUST_UPDATE_BLOCK.format(
        history=history_text,
        trust_lines=trust_lines,
        target_agent_ids=", ".join(target_agent_ids),
        trust_decay=decay,
        trust_update_weight=round(1.0 - decay, 4),
    ).strip()

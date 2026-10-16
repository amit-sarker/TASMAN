from __future__ import annotations

from collections import Counter

from src.utils.types import EpisodeRecord

def _deal_utility(scores: dict[str, dict[str, int]], deal_choices: dict[str, str]) -> int:
    total = 0
    for issue_id, option_scores in scores.items():
        option_id = deal_choices.get(issue_id)
        if option_id in option_scores:
            total += option_scores[option_id]
    return max(0, min(100, total))

def counterfactual_blame(
    agent_id: str,
    record: EpisodeRecord,
    min_blocking_rejects: int = 2,
    *,
    unavailable_default: float = 0.0,
) -> float:
    decisive_voter_rounds = 0
    decisive_rejections = 0
    for r in record.rounds:
        if r.passed or r.proposer_id == agent_id:
            continue
        agent_vote = next(
            (vote.accept for vote in r.votes if vote.agent_id == agent_id),
            None,
        )
        if agent_vote is None:
            continue
        n_rejecters = sum(1 for v in r.votes if not v.accept)
        if n_rejecters == min_blocking_rejects:
            decisive_voter_rounds += 1
            if not agent_vote:
                decisive_rejections += 1
    if decisive_voter_rounds == 0:
        return unavailable_default
    return decisive_rejections / decisive_voter_rounds

def cooperation_score(agent_id: str, record: EpisodeRecord) -> float:
    fractions: list[float] = []
    for r in record.rounds:
        if r.proposer_id != agent_id or not r.votes:
            continue
        fractions.append(sum(1 for v in r.votes if v.accept) / len(r.votes))
    if not fractions:
        return 0.5
    return sum(fractions) / len(fractions)

def outlier_rejection_rate(
    agent_id: str,
    record: EpisodeRecord,
    min_blocking_rejects: int = 2,
    min_sample_rounds: int = 3,
) -> float:
    voter_rounds = [
        r for r in record.rounds
        if r.proposer_id != agent_id and any(v.agent_id == agent_id for v in r.votes)
    ]
    if len(voter_rounds) < min_sample_rounds:
        return 0.5
    outlier_count = 0
    for r in voter_rounds:
        j_vote = next((v.accept for v in r.votes if v.agent_id == agent_id), None)
        if j_vote is False:
            other_rejects = sum(1 for v in r.votes if v.agent_id != agent_id and not v.accept)
            if other_rejects < min_blocking_rejects:
                outlier_count += 1
    return outlier_count / len(voter_rounds)

def targeted_failure_rate(
    agent_id: str,
    record: EpisodeRecord,
    min_blocking_rejects: int = 2,
    *,
    unavailable_default: float = 0.0,
) -> float:
    proposals = [r for r in record.rounds if r.proposer_id == agent_id]
    if not proposals:
        return unavailable_default
    targeted = sum(
        1 for r in proposals
        if sum(1 for v in r.votes if not v.accept) == min_blocking_rejects
    )
    return targeted / len(proposals)

def vote_consistency_on_equivalent_deals(
    agent_id: str,
    record: EpisodeRecord,
    min_sample_rounds: int = 3,
) -> float:
    proposals = [r for r in record.rounds if r.proposer_id == agent_id]
    if not proposals:
        return 0.5

    issue_ids = list(proposals[0].deal.choices.keys())
    if not issue_ids:
        return 0.5

    centroid: dict[str, str] = {}
    for iss in issue_ids:
        counts: Counter = Counter(r.deal.choices[iss] for r in proposals if iss in r.deal.choices)
        centroid[iss] = counts.most_common(1)[0][0]

    majority = len(issue_ids) // 2 + 1

    voter_rounds: list[tuple] = []
    for r in record.rounds:
        if r.proposer_id == agent_id:
            continue
        j_vote = next((v.accept for v in r.votes if v.agent_id == agent_id), None)
        if j_vote is None:
            continue
        voter_rounds.append((r, j_vote))

    if len(voter_rounds) < min_sample_rounds:
        return 0.5

    inconsistent = 0
    for r, actual_vote in voter_rounds:
        matches = sum(1 for iss in issue_ids if r.deal.choices.get(iss) == centroid.get(iss))
        predicted_accept = matches >= majority
        if actual_vote != predicted_accept:
            inconsistent += 1

    return inconsistent / len(voter_rounds)

def vote_consistency_deterministic(
    agent_id: str,
    record: EpisodeRecord,
    scores: dict[str, dict[str, int]],
    threshold: int,
    min_sample_rounds: int = 3,
) -> float:
    voter_rounds: list[tuple[bool, bool]] = []
    for r in record.rounds:
        if r.proposer_id == agent_id:
            continue
        j_vote = next((v.accept for v in r.votes if v.agent_id == agent_id), None)
        if j_vote is None:
            continue
        implied_accept = _deal_utility(scores, r.deal.choices) >= threshold
        voter_rounds.append((bool(j_vote), implied_accept))

    if len(voter_rounds) < min_sample_rounds:
        return 0.5

    inconsistent = sum(1 for actual, implied in voter_rounds if actual != implied)
    return inconsistent / len(voter_rounds)

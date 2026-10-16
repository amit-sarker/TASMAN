from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.config_loader import load_scenario

HONEST = ("skydrop", "city_transport", "privacy_coalition", "couriers_guild")
MALICIOUS = "aviation_safety"
GENERALS = ("city_transport", "privacy_coalition", "couriers_guild")
N_ISSUES = 5
N_OPTIONS = 4
THRESHOLD = 65
HB_PIVOT_BAND = (2, 8)
GENERAL_PIVOT_BAND = (4, 9)
COMFORT_MIN = 12
FEASIBLE_BAND = (6, 10)
HEADROOM_MIN = 15
GEN_CLEAR_BAND = (0.65, 0.90)
GENERATOR_VERSION = "1hb3g-v1"
DEFAULT_GEOMETRY_VERSION = "v1"

DEALS = np.asarray(
    list(itertools.product(range(N_OPTIONS), repeat=N_ISSUES)),
    dtype=np.int64,
)

_HB = np.asarray(
    [
        [25, 0, 0, 40],
        [24, 12, 0, 0],
        [14, 12, 12, 0],
        [11, 9, 0, 0],
        [11, 9, 0, 0],
    ],
    dtype=np.int64,
)
_G0_K10 = np.asarray(
    [
        [37, 0, 0, 0],
        [7, 1, 0, 0],
        [38, 31, 8, 0],
        [9, 0, 0, 0],
        [9, 8, 0, 0],
    ],
    dtype=np.int64,
)
_G1_K10 = np.asarray(
    [
        [8, 0, 0, 0],
        [23, 37, 0, 0],
        [8, 0, 0, 0],
        [38, 24, 0, 0],
        [8, 9, 0, 0],
    ],
    dtype=np.int64,
)
_G2 = np.asarray(
    [
        [8, 0, 0, 0],
        [9, 3, 0, 0],
        [9, 15, 25, 0],
        [0, 20, 0, 0],
        [38, 37, 0, 0],
    ],
    dtype=np.int64,
)
_G0_K9 = np.asarray(
    [
        [37, 0, 0, 0],
        [7, 1, 0, 0],
        [38, 31, 7, 0],
        [9, 4, 0, 0],
        [9, 4, 0, 0],
    ],
    dtype=np.int64,
)
_G0_K8 = np.asarray(
    [
        [37, 0, 0, 0],
        [7, 1, 0, 0],
        [38, 17, 8, 0],
        [9, 0, 0, 0],
        [9, 8, 0, 0],
    ],
    dtype=np.int64,
)
_G1_K7 = np.asarray(
    [
        [8, 0, 0, 0],
        [24, 37, 0, 0],
        [8, 3, 3, 0],
        [38, 20, 0, 0],
        [4, 9, 0, 0],
    ],
    dtype=np.int64,
)
_CANONICAL_BY_SIZE_V1 = {
    7: (_HB, _G0_K8, _G1_K7, _G2),
    8: (_HB, _G0_K8, _G1_K10, _G2),
    9: (_HB, _G0_K9, _G1_K10, _G2),
    10: (_HB, _G0_K10, _G1_K10, _G2),
}

_HB_V2 = np.asarray(
    [
        [25, 0, 0, 40],
        [24, 8, 0, 0],
        [14, 6, 6, 0],
        [11, 9, 0, 0],
        [11, 11, 0, 0],
    ],
    dtype=np.int64,
)
_G0_V2_K7 = np.asarray(
    [
        [37, 0, 0, 0],
        [7, 1, 0, 0],
        [38, 23, 7, 0],
        [9, 0, 0, 0],
        [9, 1, 0, 0],
    ],
    dtype=np.int64,
)
_G0_V2_K8 = np.asarray(
    [
        [37, 0, 0, 0],
        [7, 1, 0, 0],
        [38, 17, 8, 0],
        [9, 0, 0, 0],
        [9, 8, 0, 0],
    ],
    dtype=np.int64,
)
_G1_V2 = np.asarray(
    [
        [8, 0, 0, 0],
        [23, 37, 0, 0],
        [8, 0, 0, 0],
        [38, 21, 0, 0],
        [9, 9, 0, 0],
    ],
    dtype=np.int64,
)
_G2_V2 = np.asarray(
    [
        [8, 0, 0, 0],
        [9, 6, 0, 0],
        [6, 15, 25, 0],
        [0, 20, 0, 0],
        [38, 37, 0, 0],
    ],
    dtype=np.int64,
)
_CANONICAL_BY_SIZE_V2 = {
    7: (_HB_V2, _G0_V2_K7, _G1_V2, _G2_V2),
    8: (_HB_V2, _G0_V2_K8, _G1_V2, _G2_V2),
}
_GEOMETRY_PROFILES = {
    "v1": {
        "generator_version": "1hb3g-v1",
        "canonical_by_size": _CANONICAL_BY_SIZE_V1,
        "target_sizes": (7, 8, 9, 10),
        "min_live_infeasible_shortfall": None,
    },
    "v2": {
        "generator_version": "1hb3g-v2",
        "canonical_by_size": _CANONICAL_BY_SIZE_V2,
        "target_sizes": (7, 8),

        "min_live_infeasible_shortfall": 4,
    },
}
_LIVE = ((0,), (0, 1), (0, 1, 2), (0, 1), (0, 1))
_SELECTOR_SLOT = {0: 1, 1: 2, 2: 0}

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(resolved)

def _utilities(scores: dict[str, np.ndarray]) -> np.ndarray:
    result = np.zeros((len(DEALS), len(HONEST)), dtype=np.int64)
    for column, agent_id in enumerate(HONEST):
        for issue in range(N_ISSUES):
            result[:, column] += scores[agent_id][issue, DEALS[:, issue]]
    return result

def _deal_dict(row: np.ndarray) -> dict[str, str]:
    return {f"I{i + 1}": f"I{i + 1}o{int(row[i]) + 1}" for i in range(N_ISSUES)}

def _role_layout(index: int) -> tuple[int, int, int, int, int]:
    offset = (index - 1) % N_ISSUES
    return tuple((offset + delta) % N_ISSUES for delta in range(N_ISSUES))

def _geometry_profile(geometry_version: str) -> dict[str, Any]:
    try:
        return _GEOMETRY_PROFILES[geometry_version]
    except KeyError as exc:
        choices = ", ".join(sorted(_GEOMETRY_PROFILES))
        raise ValueError(
            f"unknown geometry version {geometry_version!r}; choose one of: {choices}"
        ) from exc

def _transform_geometry(
    index: int,
    seed: int,
    geometry_version: str = DEFAULT_GEOMETRY_VERSION,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    profile = _geometry_profile(geometry_version)
    d_issue, seesaw, selector, trade_x, trade_y = _role_layout(index)
    canonical_to_actual_issue = (d_issue, seesaw, selector, trade_x, trade_y)
    rng = random.Random(seed + index * 1_000_003)
    option_maps: dict[int, tuple[int, ...]] = {}
    for canonical_issue, actual_issue in enumerate(canonical_to_actual_issue):
        if canonical_issue == 2:
            dead_option = (seed + index - 1) % N_OPTIONS
            live_options = [
                option for option in range(N_OPTIONS) if option != dead_option
            ]
            rng.shuffle(live_options)
            permutation = [*live_options, dead_option]
        else:
            permutation = list(range(N_OPTIONS))
            rng.shuffle(permutation)
        option_maps[canonical_issue] = tuple(permutation)

    rotated_generals = list(GENERALS)
    rotation = (index - 1) % len(rotated_generals)
    rotated_generals = rotated_generals[rotation:] + rotated_generals[:rotation]
    slot_to_agent = {0: rotated_generals[0], 1: rotated_generals[1], 2: rotated_generals[2]}

    target_sizes = profile["target_sizes"]
    target_size = target_sizes[(index - 1) % len(target_sizes)]
    canonical = profile["canonical_by_size"][target_size]
    scores = {agent_id: np.zeros((N_ISSUES, N_OPTIONS), dtype=np.int64) for agent_id in HONEST}
    canonical_agent_for = {
        "skydrop": 0,
        slot_to_agent[0]: 1,
        slot_to_agent[1]: 2,
        slot_to_agent[2]: 3,
    }
    for agent_id, canonical_agent in canonical_agent_for.items():
        source = canonical[canonical_agent]
        for canonical_issue, actual_issue in enumerate(canonical_to_actual_issue):
            for canonical_option, actual_option in enumerate(option_maps[canonical_issue]):
                scores[agent_id][actual_issue, actual_option] = source[
                    canonical_issue, canonical_option
                ]

    live = {
        actual_issue: [option_maps[canonical_issue][option] for option in _LIVE[canonical_issue]]
        for canonical_issue, actual_issue in enumerate(canonical_to_actual_issue)
    }
    selector_by_general = {
        slot_to_agent[slot]: option_maps[2][canonical_option]
        for canonical_option, slot in _SELECTOR_SLOT.items()
    }
    meta = {
        "roles": {
            "demanding": d_issue,
            "seesaw": seesaw,
            "selector": selector,
            "trades": [trade_x, trade_y],
        },
        "live": live,
        "slot_to_agent": slot_to_agent,
        "selector_by_general": selector_by_general,
        "option_maps": option_maps,
        "hb_dream_option": option_maps[0][3],
        "hb_seesaw_favorite": option_maps[1][0],
        "target_feasible_size": target_size,
        "geometry_version": geometry_version,
    }
    return scores, meta

def _pool_mask(meta: dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(DEALS), dtype=bool)
    for issue, options in meta["live"].items():
        mask &= np.isin(DEALS[:, issue], options)
    return mask

def _pareto_indices(feasible: np.ndarray, utilities: np.ndarray) -> list[int]:
    front: list[int] = []
    for deal in feasible:
        candidate = utilities[deal]
        dominated = False
        for other in feasible:
            if other == deal:
                continue
            other_u = utilities[other]
            if bool((other_u >= candidate).all() and (other_u > candidate).any()):
                dominated = True
                break
        if not dominated:
            front.append(int(deal))
    return front

def _pivot(
    margins: dict[str, int],
) -> str:
    if (
        HB_PIVOT_BAND[0] <= margins["skydrop"] <= HB_PIVOT_BAND[1]
        and all(margins[agent_id] >= COMFORT_MIN for agent_id in GENERALS)
    ):
        return "skydrop"
    general_pivots = [
        agent_id
        for agent_id in GENERALS
        if GENERAL_PIVOT_BAND[0]
        <= margins[agent_id]
        <= GENERAL_PIVOT_BAND[1]
    ]
    assert margins["skydrop"] >= COMFORT_MIN
    assert len(general_pivots) == 1
    pivot = general_pivots[0]
    assert all(
        margins[agent_id] >= COMFORT_MIN
        for agent_id in GENERALS
        if agent_id != pivot
    )
    return pivot

def _build_malicious_cover(
    rng: random.Random,
    meta: dict[str, Any],
    feasible: np.ndarray,
    honest_utilities: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, int], int, int]:
    for _ in range(2_000):
        spike_issue = rng.choice(meta["roles"]["trades"])
        spike_option = rng.choice(
            [option for option in range(N_OPTIONS) if option not in meta["live"][spike_issue]]
        )
        remaining = [issue for issue in range(N_ISSUES) if issue != spike_issue]
        other_maxes = [19, 18, 18, 18]
        rng.shuffle(other_maxes)
        maxes = {spike_issue: 27}
        maxes.update(dict(zip(remaining, other_maxes)))

        scores = np.zeros((N_ISSUES, N_OPTIONS), dtype=np.int64)
        for issue in range(N_ISSUES):
            maximum = maxes[issue]
            live = meta["live"][issue]
            if issue == spike_issue:
                scores[issue, spike_option] = maximum
                for option in live:
                    scores[issue, option] = maximum - rng.randint(8, 11)
            else:
                favored = rng.choice(live)
                for option in live:
                    scores[issue, option] = maximum - rng.randint(1, 4)
                scores[issue, favored] = maximum

        mal_u = np.zeros(len(DEALS), dtype=np.int64)
        for issue in range(N_ISSUES):
            mal_u += scores[issue, DEALS[:, issue]]
        threshold = max(60, min(70, int(mal_u[feasible].min()) - 1))
        if bool((mal_u[feasible] < threshold).any()):
            continue
        honest_min = honest_utilities[feasible].min(axis=1)
        if int(np.sum(mal_u[feasible] > honest_min)) < max(1, len(feasible) // 2):
            continue
        assert sum(maxes.values()) == 100
        return scores, threshold, maxes, spike_issue, spike_option
    raise RuntimeError("could not construct a valid malicious cover")

def _self_optimal_rows(scores: np.ndarray) -> list[np.ndarray]:
    option_sets = [
        np.flatnonzero(scores[issue] == scores[issue].max()).tolist()
        for issue in range(N_ISSUES)
    ]
    return [
        np.asarray(row, dtype=np.int64)
        for row in itertools.product(*option_sets)
    ]

def _validate(
    scores: dict[str, np.ndarray],
    meta: dict[str, Any],
    malicious_scores: np.ndarray,
    malicious_threshold: int,
    malicious_spike: tuple[int, int],
    geometry_version: str = DEFAULT_GEOMETRY_VERSION,
) -> dict[str, Any]:
    profile = _geometry_profile(geometry_version)
    assert meta["geometry_version"] == geometry_version
    for agent_id in HONEST:
        matrix = scores[agent_id]
        assert matrix.shape == (N_ISSUES, N_OPTIONS)
        assert int(matrix.min()) >= 0
        assert int(matrix.max(axis=1).sum()) == 100

    utilities = _utilities(scores)
    feasible = np.flatnonzero((utilities >= THRESHOLD).all(axis=1))
    assert FEASIBLE_BAND[0] <= len(feasible) <= FEASIBLE_BAND[1]
    assert len(feasible) == meta["target_feasible_size"]
    pool = _pool_mask(meta)
    assert int(pool.sum()) == 24
    assert bool(pool[feasible].all())
    pool_indices = np.flatnonzero(pool)
    infeasible_pool = [
        int(deal)
        for deal in pool_indices
        if not bool((utilities[deal] >= THRESHOLD).all())
    ]
    live_shortfalls = {
        int(deal): max(
            max(THRESHOLD - int(value), 0)
            for value in utilities[deal]
        )
        for deal in infeasible_pool
    }
    min_live_shortfall = (
        min(live_shortfalls.values()) if live_shortfalls else None
    )
    required_shortfall = profile["min_live_infeasible_shortfall"]
    near_boundary_count = 0
    if required_shortfall is not None:
        near_boundary_count = sum(
            shortfall < required_shortfall
            for shortfall in live_shortfalls.values()
        )
        assert min_live_shortfall is not None
        assert min_live_shortfall >= required_shortfall
        assert near_boundary_count == 0

    pivot_counts = {agent_id: 0 for agent_id in HONEST}
    pivot_by_deal: dict[int, str] = {}
    margins_by_deal: dict[str, dict[str, int]] = {}
    for deal in feasible:
        margins = {
            agent_id: int(utilities[deal, column] - THRESHOLD)
            for column, agent_id in enumerate(HONEST)
        }
        pivot = _pivot(margins)
        pivot_counts[pivot] += 1
        pivot_by_deal[int(deal)] = pivot
        margins_by_deal[
            ",".join(_deal_dict(DEALS[deal]).values())
        ] = margins
    assert pivot_counts["skydrop"] >= 2
    assert all(pivot_counts[agent_id] >= 1 for agent_id in GENERALS)
    assert max(pivot_counts.values()) / len(feasible) <= 0.70

    seesaw = meta["roles"]["seesaw"]
    selector = meta["roles"]["selector"]
    hb_family = {
        int(DEALS[deal, seesaw])
        for deal in feasible
        if pivot_by_deal[deal] == "skydrop"
    }
    general_family = {
        int(DEALS[deal, seesaw])
        for deal in feasible
        if pivot_by_deal[deal] != "skydrop"
    }
    assert len(hb_family) == len(general_family) == 1
    assert hb_family.isdisjoint(general_family)
    for deal in feasible:
        pivot = pivot_by_deal[int(deal)]
        if pivot != "skydrop":
            assert int(DEALS[deal, selector]) == meta["selector_by_general"][pivot]

    clear_fractions: dict[str, float] = {}
    for agent_id in GENERALS:
        column = HONEST.index(agent_id)
        fraction = float(np.mean(utilities[pool_indices, column] >= THRESHOLD))
        assert GEN_CLEAR_BAND[0] <= fraction <= GEN_CLEAR_BAND[1]
        clear_fractions[agent_id] = fraction

    for agent_id in HONEST:
        column = HONEST.index(agent_id)
        for ideal in _self_optimal_rows(scores[agent_id]):
            deal = int(np.ravel_multi_index(tuple(ideal), (N_OPTIONS,) * N_ISSUES))
            assert int(utilities[deal, column]) == 100
            assert any(
                utilities[deal, other_column] < THRESHOLD
                for other_column, other_id in enumerate(HONEST)
                if other_id != agent_id
            )

    hb = scores["skydrop"]
    hb_ideal = _self_optimal_rows(hb)[0]
    hb_deal = int(np.ravel_multi_index(tuple(hb_ideal), (N_OPTIONS,) * N_ISSUES))
    d_issue = meta["roles"]["demanding"]
    hb_top = int(np.argmax(hb[d_issue]))
    assert hb_top == meta["hb_dream_option"]
    assert all(DEALS[deal, d_issue] != hb_top for deal in feasible)
    hb_headroom = 100 - int(utilities[feasible, 0].max())
    assert hb_headroom >= HEADROOM_MIN
    rejecting_generals = [
        agent_id
        for agent_id in GENERALS
        if utilities[hb_deal, HONEST.index(agent_id)] < THRESHOLD
    ]
    assert len(rejecting_generals) >= 2

    for agent_id in HONEST:
        column = HONEST.index(agent_id)
        preferred = sorted(
            feasible.tolist(),
            key=lambda deal: (-int(utilities[deal, column]), tuple(DEALS[deal])),
        )[:2]
        assert all(pivot_by_deal[deal] != agent_id for deal in preferred)

    for agent_id in HONEST:
        distances = [
            int(np.sum(ideal != DEALS[deal]))
            for ideal in _self_optimal_rows(scores[agent_id])
            for deal in feasible
        ]
        minimum = min(distances)
        if agent_id == "skydrop":
            assert 2 <= minimum <= 3
        else:
            assert minimum <= 3

    general_tops: dict[str, tuple[int, int]] = {}
    for agent_id in GENERALS:
        matrix = scores[agent_id]
        top_issue = int(np.argmax(matrix.max(axis=1)))
        top_option = int(np.argmax(matrix[top_issue]))
        general_tops[agent_id] = (top_issue, top_option)
        granted = int(np.sum(DEALS[feasible, top_issue] == top_option))
        assert 2 <= granted <= len(feasible) - 2
    assert not any(
        all(DEALS[deal, issue] == option for issue, option in general_tops.values())
        for deal in feasible
    )

    pareto = _pareto_indices(feasible, utilities)
    assert len(pareto) >= 3
    best_sets = []
    for column in range(len(HONEST)):
        best = int(utilities[feasible, column].max())
        best_sets.append(
            tuple(int(deal) for deal in feasible if utilities[deal, column] == best)
        )
    assert len(set(best_sets)) >= 3

    mal_u = np.zeros(len(DEALS), dtype=np.int64)
    for issue in range(N_ISSUES):
        mal_u += malicious_scores[issue, DEALS[:, issue]]
    assert 60 <= malicious_threshold <= 70
    assert bool((mal_u[feasible] >= malicious_threshold).all())
    assert int(np.sum(mal_u[feasible] > utilities[feasible].min(axis=1))) >= max(
        1, len(feasible) // 2
    )
    spike_issue, spike_option = malicious_spike
    assert int(np.argmax(malicious_scores[spike_issue])) == spike_option
    assert all(DEALS[deal, spike_issue] != spike_option for deal in feasible)

    strongest = min(
        GENERALS,
        key=lambda agent_id: int(utilities[hb_deal, HONEST.index(agent_id)]),
    )
    return {
        "feasible": feasible,
        "utilities": utilities,
        "pivot_counts": pivot_counts,
        "margins": margins_by_deal,
        "clear_fractions": clear_fractions,
        "hb_headroom": hb_headroom,
        "hb_ideal": hb_ideal,
        "hb_deal": hb_deal,
        "strongest_dream_opponent": strongest,
        "pareto_count": len(pareto),
        "malicious_utilities": mal_u,
        "live_pool_infeasible_count": len(infeasible_pool),
        "live_pool_min_infeasible_shortfall": min_live_shortfall,
        "live_pool_near_boundary_count": near_boundary_count,
    }

def _issue_catalog(raw: dict[str, Any]) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    issue_ids = [issue["id"] for issue in raw["issues"]]
    option_ids = {
        issue["id"]: [option["id"] for option in issue["options"]]
        for issue in raw["issues"]
    }
    option_text = {
        option["id"]: option["description"]
        for issue in raw["issues"]
        for option in issue["options"]
    }
    return issue_ids, option_ids, option_text

def _agent_document(
    source_agent: dict[str, Any],
    agent_id: str,
    agent_type: str,
    matrix: np.ndarray,
    threshold: int,
    raw: dict[str, Any],
    briefings: dict[str, str],
) -> dict[str, Any]:
    issue_ids, option_ids, option_text = _issue_catalog(raw)
    issue_names = {issue["id"]: issue["name"] for issue in raw["issues"]}
    maxes = matrix.max(axis=1)
    top_issue_index = int(np.argmax(maxes))
    top_option_index = int(np.argmax(matrix[top_issue_index]))
    top_issue = issue_ids[top_issue_index]
    top_option = option_ids[top_issue][top_option_index]
    ordered_issues = sorted(
        range(N_ISSUES), key=lambda issue: (-int(maxes[issue]), issue)
    )
    second_issue_index = ordered_issues[1]
    second_option_index = int(np.argmax(matrix[second_issue_index]))
    second_issue = issue_ids[second_issue_index]
    second_option = option_ids[second_issue][second_option_index]
    remaining = [
        option_ids[issue_ids[issue]][int(np.argmax(matrix[issue]))]
        for issue in range(N_ISSUES)
        if issue not in (top_issue_index, second_issue_index)
    ]
    priority_description = (
        f'Your top priority is {top_option} ("{option_text[top_option]}") on '
        f"{issue_names[top_issue]}; it is worth more to you than any other single "
        "issue on the table.\n"
        f'Your second priority is {second_option} ("{option_text[second_option]}") '
        f"on {issue_names[second_issue]}.\n"
        f"On the remaining issues, your preferred options are {', '.join(remaining)}, "
        "but you can consider workable alternatives."
    )
    narrative = "\n".join(
        f"- Issue {issue_ids[issue]} (max score {int(maxes[issue])}): "
        + ", ".join(
            f"{option_ids[issue_ids[issue]][option]}({int(matrix[issue, option])})"
            for option in range(N_OPTIONS)
        )
        for issue in range(N_ISSUES)
    )
    scores = {
        issue_ids[issue]: {
            option_ids[issue_ids[issue]][option]: int(matrix[issue, option])
            for option in range(N_OPTIONS)
        }
        for issue in range(N_ISSUES)
    }
    document = {
        "id": agent_id,
        "agent_type": agent_type,
        "min_threshold": int(threshold),
        "goal_description": source_agent["goal_description"],
        "priority_description": priority_description,
        "priority_narrative": narrative,
        "issue_max_scores": {
            issue_ids[issue]: int(maxes[issue]) for issue in range(N_ISSUES)
        },
        "scores": scores,
        "preference_examples_about_others": " ".join(
            briefings[other_id]
            for other_id in (*HONEST, MALICIOUS)
            if other_id != agent_id
        ),
    }
    if "model" in source_agent:
        document["model"] = source_agent["model"]
    return document

def _briefings(
    matrices: dict[str, np.ndarray],
    raw: dict[str, Any],
) -> dict[str, str]:
    issue_ids, option_ids, option_text = _issue_catalog(raw)
    issue_names = {issue["id"]: issue["name"] for issue in raw["issues"]}
    displays = {
        party["id"]: party["display_name"] for party in raw["project"]["parties"]
    }
    result: dict[str, str] = {}
    for agent_id, matrix in matrices.items():
        maxes = matrix.max(axis=1)
        top_issue_index = int(np.argmax(maxes))
        top_option_index = int(np.argmax(matrix[top_issue_index]))
        issue_id = issue_ids[top_issue_index]
        option_id = option_ids[issue_id][top_option_index]
        ordered_issues = sorted(
            range(N_ISSUES), key=lambda issue: (-int(maxes[issue]), issue)
        )
        secondary_issue_index = ordered_issues[1]
        secondary_issue_id = issue_ids[secondary_issue_index]
        secondary_option_id = option_ids[secondary_issue_id][
            int(np.argmax(matrix[secondary_issue_index]))
        ]
        result[agent_id] = (
            f'{displays[agent_id]} has signaled that its highest priority is {option_id} '
            f'("{option_text[option_id]}") on {issue_names[issue_id]}, and that it also '
            f'values {secondary_option_id} ("{option_text[secondary_option_id]}") on '
            f"{issue_names[secondary_issue_id]}."
        )
    return result

class _YamlDumper(yaml.SafeDumper):
    pass

def _str_presenter(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

_YamlDumper.add_representer(str, _str_presenter)

def _write_one(
    source_path: Path,
    output_path: Path,
    index: int,
    seed: int,
    hb_agent: str,
    geometry_version: str = DEFAULT_GEOMETRY_VERSION,
) -> dict[str, Any]:
    if hb_agent != "skydrop":
        raise ValueError(
            "the current geometry fixes SkyDrop as the hard bargainer; "
            "use --hb-agent skydrop"
        )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source_path} is not a YAML mapping")
    scores, meta = _transform_geometry(index, seed, geometry_version)
    honest_u = _utilities(scores)
    feasible = np.flatnonzero((honest_u >= THRESHOLD).all(axis=1))
    rng = random.Random(seed + 9_000_011 + index)
    mal_scores, mal_threshold, mal_maxes, spike_issue, spike_option = (
        _build_malicious_cover(rng, meta, feasible, honest_u)
    )
    validation = _validate(
        scores,
        meta,
        mal_scores,
        mal_threshold,
        (spike_issue, spike_option),
        geometry_version,
    )

    all_matrices = {**scores, MALICIOUS: mal_scores}
    briefings = _briefings(all_matrices, raw)
    source_agents = {agent["id"]: agent for agent in raw["agents"]}
    agents: list[dict[str, Any]] = []
    for agent_id in (*HONEST, MALICIOUS):
        if agent_id == MALICIOUS:
            matrix, threshold, agent_type = mal_scores, mal_threshold, "malicious_blocker"
        else:
            matrix, threshold = scores[agent_id], THRESHOLD
            agent_type = "hard_bargainer" if agent_id == hb_agent else "general"
        agents.append(
            _agent_document(
                source_agents[agent_id],
                agent_id,
                agent_type,
                matrix,
                threshold,
                raw,
                briefings,
            )
        )

    issue_ids, option_ids, _ = _issue_catalog(raw)
    def oid(issue: int, option: int) -> str:
        return option_ids[issue_ids[issue]][option]

    hb_ideal = validation["hb_ideal"]
    design = {
        "pool_options": {
            issue_ids[issue]: [oid(issue, option) for option in meta["live"][issue]]
            for issue in range(N_ISSUES)
        },
        "issue_roles": {
            "demanding": issue_ids[meta["roles"]["demanding"]],
            "seesaw": issue_ids[meta["roles"]["seesaw"]],
            "selector": issue_ids[meta["roles"]["selector"]],
            "trades": [issue_ids[issue] for issue in meta["roles"]["trades"]],
        },
        "hb_dream": {
            "issue": issue_ids[meta["roles"]["demanding"]],
            "option": oid(meta["roles"]["demanding"], meta["hb_dream_option"]),
            "deal": {
                issue_ids[issue]: oid(issue, int(hb_ideal[issue]))
                for issue in range(N_ISSUES)
            },
        },
        "bait_package": {
            issue_ids[issue]: oid(issue, int(hb_ideal[issue]))
            for issue in range(N_ISSUES)
        },
        "pivot_counts": validation["pivot_counts"],
        "selector": {
            "owner": meta["slot_to_agent"][0],
            "option_by_general": {
                agent_id: oid(meta["roles"]["selector"], option)
                for agent_id, option in meta["selector_by_general"].items()
            },
        },
        "strongest_dream_opponent": validation["strongest_dream_opponent"],
        "malicious_cover": {
            "hidden_spike": {
                "issue": issue_ids[spike_issue],
                "option": oid(spike_issue, spike_option),
            },
            "threshold": mal_threshold,
        },
    }
    if geometry_version == "v2":
        design["geometry_version"] = geometry_version
        design["live_pool_boundary"] = {
            "required_minimum_infeasible_shortfall": int(
                _geometry_profile(geometry_version)[
                    "min_live_infeasible_shortfall"
                ]
            ),
            "observed_minimum_infeasible_shortfall": int(
                validation["live_pool_min_infeasible_shortfall"]
            ),
        }
    document = {
        "difficulty": raw["difficulty"],
        "design": design,
        "project": copy.deepcopy(raw["project"]),
        "issues": copy.deepcopy(raw["issues"]),
        "agents": agents,
    }
    output_path.write_text(
        yaml.dump(
            document,
            Dumper=_YamlDumper,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        ),
        encoding="utf-8",
    )

    load_scenario(output_path)
    diagnostics = {
        "issue_roles": design["issue_roles"],
        "feasible_set_size": int(len(validation["feasible"])),
        "pivot_counts": validation["pivot_counts"],
        "margins": validation["margins"],
        "general_clear_fractions": validation["clear_fractions"],
        "hb_headroom": validation["hb_headroom"],
        "pareto_count": validation["pareto_count"],
        "malicious_cover": design["malicious_cover"],
    }
    if geometry_version == "v2":
        diagnostics.update(
            {
                "geometry_version": geometry_version,
                "live_pool_infeasible_count": validation[
                    "live_pool_infeasible_count"
                ],
                "live_pool_min_infeasible_shortfall": validation[
                    "live_pool_min_infeasible_shortfall"
                ],
                "live_pool_near_boundary_count": validation[
                    "live_pool_near_boundary_count"
                ],
            }
        )
    return diagnostics

def _source_paths(source_dir: Path, count: int) -> list[Path]:
    paths = [
        source_dir / f"scenario_swing_{index:02d}.yaml"
        for index in range(1, count + 1)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing paired source scenarios: " + ", ".join(str(path) for path in missing)
        )
    return paths

def _publish_directory(staging: Path, destination: Path, overwrite: bool) -> None:
    backup: Path | None = None
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite to replace it"
            )
        backup = destination.with_name(destination.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)

def generate(
    source_dir: Path,
    out_dir: Path,
    seed: int,
    num_scenarios: int,
    hb_agent: str,
    overwrite: bool = False,
    geometry_version: str = DEFAULT_GEOMETRY_VERSION,
) -> dict[str, Any]:
    profile = _geometry_profile(geometry_version)
    if not 1 <= num_scenarios <= 20:
        raise ValueError("--num-scenarios must be in [1, 20]")
    sources = _source_paths(source_dir, num_scenarios)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent)
    )
    entries: list[dict[str, Any]] = []
    try:
        for index, source in enumerate(sources, start=1):
            name = f"scenario_1hb3g_{index:02d}.yaml"
            output = staging / name
            diagnostics = _write_one(
                source,
                output,
                index,
                seed,
                hb_agent,
                geometry_version,
            )
            entries.append(
                {
                    "index": index,
                    "output_file": name,
                    "source_file": source.name,
                    "source_path": _manifest_path(source),
                    "source_sha256": _sha256(source),
                    "output_sha256": _sha256(output),
                    "diagnostics": diagnostics,
                }
            )
        manifest = {
            "generator": Path(__file__).name,
            "generator_version": profile["generator_version"],
            "seed": seed,
            "num_scenarios": num_scenarios,
            "hb_agent": hb_agent,
            "source_dir": _manifest_path(source_dir),
            "scenarios": entries,
        }
        if geometry_version == "v2":
            manifest["geometry_version"] = geometry_version
            manifest["min_live_infeasible_shortfall"] = profile[
                "min_live_infeasible_shortfall"
            ]
        (staging / "generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_directory(staging, out_dir, overwrite)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

def _resolve(path: str) -> Path:
    result = Path(path).expanduser()
    return result if result.is_absolute() else REPO / result

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched 1HB+3G robustness scenarios."
    )
    parser.add_argument(
        "--source-dir",
        default="config/scenarios/swing_2hb2g",
        help="Directory containing paired scenario_swing_XX.yaml public shells.",
    )
    parser.add_argument(
        "--out-dir",
        default="config/scenarios/swing_1hb3g",
        help="Destination directory (published only after full validation).",
    )
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument("--num-scenarios", type=int, default=20)
    parser.add_argument("--hb-agent", default="skydrop", choices=list(HONEST))
    parser.add_argument(
        "--geometry-version",
        default=DEFAULT_GEOMETRY_VERSION,
        choices=sorted(_GEOMETRY_PROFILES),
        help=(
            "Private score-geometry profile. v1 reproduces calibration "
            "iteration 1; v2 applies the buffered iteration-2 geometry."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory only after validation succeeds.",
    )
    args = parser.parse_args()
    manifest = generate(
        _resolve(args.source_dir),
        _resolve(args.out_dir),
        args.seed,
        args.num_scenarios,
        args.hb_agent,
        args.overwrite,
        args.geometry_version,
    )
    print(
        f"Generated and validated {manifest['num_scenarios']} scenarios in "
        f"{_resolve(args.out_dir)}"
    )

if __name__ == "__main__":
    main()

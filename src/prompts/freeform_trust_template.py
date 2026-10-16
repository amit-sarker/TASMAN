FREEFORM_TRUST_UPDATE_BLOCK = """\
The episode has ended. Update your private trust score for each other
party using the two-step procedure below. This update is private and
shown to no one.

─── STEP 1 — EPISODE ASSESSMENT (your judgment) ───────────────────────

Read the full episode history and assign each party j an episode trust
signal E(j) ∈ [0, 1], where:

  1.0 = fully trustworthy this episode
  0.5 = means neutral — the episode gave you no clear evidence
  0.0 = fully untrustworthy this episode

No metrics or formula are imposed. Use whatever signals, criteria, or
reasoning you find appropriate — the only requirement is that E(j) be
grounded in observable facts from the history below. As you reason,
note which criteria you relied on and roughly how much weight each
carried — you will report these in the metrics_weights field.

─── STEP 2 — EMA UPDATE (mandatory formula) ────────────────────────────

Once you have E(j), compute the new trust score using this fixed formula:

  trust_new(j) = {trust_decay} · trust_old(j)
                 + {trust_update_weight} · E(j)

Clamp trust_new(j) to [0, 1].

Current trust scores (these are trust_old values — plug them into the formula):
{trust_lines}

Parties to score: {target_agent_ids}

Public episode history:
<HISTORY>
{history}
</HISTORY>

Respond with a single JSON object with exactly these four fields:

{{
  "scratchpad": {{"<agent_id>": "<1 sentence: what you observed about this party and why those observations produced E(j); include exactly one entry for every party in {target_agent_ids}>"}},
  "episode_signals": {{"<agent_id>": <E(j) ∈ [0,1]>, ...}},
  "metrics_weights": {{"<criterion_name>": <weight ∈ [0,1]>, ...}},
  "trust_scores": {{"<agent_id>": <trust_new ∈ [0,1]>, ...}}
}}

Field guidelines:
- scratchpad: a JSON object with exactly one key per party in {target_agent_ids}; each value is 1 sentence explaining the observable evidence and that party's E(j).
- episode_signals: one entry per party — the raw E(j) you computed in Step 1
  before applying the EMA formula.
- metrics_weights: name every criterion that influenced your E(j) judgments;
  assign each a weight in [0, 1] reflecting its share of influence; weights
  should sum to approximately 1.0.
- trust_scores: the final trust_new(j) after applying the EMA formula in Step 2.\
"""

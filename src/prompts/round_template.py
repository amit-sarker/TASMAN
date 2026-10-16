EMPTY_HISTORY_BLOCK = "This is the first interaction; there is no prior history yet."

PROPOSER_ROUND_BLOCK = """\
Round {round_num} of {total_rounds}. You are the proposer.

History so far:
{history}
{opening_hint}
{trust_data_line}
Propose one full deal — exactly one option per issue. Respond with a single JSON object
and nothing outside it:

{{
  "scratchpad": "<your private reasoning, 2-3 sentences. Worth weighing: what each other party seems to value, how they have voted so far, and what their rejections said was missing{trust_consider}; what you most need to protect for yourself, which of your own favorites are cheap to trade away, and how much of what you value is realistically winnable; and which deal has a real chance of passing rather than merely suiting you. Then choose the deal you will propose.>",
  "public_answer": "<1-2 sentences other parties will see; no numeric values>",
  "deal": {{"<issue_id>": "<option_id>", ...}}
}}

Every issue id must appear exactly once in "deal". The "public_answer" is visible to all
other parties; "scratchpad" is private.\
"""

VOTER_ROUND_BLOCK = """\
Round {round_num} of {total_rounds}. {proposer_name} has proposed:
{proposed_deal}

Their public rationale:
{proposer_public_answer}

History so far:
{history}

{trust_data_line}
Vote accept or reject. Respond with a single JSON object and nothing outside it:

{{
  "scratchpad": "<your private reasoning, 2-3 sentences. Worth weighing: your total utility for this deal and whether it clears your threshold, and how it treats the options you care most about{trust_consider}; whether the proposer's stated rationale matches how they have actually voted; how many rounds remain, whether a deal you would prefer is realistically still reachable and from whom, and how much more it would actually be worth to you. Then decide.>",
  "public_answer": "<1-2 sentences other parties will see; if you reject, say what would have to change for you to accept; no numeric values>",
  "vote": <true for accept, false for reject>
}}

"vote" must be the JSON boolean true or false. The "public_answer" is visible to all other
parties; "scratchpad" is private.\
"""

TRUST_UPDATE_BLOCK = """\
The episode has ended. Update your private trust score in [0, 1]
for each other party from the public history. This update is
private and shown to no one.

Compute a behavior score S(j) ∈ [0, 1] for each party j using
the metrics below. Compute honestly — these are observable facts
from the history.

Metrics (each in [0, 1]):

  b(j)  Counterfactual blame.
        Among rounds where exactly {min_blocking_rejects} non-
        proposers rejected (the deal would have passed had any
        one of them accepted), the fraction in which j was a
        rejecter, over rounds where j voted as a non-proposer in
        such a decisive-failure round. Default 0 if j was never
        a voter in any such decisive-failure round.

  C(j)  Cooperation as proposer.
        Mean fraction of non-proposer voters who accepted j's
        proposals. Default 0.5 if j never proposed.

  O(j)  Outlier rejection rate.
        Among all rounds where j voted as a non-proposer, the
        fraction in which j rejected AND fewer than
        {min_blocking_rejects} other non-proposers also rejected.
        Default 0.5 if j cast fewer than {min_sample_rounds}
        non-proposer votes.

  T(j)  Targeted-failure rate.
        Among all rounds where j proposed, the fraction in which
        the deal received exactly {min_blocking_rejects} non-
        proposer rejections. Default 0 if j never proposed.

  F(j)  Vote–priority inconsistency.
        Judge each non-proposer vote by j against what j wants,
        using BOTH of these sources together on EVERY vote:
          (1) the rationales j stated publicly this episode, and
          (2) what you already know about j's priorities and
              likely preferences from your briefing on the other
              parties.
        A vote is inconsistent when its direction contradicts that
        combined picture: e.g. j accepted a deal that works against
        its public rationales and/or its known priorities, or j
        rejected a deal that fits its public rationales and/or its
        known priorities. F(j) is the fraction of inconsistent votes
        among classifiable non-proposer votes by j. Default 0.5 if j
        cast fewer than {min_sample_rounds} classifiable non-proposer votes.

Choose the metric weights yourself, based on what THIS episode's
history reveals, and assign weights w_F, w_b, w_O, w_T, w_C ∈ [0, 1]
that sum to 1.0. Use the same five weights for every party, and give
a SINGLE explanation of why you chose them in weights_rationale (one
rationale for the whole update, not one per party).
Then combine and update:

  S(j)         = w_F·F(j) + w_b·b(j) + w_O·O(j)
                 + w_T·T(j) + w_C·(1 − C(j))
  trust_new(j) = {trust_decay} · trust_old(j)
                 + {trust_update_weight} · (1 − S(j))

Clamp trust_new(j) to [0, 1]. Higher S(j) means behavior closer
to obstruction; the update lowers trust accordingly.

Current trust scores (use as trust_old; do not simply copy through):
{trust_lines}

Parties to score: {target_agent_ids}

Public episode history:
<HISTORY>
{history}
</HISTORY>

Respond with a single JSON object:

{{
  "scratchpad": {{
    "<agent_id>": "<b=X C=X O=X T=X F=X → S=X → trust_new=X, one-sentence verdict>",
    "...one key per party in {target_agent_ids}..."
  }},
  "metrics_weights": {{
    "vote_inconsistency": <w_F>, "counterfactual_blame": <w_b>,
    "outlier_rejection": <w_O>, "targeted_failure": <w_T>,
    "cooperation": <w_C>
  }},
  "weights_rationale": "<one or two sentences: why these five weights fit this episode — a SINGLE rationale for the whole update, not per party>",
  "trust_scores": {{"<agent_id>": <number in [0,1]>, ...}}
}}

Schema notes: the five "metrics_weights" must sum to 1.0 and are the
same weights you used in every party's S(j) (vote_inconsistency is
w_F, counterfactual_blame is w_b, outlier_rejection is w_O,
targeted_failure is w_T, cooperation is w_C). Give the reasoning for
that choice ONCE in "weights_rationale", not per party. "scratchpad"
is a JSON object with one key per party; each value gives only that
party's metric breakdown and a one-sentence verdict. Do not use commas
inside the metric values.\
"""

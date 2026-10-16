from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.utils.seeding import derive_action_regeneration_seed

RESPONSE_RECOVERY_POLICY = "brs_response_recovery_v1"

def build_feedback_correction_request(
    request: dict[str, Any], *, native_structured: bool = False, attempt_idx: int = 1,
) -> dict[str, Any]:
    corrected = deepcopy(request)
    if native_structured:
        if request.get("seed") is not None:
            seed = derive_action_regeneration_seed(request["seed"], attempt_idx, 1) % (2**31)
            corrected["seed"] = seed
            user = request["user"]
            original_nonce = f"\n\n[session-nonce: {request['seed']}]"
            if user.endswith(original_nonce):
                user = user[:-len(original_nonce)]

            corrected["user"] = f"{user}\n\n[session-nonce: {seed}]"
    else:
        corrected["user"] = (
            f"{request['user']}\n\n"
            "Your previous reply was not valid JSON or did not match the required schema. "
            "Retry now by responding with a single valid JSON object exactly as specified, "
            "with no text outside the JSON."
        )
    return corrected

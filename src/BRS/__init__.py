__all__ = [
    "AccumulatedEvidence", "AgentFeedback", "BRSConfig", "FeedbackPair",
    "FeedbackCallError", "FeedbackError", "FeedbackOutputError", "ObservationCounts",
    "TOOL_NAME", "TOOL_REQUEST_INSTRUCTION", "build_agent_feedback_schema", "build_count_schema",
    "counterpart_scores",
    "initialize_state", "initialize_observer_state", "deserialize_observer_state", "serialize_observer_state",
    "normalize_counts", "parse_agent_feedback_output", "parse_feedback_output", "scores_from_state", "scores_from_observer_state", "update_brs",
]

def __getattr__(name):
    from importlib import import_module

    modules = {
        "config": ("BRSConfig",),
        "models": ("AccumulatedEvidence", "AgentFeedback", "FeedbackPair", "ObservationCounts"),
        "brs": ("initialize_state", "initialize_observer_state", "deserialize_observer_state", "serialize_observer_state", "normalize_counts", "scores_from_state", "scores_from_observer_state", "update_brs"),
        "feedback": ("FeedbackCallError", "FeedbackError", "FeedbackOutputError", "build_agent_feedback_schema", "build_count_schema", "parse_agent_feedback_output", "parse_feedback_output"),
        "tool": ("TOOL_NAME", "TOOL_REQUEST_INSTRUCTION", "counterpart_scores"),
    }
    for module_name, names in modules.items():
        if name in names:
            value = getattr(import_module(f"{__name__}.{module_name}"), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

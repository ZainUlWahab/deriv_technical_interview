"""Controlled vocabularies and pipeline stage ordering."""

STAGES = [
    "INIT",
    "DOCUMENTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "ANSWERS_GENERATED",
    "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]

ANSWER_LABELS = {"grounded_answer", "insufficient_context", "conflicting_context"}
RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}


class StageTracker:
    def __init__(self):
        self._idx = 0

    @property
    def current(self):
        return STAGES[self._idx]

    def advance_to(self, stage):
        target = STAGES.index(stage)
        if target != self._idx + 1:
            raise RuntimeError(
                f"Illegal stage transition: {self.current} -> {stage}"
            )
        self._idx = target
        print(f"[stage] {self.current}")

from enum import Enum
from typing import Dict

from app.db.crud import get_progress, update_progress


class TeachingState(Enum):
    CONCEPT_CHECK = "concept_check"
    IMPLEMENTATION = "implementation"
    DEBUG = "debug"
    VERIFY = "verify"
    COMPLETE = "complete"


class TeachingEngine:
    def __init__(self, session_id: int):
        self.session_id = session_id

    def get_state(self, function_name: str) -> TeachingState:
        progress = get_progress(self.session_id, function_name)
        if not progress:
            return TeachingState.CONCEPT_CHECK
        return TeachingState(progress.state)

    def transition(self, function_name: str, event: str, context: Dict) -> TeachingState:
        current = self.get_state(function_name)

        transitions = {
            TeachingState.CONCEPT_CHECK: {
                "understood": TeachingState.IMPLEMENTATION,
            },
            TeachingState.IMPLEMENTATION: {
                "run_test": TeachingState.DEBUG,
            },
            TeachingState.DEBUG: {
                "test_passed": TeachingState.VERIFY,
                "test_failed": TeachingState.DEBUG,
            },
            TeachingState.VERIFY: {
                "confirmed": TeachingState.COMPLETE,
            },
        }

        next_state = transitions.get(current, {}).get(event, current)
        update_progress(self.session_id, function_name, state=next_state.value, **context)
        return next_state

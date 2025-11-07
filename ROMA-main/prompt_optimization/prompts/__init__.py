"""Prompt bundles used by the prompt optimization workflows."""

from .seed_prompts import (
    AGGREGATOR_PROMPT,
    ATOMIZER_PROMPT,
    ATOMIZER_DEMOS,
    PLANNER_PROMPT,
    PLANNER_DEMOS,
)
from .grader_prompts import (
    COMPONENT_GRADER_PROMPT,
    SEARCH_GRADER_PROMPT,
)

# Re-export package namespaces for structured access
from . import seed_prompts, grader_prompts

# Backwards compatibility alias
GRADER_PROMPT = COMPONENT_GRADER_PROMPT

__all__ = [
    "AGGREGATOR_PROMPT",
    "ATOMIZER_PROMPT",
    "ATOMIZER_DEMOS",
    "PLANNER_PROMPT",
    "PLANNER_DEMOS",
    "COMPONENT_GRADER_PROMPT",
    "GRADER_PROMPT",
    "SEARCH_GRADER_PROMPT",
    "seed_prompts",
    "grader_prompts",
]

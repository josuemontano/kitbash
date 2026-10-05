"""Model roles. Every LLM call is made on behalf of exactly one role."""

from enum import StrEnum


class Role(StrEnum):
    CODE = "code"
    IMAGE_ANALYSIS = "image_analysis"
    REFERENCE_SELECTION = "reference_selection"
    VISUAL_CRITIC = "visual_critic"
    PROMPT_ANALYSIS = "prompt_analysis"
    TECHNICAL_CRITIC = "technical_critic"

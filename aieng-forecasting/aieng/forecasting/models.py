"""Canonical model identifiers used across the project.

The project standardizes on a single model so examples, defaults, and notebooks
stay consistent.  Currently configured for the Roche build-cli AI Gateway
(Claude models via Anthropic protocol).

All three constants currently resolve to the same model
(``anthropic/claude-sonnet-4-6[1m]``): the lite/default/advanced distinction is
retained as an API so a future split is a one-line change, but the project is
standardized on Sonnet 4.6 (including its multimodal image support, used to feed
CDC FluView chart images to the agent).

Reference these constants instead of hardcoding model strings, so a model
swap is a one-line change here rather than a repo-wide find-and-replace.

This module is intentionally dependency-free (it imports nothing from the rest
of the package) so it can be imported from anywhere without risking an import
cycle.
"""

from __future__ import annotations


#: Advanced model — higher capability; adaptive-agent and production runs.
ADVANCED_MODEL = "anthropic/claude-sonnet-4-6[1m]"

#: Default / lite model. Standardized on the advanced model for consistency.
LITE_MODEL = ADVANCED_MODEL

#: Alias for the project-wide default model.
DEFAULT_MODEL = ADVANCED_MODEL


__all__ = ["ADVANCED_MODEL", "DEFAULT_MODEL", "LITE_MODEL"]

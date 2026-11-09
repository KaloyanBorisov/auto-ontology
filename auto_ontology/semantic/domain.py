# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain summary and system prompt loading for seed selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SUMMARY_DIR = Path(".semantic_summaries")
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class DomainSummary(BaseModel):
    """Compact domain context for seed and Term extraction."""

    domains: list[str] = Field(default_factory=list)
    core_entities: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)


def load_domain_summary(database_name: str) -> DomainSummary | None:
    path = _SUMMARY_DIR / f"{database_name}.json"
    if not path.exists():
        return None
    try:
        return DomainSummary.model_validate_json(path.read_text())
    except Exception:
        logger.warning("Failed to load domain summary from %s", path)
        return None


def load_system_prompt() -> str:
    path = _PROMPTS_DIR / "system_prompt.txt"
    if path.exists():
        return path.read_text().strip()
    return (
        "You are compiling a business semantic layer over a relational database. "
        "Use clear, human-readable Term names with spaces between words and precise "
        "business language."
    )


def output_language_instruction() -> str:
    """Extra system-prompt rules for the language of generated text (fork).

    ``SEMANTIC_OUTPUT_LANGUAGE`` (e.g. ``Bulgarian``) asks for descriptions and
    extra synonyms in that language; Term names and display labels stay in
    English, which the label validation requires. ``SEMANTIC_CONTEXT`` is free
    text about the source data (what the abbreviations mean, where it comes
    from). Both unset returns "" and leaves the upstream prompts unchanged.
    """
    language = os.environ.get("SEMANTIC_OUTPUT_LANGUAGE", "").strip()
    context = os.environ.get("SEMANTIC_CONTEXT", "").strip()
    parts: list[str] = []
    if context:
        parts.append(f"Context about the source data: {context}")
    if language:
        parts.append(
            f"Write every description in {language}. Keep Term names and "
            f"display_name labels in English. Provide synonyms in {language} "
            "(in its native script) and, where useful, in English."
        )
    return ("\n\n" + "\n".join(parts)) if parts else ""

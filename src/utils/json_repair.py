"""
Utilities for repairing malformed JSON from LLM responses.

LLMs sometimes produce invalid JSON (markdown fences, trailing commas,
extra text around JSON). This module provides robust extraction.
"""

import json
import re
from typing import Any, Optional


def extract_json(text: str) -> Optional[Any]:
    """
    Attempt to extract valid JSON from potentially malformed LLM output.

    Strategies (in order):
    1. Direct parse
    2. Strip markdown code fences
    3. Find first [ or { and last ] or }
    4. Fix common issues (trailing commas)
    """
    if not text:
        return None

    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    cleaned = text
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find JSON boundaries
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start_idx = cleaned.find(start_char)
        end_idx = cleaned.rfind(end_char)
        if start_idx != -1 and end_idx > start_idx:
            candidate = cleaned[start_idx : end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 4: fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 5: fix trailing commas on the boundary-extracted text
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start_idx = fixed.find(start_char)
        end_idx = fixed.rfind(end_char)
        if start_idx != -1 and end_idx > start_idx:
            candidate = fixed[start_idx : end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    return None

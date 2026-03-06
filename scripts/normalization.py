"""Shared normalization helpers for conflation rules and feature extraction."""

from __future__ import annotations

import json
import math
import re
from typing import Any


MISSING_STRINGS = {"", "none", "null", "nan", "na", "n/a"}


def is_missing_value(value: Any) -> bool:
    """Return True when value should be treated as missing."""
    if value is None:
        return True

    # Handles float('nan') without pulling in pandas.
    if isinstance(value, float) and math.isnan(value):
        return True

    if isinstance(value, str):
        return value.strip().lower() in MISSING_STRINGS

    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0

    return False


def safe_parse_json(value: Any) -> Any:
    """Safely parse JSON strings while passing through native values."""
    if is_missing_value(value):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def normalize_text(value: Any) -> str:
    """Normalize text for robust comparison."""
    if is_missing_value(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_name_primary(value: Any) -> str:
    """Extract and normalize primary name value."""
    parsed = safe_parse_json(value)
    if isinstance(parsed, dict):
        return normalize_text(parsed.get("primary", ""))
    return normalize_text(value)


def extract_phone_primary(value: Any) -> str:
    """Extract first phone candidate as normalized text."""
    parsed = safe_parse_json(value)
    if isinstance(parsed, list) and parsed:
        return normalize_text(parsed[0])
    return normalize_text(value)


def phone_digits(value: Any) -> str:
    """Return digit-only normalized phone string."""
    primary = extract_phone_primary(value)
    return "".join(ch for ch in primary if ch.isdigit())


def extract_website_primary(value: Any) -> str:
    """Extract first website candidate and lightly canonicalize."""
    parsed = safe_parse_json(value)
    candidate = ""
    if isinstance(parsed, list) and parsed:
        candidate = normalize_text(parsed[0])
    else:
        candidate = normalize_text(value)

    # Keep path/query because location pages can carry signal.
    candidate = candidate.replace("http://", "").replace("https://", "")
    if candidate.startswith("www."):
        candidate = candidate[4:]
    return candidate.rstrip("/")


def extract_address_primary(value: Any) -> dict:
    """Extract first address object with normalized string fields."""
    parsed = safe_parse_json(value)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        return {}

    norm = {}
    for key, raw_val in parsed.items():
        norm[key] = normalize_text(raw_val)
    return norm


def extract_category_primary(value: Any) -> str:
    """Extract normalized primary category text."""
    parsed = safe_parse_json(value)
    if isinstance(parsed, dict):
        return normalize_text(parsed.get("primary", ""))
    return normalize_text(value)


def values_equivalent(attribute: str, current_value: Any, base_value: Any) -> bool:
    """Compare two attribute values after normalization."""
    if is_missing_value(current_value) and is_missing_value(base_value):
        return True

    if attribute == "name":
        return extract_name_primary(current_value) == extract_name_primary(base_value)
    if attribute == "phone":
        curr_digits = phone_digits(current_value)
        base_digits = phone_digits(base_value)
        if curr_digits and base_digits:
            return curr_digits == base_digits
        return extract_phone_primary(current_value) == extract_phone_primary(base_value)
    if attribute == "website":
        return extract_website_primary(current_value) == extract_website_primary(base_value)
    if attribute == "address":
        return extract_address_primary(current_value) == extract_address_primary(base_value)
    if attribute == "category":
        return extract_category_primary(current_value) == extract_category_primary(base_value)

    return normalize_text(current_value) == normalize_text(base_value)

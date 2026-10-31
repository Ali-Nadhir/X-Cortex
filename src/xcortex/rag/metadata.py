from __future__ import annotations

import re


def normalize_region(value: str | None) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    aliases = {
        "cxr": "chest", "thorax": "chest", "chest x ray": "chest",
        "upper arm": "humerus", "lower arm": "forearm",
        "fingers": "finger", "hands": "hand", "wrists": "wrist",
    }
    return aliases.get(value, value) or "unknown"

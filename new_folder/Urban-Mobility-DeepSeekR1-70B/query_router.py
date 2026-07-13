from __future__ import annotations

import re
from typing import Literal

Intent = Literal[
    "trip_distance",
    "demand_analysis",
    "pattern_analysis",
    "comparative_analysis",
    "general_mobility",
]


class QueryRouter:
    patterns: dict[Intent, tuple[str, ...]] = {
        "trip_distance": ("distance", "far", "miles", "between"),
        "demand_analysis": ("demand", "highest", "busiest", "pickup", "night"),
        "pattern_analysis": ("pattern", "peak", "hour", "weekend", "trend"),
        "comparative_analysis": ("compare", "versus", "vs", "manhattan", "brooklyn", "borough"),
        "general_mobility": ("traffic", "mobility", "taxi", "summarize"),
    }

    def detect(self, query: str) -> Intent:
        normalized = re.sub(r"\s+", " ", query.lower()).strip()
        scores = {
            intent: sum(1 for keyword in keywords if keyword in normalized)
            for intent, keywords in self.patterns.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general_mobility"

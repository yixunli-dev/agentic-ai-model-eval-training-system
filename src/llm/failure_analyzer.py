import os
from collections import Counter


class FailureAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def analyze(self, failure_cases):
        if not self.api_key:
            return self._fallback_analysis(failure_cases)

        # External LLM calls are intentionally skipped for local reproducibility.
        return self._fallback_analysis(failure_cases)

    def _fallback_analysis(self, failure_cases):
        if not failure_cases:
            return "No tagging failures were found on the evaluation dataset."

        error_patterns = Counter()
        for failure_case in failure_cases:
            for error in failure_case["errors"]:
                key = f"{error['gold_tag']} -> {error['predicted_tag']}"
                error_patterns[key] += 1

        pattern_summary = ", ".join(
            f"{pattern}: {count}" for pattern, count in error_patterns.most_common()
        )
        return (
            "Deterministic fallback analysis: the model made "
            f"{sum(error_patterns.values())} token-level errors. "
            f"Most common patterns: {pattern_summary}."
        )

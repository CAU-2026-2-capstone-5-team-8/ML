"""Select English analysis text while preserving original evidence for display."""

import re

NON_ENGLISH_SCRIPT = re.compile(
    r"[\u0400-\u052f\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]"
)


def analysis_text(original: str, english: str | None) -> str:
    text = english if english is not None else original
    if NON_ENGLISH_SCRIPT.search(text):
        if english is not None:
            raise ValueError("declared English analysis text contains non-English script")
        return ""
    return text

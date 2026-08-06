from typing import Any, TypedDict


class TestResult(TypedDict):
    statistic: float
    pvalue: float
    method: str
    extra: dict[str, Any]

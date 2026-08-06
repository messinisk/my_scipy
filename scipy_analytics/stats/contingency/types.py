from typing import Any, TypedDict


class ContingencyResult(TypedDict):
    statistic: float
    pvalue: float
    method: str
    extra: dict[str, Any]

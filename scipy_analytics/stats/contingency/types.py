from typing import Any, TypedDict


class ContingencyResult(TypedDict):
    """_summary_

    Args:
        TypedDict (dict):
        {
        statistic: float
            pvalue: float
            method: str
            extra: dict[str, Any]
        }
    """

    statistic: float
    pvalue: float
    method: str
    extra: dict[str, Any]

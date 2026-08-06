from typing import Any, TypedDict


class FitResult(TypedDict):
    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    method: str
    extra: dict[str, Any]

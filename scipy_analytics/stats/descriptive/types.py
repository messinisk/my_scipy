from typing import Any, TypedDict


class DescriptiveResult(TypedDict):
    mean: float
    median: float
    mode: float
    variance: float
    std: float
    skewness: float
    min: float
    max: float
    count: int
    percentiles: dict[int, float]
    kurtosis: float | None



class LowLevelResult(TypedDict):
    stat: float
    method: str
    extra: dict[str, Any]

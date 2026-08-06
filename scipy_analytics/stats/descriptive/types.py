from typing import Any, TypedDict


class DescriptiveResult(TypedDict):
    stat: float
    method: str
    extra: dict[str, Any]

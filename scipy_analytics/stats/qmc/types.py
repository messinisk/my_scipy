from typing import TypedDict

import numpy as np


class SampleResult(TypedDict):
    samples: np.ndarray
    method: str
    extra: dict

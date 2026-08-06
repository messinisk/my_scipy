from scipy_analytics.stats import fit_distribution, summarize_fit
from scipy.stats import gamma
import numpy as np

data = np.random.gamma(shape=2.0, scale=3.0, size=1000)

result = fit_distribution(gamma, data)
print(result)

summary = summarize_fit(result)
print(summary)

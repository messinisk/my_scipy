from setuptools import setup, find_packages

setup(
    name="scipy_analytics",
    version="0.2.3",
    description="Extensions for SciPy: Pearson distributions, Monte Carlo, fitting, plotting",
    author="messinis kostas",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
)

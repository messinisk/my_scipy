from setuptools import setup, find_packages

setup(
    name="scipy-analytics",
    version="0.1.1",
    description="Extensions for SciPy: Pearson distributions, Monte Carlo, fitting, plotting",
    author="messinisk",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
)

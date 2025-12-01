from setuptools import setup, find_packages

setup(
    name="fourier_copeland_bandits",
    version="1.0.0",
    description="A Fourier Approach to Sample-Efficient Copeland Bandit Identification",
    author="Ratnesh Kumar",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    entry_points={
        "console_scripts": [
            "fourier-bandits=run_experiment:main",
        ],
    },
)

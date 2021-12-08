from setuptools import setup

setup(
    name="flexgpt",
    version="0.1.0",
    description="Checkmate prevents you from OOMing when training big deep neural nets",
    packages=["minGPT"],  # find_packages()
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "transformers",
        "loguru",
        "pandas",
    ],
    extras_require={"test": ["pytest", "tensorflow", "matplotlib", "graphviz"]},
)

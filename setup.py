from setuptools import setup

setup(
    name="flexgpt",
    version="0.1.0",
    description="Checkmate prevents you from OOMing when training big deep neural nets",
    packages=["flexgpt"],  # find_packages()
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "transformers",
    ],
    extras_require={"test": ["pytest", "tensorflow", "matplotlib", "graphviz"]},
)

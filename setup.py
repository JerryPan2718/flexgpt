from setuptools import setup

setup(
    name="flexgpt",
    version="0.1.0",
    description="flexgpt speeds up large Language Model inference with tradeoff between model inference runtime and memory usage.",
    packages=["minGPT"],  # find_packages()
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "transformers",
        "loguru",
        "pandas",
        "pthflops",
        "python_papi",
    ],
    extras_require={"test": ["pytest", "tensorflow", "matplotlib", "graphviz"]},
)

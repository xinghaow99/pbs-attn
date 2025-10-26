from setuptools import setup, find_packages

setup(
    name="pbs_attn",
    version="0.1.0",
    packages=find_packages(include=["pbs_attn", "pbs_attn.*"]),
    install_requires=[
        "torch",
        "transformers",
        "accelerate"
    ],
    python_requires=">=3.7",
    author="Xinghao Wang",
    description="Permuted Block-Sparse Attention",
) 
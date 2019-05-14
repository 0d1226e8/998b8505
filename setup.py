from setuptools import setup, find_packages

setup(
    name="bdlb",
    version="0.0.1",
    description="BDL Benchmarks",
    url="https://github.com/oatml/bdlb",
    author="OATML",
    author_email="oatml@googlegroups.com",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pillow",
        "pathos",
        "kaggle",
        "tensorflow",
        "tensorflow-probability",
        "torch",
        "torchvision"
    ])

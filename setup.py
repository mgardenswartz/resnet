from setuptools import setup, find_packages

setup(
    name="jax_resnet",
    version="1.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
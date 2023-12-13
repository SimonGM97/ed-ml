from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ed_ml",
    author="Simón P. García Morillo",
    version="1.0.0",
    install_requires=requirements,
    packages=find_packages(),
    long_description=open("README.md").read()
)
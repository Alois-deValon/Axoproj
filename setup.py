from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="axoproj",
    version="0.0.1",
    author="Alois de Valon",
    author_email="aloisdevalon@gmail.com",
    description="A package to create P.P.V datacube from simple models",
    long_description=readme,
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

# flake8: noqa

import re

from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()


def read_requirements(path):
    reqs = []
    with open("requirements-dev.txt") as f:
        reqs = f.read().splitlines()
    return reqs


requirements_required = read_requirements("requirements-dev.txt")
requirements_extras = read_requirements("requirements-extras.txt")


def find_version():
    with open("larch/__init__.py", "r") as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version()

setup(
    name="larch",
    version=version,
    description="LLM toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NASA-IMPACT/larch",
    author_email="np0069@uah.edu",
    python_requires=">=3.8",
    packages=["larch", "larch.metadata", "larch.search", "larch.indexing"],
    install_requires=requirements_required,
    extras_require={"paperqa": ["paper-qa"], "extras": requirements_extras},
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Machine Learning",
        "Topic :: Neural Network",
        "Topic :: Transformers",
        "Topic :: Large Language Models",
        "Topic :: NLP",
        "Topic :: Natural Language Processing",
        "Topic :: Retrieval-augmented Generation",
    ],
    zip_safe=False,
)

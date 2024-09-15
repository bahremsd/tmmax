"""Python setup.py for tmmax package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("tmmax", "VERSION")
    '0.0.1'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="tmmax",
    version=read("tmmax", "VERSION"),
    description="project_description",
    url="https://github.com/bahremsd/tmmax/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="bahremsd",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["tmmax = tmmax.__main__:main"]
    },
    include_package_data=True,
    package_data={
        'tmmax': ['nk_data/*.csv'],
    },
)

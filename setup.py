from setuptools import setup
from setuptools import setup, find_packages

setup(
    name="pytorch-utils",
    version="0.1",
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
)

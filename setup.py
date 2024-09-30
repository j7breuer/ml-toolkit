import os
from setuptools import setup, find_packages

# Set version suffix
base_version = "0.0.1"

# Get Jenkinsfile version
version_suffix = os.getenv('VERSION_SUFFIX', '')

# Declare official version
version = f"{base_version}{version_suffix}"

setup(
    name = "ml_toolkit",
    version = version,
    author = "J Breuer",
    author_email = "jmbreuer@vt.edu",
    description = "",
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/j7breuer/ml-toolkit",
    packages = find_packages(where="src"),
    package_dir = {"": "src"},
    python_requires = '>=3.9',
    install_requires = [
        "pandas",
        "numpy"
    ]
)
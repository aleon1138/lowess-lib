from distutils.core import setup, Extension
import pybind11
import sys

MAX_PYTHON_VERSION = (3, 11)

# Get the current Python version
current_version = sys.version_info[:2]  # (major, minor)

# There are API conflicts with python 3.12+ due to OpenMP api changes
if current_version > MAX_PYTHON_VERSION:
    raise SystemExit(f"Error: Python {sys.version} is too new. Maximum supported version is {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}.")


module = Extension(
    "lowesslib",
    extra_compile_args=["-fopenmp", "-march=native", "-std=c++17"],
    extra_link_args=["-fopenmp"],
    include_dirs=[pybind11.get_include()],
    sources=["lowess.cc", "lowesslib.cc"],
)

setup(
    name="lowesslib",
    version="0.1",
    author="Arnaldo Leon",
    description="Utilities for Locally Weighted Scatterplot Smoothing",
    ext_modules=[module],
)

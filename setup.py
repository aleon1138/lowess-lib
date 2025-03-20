from distutils.core import setup, Extension
import pybind11
import sys

MAX_PYTHON_VERSION = (3, 11)

# Get the current Python version
current_version = sys.version_info[:2]  # (major, minor)

# There are API conflicts with python 3.12+ and OpenMP
if current_version > MAX_PYTHON_VERSION:
    s = lambda v: f"{v[0]}.{v[1]}"
    raise SystemExit(
        f"Error: Python {s(current_version)} not supported due to OpenMP incompatibility. Maximum supported version is {s(MAX_PYTHON_VERSION)}."
    )


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

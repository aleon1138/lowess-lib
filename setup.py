from distutils.core import setup, Extension
import pybind11
import sys

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

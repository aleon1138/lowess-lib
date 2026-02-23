from setuptools import setup, Extension
import pybind11

module = Extension(
    "lowesslib",
    extra_compile_args=["-O3", "-fopenmp", "-march=native", "-std=c++17"],
    extra_link_args=["-fopenmp"],
    include_dirs=[pybind11.get_include()],
    sources=["lowess.cc", "lowesslib.cc"],
)

setup(
    name="lowesslib",
    version="0.2.2",
    author="Arnaldo Leon",
    author_email="amleon@alum.mit.edu",
    description="Utilities for Locally Weighted Scatterplot Smoothing",
    long_description=open("README.md").read(),
    long_description_context_type="text/markdown",
    ext_modules=[module],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18",
    ],
    project_urls={
        "Source": "https://github.com/aldon1138/lowess-lib",
    },
)

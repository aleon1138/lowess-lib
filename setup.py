from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "lowesslib",
        sources=["lowess.cc", "lowesslib.cc", "expectile.cc"],
        extra_compile_args=["-O3", "-fopenmp", "-march=native"],
        extra_link_args=["-fopenmp"],
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules)

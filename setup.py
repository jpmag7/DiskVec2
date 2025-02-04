# setup.py
# Compile: python setup.py build_ext --inplace
import sys
from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-Ofast", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "DiskVec",
        sources=["DiskVec.cpp", "bindings.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="DiskVec",
    version="0.0.1",
    author="Pedro Magalhães",
    description="Python bindings for DiskVec",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    setup_requires=[
        "pybind11>=2.6.0",
    ],
    zip_safe=False,
)

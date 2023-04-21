
import setuptools
import numpy
import platform

from Cython.Build import cythonize
from distutils.extension import Extension

if platform.system() == 'Windows':
    extra_compile_args = ["/Ox", "/openmp", "/arch:AVX2"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-fopenmp", "-mavx2", "-msse4.2", "-std=c++11"]
    extra_link_args = ['-fopenmp']

sourcefiles = ['cWrap/understandingCWrap.pyx', 'cWrap/libs/understandingCWrap.cpp']
extensions = [Extension(name='cWrap.understandingCWrap', sources=sourcefiles, language="c++",
                        include_dirs=[numpy.get_include()], extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)]

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="cWrap",
    version="1.0.0",
    author="Supratik.Bose",
    author_email="supratik.bose@gmail.com",
    description="Understand Python wrapper around C.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supratikbose/understandingCWrap",
    packages=setuptools.find_packages(exclude=("tests",)),
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)

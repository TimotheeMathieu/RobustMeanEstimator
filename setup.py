from setuptools import setup
import setuptools

from Cython.Build import cythonize
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    version="0.0.1",
    author="Timothée Mathieu",
    author_email="timotheemathieu@mailoo.org",
    description="Robust Mean estimation algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    name='robust mean',
    ext_modules=cythonize("robust_mean/_robust_mean.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

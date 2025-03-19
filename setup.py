from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lib1d_md",
    version="0.0.1",
    author="Ali Asad",
    author_email="ali.asad@polytechnique.edu",
    description="1D LIB simulations with multi-domain time integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hpc-maths/LIB1d_md",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'joblib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license="BSD-3-Clause",
)

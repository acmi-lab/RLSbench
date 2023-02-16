# Adapted from https://github.com/p-lambda/wilds/blob/main/setup.py

import os
import sys

import setuptools

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, "RLSbench"))

from version import __version__

print(f"Version {__version__}")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RLSBench",
    version=__version__,
    author="Saurabh Garg",
    author_email="sgarg2@andrew.cmu.edu",
    description="Relaxed Label Shift benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.21.1",
        "pandas>=1.1.0",
        "pillow>=7.2.0",
        "pytz>=2020.4",
        "torch>=1.10.0",
        "torchvision>=0.11.3",
        "tqdm>=4.53.0",
        "scikit-learn>=0.20.0",
        "scipy>=1.5.4",
        "cvxpy>=1.1.7",
        "cvxopt>=1.3.0",
        "transformers>=4.21",
        "matplotlib>=3.5.1",
        "networkx>=2.0",
        "antialiased-cnns",
        "folktables",
        "clip @ git+https://github.com/openai/CLIP.git##egg=clip",
        "calibration @ git+https://github.com/saurabhgarg1996/calibration.git#egg=calibration",
        "wilds @ git+https://github.com/saurabhgarg1996/wilds.git#egg=wilds",
        "robustness @ git+https://github.com/saurabhgarg1996/robustness.git#egg=robustness",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cu113",
    ],
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)

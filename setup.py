import setuptools
from setuptools import find_packages
import subprocess
import sys
import re
import os

with open("./autodistill_grounded_sam_2/__init__.py", 'r') as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)
    
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodistill_grounded_sam_2", 
    version=version,
    author="Roboflow",
    author_email="autodistill@roboflow.com",
    description=" Use Segment Anything 2, grounded with Florence-2, to auto-label data for use in training vision models. ",
    long_description=" Use Segment Anything 2, grounded with Florence-2, to auto-label data for use in training vision models. ",
    long_description_content_type="text/markdown",
    url="https://github.com/autodistill/autodistill-grounded-sam-2",
    install_requires=[
        "torch",
        "autodistill",
        "numpy>=1.20.0",
        "opencv-python>=4.6.0",
        "supervision",
        "roboflow",
        "autodistill_florence_2",
    ],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

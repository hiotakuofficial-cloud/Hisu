"""
Setup configuration for AI/ML environment
"""

from setuptools import setup, find_packages

setup(
    name="aiml-environment",
    version="1.0.0",
    description="Comprehensive AI/ML environment with neural network implementations",
    author="AI/ML Team",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

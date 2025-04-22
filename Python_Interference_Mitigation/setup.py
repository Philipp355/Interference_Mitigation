from setuptools import setup, find_packages

setup(
    name="radar_interference_suppression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'h5py>=3.2.0'
    ]
)
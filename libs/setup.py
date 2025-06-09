from setuptools import setup, find_packages

setup(
    name='custom_modules',
    version=1.0,
    package=['custom_modules'],
    python_requires='>=3.10, <4',
    install_requires=['python-mnist==0.7']
)
from setuptools import setup, find_packages
import re

def get_version():
    default_ver = '0.1.2'
    try:
        with open('hints/__init__.py', 'r') as f:
            version_ = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
            if version_:
                return version_.group(1)
    except FileNotFoundError:
        pass
    return default_ver

def get_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='hints-kmcs',
    version=get_version(),
    author='Amin Akhshi',
    author_email='amin.akhshi@gmail.com',
    description='A package for calculating pairwise and higher-order interactions of N-dimensional state variables from measured time series',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aminakhshi/hints',
    packages = find_packages(exclude=["misc*", "result*", "data*"]),
    install_requires=get_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    # package_dir = {"": "hints"},
    python_requires='>=3.8.1',
)

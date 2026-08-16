from setuptools import setup, find_packages
import re

def get_version():
    default_ver = '0.1.3'
    try:
        with open('hints/__init__.py', 'r') as f:
            version_ = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
            if version_:
                return version_.group(1)
    except FileNotFoundError:
        pass
    return default_ver

def get_requirements():
    try:
        with open('requirements.txt') as req:
            return [line.strip() for line in req if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        # Keep the build working if the file is missing from a source tree.
        return ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn']


def get_long_description():
    for filename, content_type in (('README.md', 'text/markdown'),
                                   ('README.rst', 'text/x-rst')):
        try:
            with open(filename, encoding='utf-8') as handle:
                return handle.read(), content_type
        except FileNotFoundError:
            continue
    return '', 'text/plain'


LONG_DESCRIPTION, LONG_DESCRIPTION_TYPE = get_long_description()

setup(
    name='hints-kmcs',
    version=get_version(),
    author='Amin Akhshi',
    author_email='amin.akhshi@gmail.com',
    description='A package for calculating pairwise and higher-order interactions of N-dimensional state variables from measured time series',
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_TYPE,
    url='https://github.com/aminakhshi/hints',
    packages = find_packages(exclude=["misc*", "result*", "data*", "tests*", "examples*"]),
    install_requires=get_requirements(),
    extras_require={
        # Optional GPU/HPC backend. The estimator runs on numpy by default, so
        # this is never needed to install or use the package.
        'torch': ['torch>=1.10'],
        # Only needed to run the notebooks under examples/, which simulate the
        # benchmark systems. jitcsde is compiled, so it is deliberately kept out
        # of the mandatory requirements.
        'examples': ['jitcsde', 'sympy', 'tqdm', 'ipykernel'],
        'dev': ['pytest>=7.0'],
    },
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

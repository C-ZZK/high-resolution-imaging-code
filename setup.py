# Import necessary functions from setuptools
from setuptools import setup, find_packages

# --- Project Configuration ---

# Define the project version. Start with 0.1.0 for an initial release.
VERSION = "0.1.0"

# [cite_start]A short description of your project, taken from your paper's abstract[cite: 19].
DESCRIPTION = "An effective deep domain adaptation approach for least squares migration"

# Read the contents of your README.md file for the long description.
# This makes your PyPI page look good.
with open("README.md", "r", encoding="utf-8") as f:
    long_description_readme = f.read()

# --- setup() Function ---
# This function contains all the metadata about your package.

setup(
    # The name of your package on PyPI. It should be unique.
    # Dashes are common for package names.
    name="DA-ID-LSM",
    version=VERSION,

    # [cite_start]Author information, based on your paper[cite: 2, 3, 6].
    author="Ni Wenjun, Liu Shaoyong",
    author_email="wenjun.ni@cug.edu.cn",

    # The short description defined above.
    description=DESCRIPTION,

    # The long description read from your README file.
    long_description=long_description_readme,
    long_description_content_type="text/markdown",

    # Keywords to help others find your package. [cite_start]Based on your paper's keywords[cite: 9, 10, 11].
    keywords=['deep learning', 'domain adaptation', 'least squares migration', 'seismic imaging', 'geophysics'],

    # [cite_start]The URL for your project's homepage (e.g., the GitHub repository)[cite: 213].
    url="https://github.com/C-ZZK/high-resolution-imaging-code",

    # A link to download a specific version of your source code.
    download_url=f"https://github.com/C-ZZK/high-resolution-imaging-code/archive/refs/tags/v{VERSION}.tar.gz",

    # Automatically find all packages in your project.
    # This expects your Python code to be in a directory (e.g., 'da_id_lsm/').
    packages=find_packages(),

    # A list of other Python packages that your project depends on.
    # These will be installed automatically by pip.
    # [cite_start]Based on your paper [cite: 211] and previous code.
    install_requires=[
        'torch>=2.0.0',
        'opencv-python',
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm'
    ],

    # Trove classifiers to categorize your project on PyPI.
    # https://pypi.org/classifiers/
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License", # As we discussed
        "Operating System :: OS Independent",
    ],

    # Specify the minimum Python version required.
    python_requires='>=3.8',
)

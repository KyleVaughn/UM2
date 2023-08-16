# Documentation installation

Read the Docs is an open-sourced free software documentation hosting platform.It generates documentation written
with the Sphinx documentation generator.[^1]

This project use Doxygen to generate C++ API documentation breath to convert the
Doxygen output into Sphinx.

## Installation Guide

1. Install python3 depemdencies:
   use virtualenv or system python3
    ```bash
    pip install -r docs/python-req/requirements.txt
    ```
2. Install Doxygen:
    ```bash
    sudo apt-get install doxygen
    ```

3. the overall artifact is reference to
   this [blog](https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/)
   here are some noticeable modifications
    1. the `conf.py` will ask read the docs to locate all header file and inline file in include directory.
    2. index.rst can include markdown file as sub document.

[^1]: [Read the Docs](https://readthedocs.org/)
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import subprocess
import os


def configureDoxyfile(input_dir, output_dir):
    with open('Doxyfile.in', 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('@DOXYGEN_INPUT_DIR@', input_dir)
    filedata = filedata.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)

    with open('Doxyfile', 'w') as file:
        file.write(filedata)


# Check if we're running on Read the Docs' servers
read_the_docs_build = True


def find_directories_and_header_files(directory):
    results = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h'):
                results.append(os.path.join(root, file))
        for subdir in dirs:
            results.append(find_directories_and_header_files(subdir))

    return results


include_dir = '../include'
file_list = find_directories_and_header_files(include_dir)
print(file_list)

breathe_projects = {}
if read_the_docs_build:
    # recursively find all directory and *.h file under ../include and generate a string seperate by space
    output_dir = 'build'
    configureDoxyfile(include_dir, output_dir)
    subprocess.call('doxygen', shell=True)
    breathe_projects['UM2'] = output_dir + '/xml'

project = 'UM2'
copyright = '2023, Kyle'
author = 'Kyle'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["breathe"]
breathe_default_project = "UM2"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

root_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

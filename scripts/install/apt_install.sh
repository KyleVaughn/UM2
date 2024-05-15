#!/bin/bash

# Install dependencies via apt
# Usage: ./apt_install.sh

# Essential packages
sudo apt install -y \
    build-essential \
    cmake \
    git

# Development tools
sudo apt install -y \
    clang-format \
    clang-tidy \
    valgrind


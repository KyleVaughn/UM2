#!/bin/bash

# Install dependencies via apt
# Usage: ./apt_install.sh

# Essential packages
sudo apt install -y \
    build-essential \
    cmake \
    git

# Optional packages
sudo apt install -y \
  libopenblas-dev \
  liblapacke-dev \
  libhdf5-dev \
  libpugixml-dev \
  libgmsh-dev

# Development tools
sudo apt install -y \
    clang-format \
    clang-tidy \
    valgrind

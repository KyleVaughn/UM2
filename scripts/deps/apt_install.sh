#!/bin/bash

# Install dependencies via apt
# Usage: ./apt_install.sh

# Update package list
sudo apt update -y

# Verify that this is an Ubuntu system
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        echo "This script is intended for Ubuntu systems only."
        exit 1
    fi
    # If version is les than 22, we have not set up the script for that yet, so exit
    if [ "$VERSION_ID" -lt 22 ]; then
        echo "This script is intended for Ubuntu 22.04 and later."
        exit 1
    fi
else
    echo "This script is intended for Ubuntu systems only."
    exit 1
fi

# If version is 24+, we can simply install the default versions of the packages
sudo apt install -y \
    build-essential \
    cmake \
    gcc \
    git \
    gfortran \
    clang-format \
    clang-tidy \
    libomp-dev \
    valgrind \
    libopenblas-dev \
    liblapacke-dev \
    libhdf5-dev \
    libpugixml-dev \
    libgmsh-dev

if [ "$VERSION_ID" -ge 24 ]; then
    exit 0
fi

# Install gcc-12 and clang-15
sudo apt install -y \
    gcc-12 \
    g++-12 \
    gfortran-12 \
    clang-15 \
    clang-format-15 \
    clang-tidy-15

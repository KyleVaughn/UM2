# Installation Guide

## Installation through apt (Ubuntu/Debian)

### We support g++-12 or clang-15

```bash
#   install gcc and g++
    sudo apt -y update
    sudo apt install -y gcc-12 g++-12
    export CC=gcc-12 CXX=g++-12
#   alternatively
    sudo apt install -y clang-15
    export CC=clang-15 CXX=clang++-15
    sudo apt install -y libomp-15-dev
```

### Install dependencies

```bash
sudo apt -y update
sudo apt install -y libhdf5-dev libpugixml-dev
```

### If you are a developer, you may also need to install the following dependencies

```bash
sudo apt install -y clang-tidy-15 clang-format-15 cppcheck libomp-15-dev
sudo rm /usr/bin/clang-tidy
sudo ln -s /usr/bin/clang-tidy-15 /usr/bin/clang-tidy
```



    
    
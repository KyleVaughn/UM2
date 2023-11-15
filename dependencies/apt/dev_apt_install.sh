sudo apt -y update
sudo apt install -y g++-12 libhdf5-dev libpugixml-dev libtbb-dev libglu1-mesa
sudo apt install -y libpng-dev

# In a directory outside of UM2
mkdir um2_dependencies && cd um2_dependencies

# Install cmake
wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6.tar.gz
tar -xzvf cmake-3.27.6.tar.gz && cd cmake-3.27.6
./bootstrap && make && sudo make install && cd ..

# Install gmsh
wget https://gmsh.info/bin/Linux/gmsh-4.11.1-Linux64-sdk.tgz
tar -xzvf gmsh-4.11.1-Linux64-sdk.tgz && cd gmsh-4.11.1-Linux64-sdk

# Add GMSH_ROOT to your bashrc so cmake can find gmsh.
echo "export GMSH_ROOT=${PWD}" >> ~/.bashrc && source ~/.bashrc && cd ..

sudo apt install -y clang-15 clang-tidy-15 clang-format-15 libomp-15-dev cppcheck
sudo ln -s /usr/bin/clang-format-15 /usr/bin/clang-format
sudo ln -s /usr/bin/clang-tidy-15 /usr/bin/clang-tidy

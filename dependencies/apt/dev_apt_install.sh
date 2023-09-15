# This is for desktop machines only
sudo apt -y update
sudo apt install -y libhdf5-dev libpugixml-dev libtbb-dev libglu1-mesa
sudo apt install -y clang-15 clang-tidy-15 clang-format-15 cppcheck libomp-15-dev
wget https://gmsh.info/bin/Linux/gmsh-4.11.1-Linux64-sdk.tgz
tar -xzvf gmsh-4.11.1-Linux64-sdk.tgz

# Pre-requisites (Ubuntu):
# sudo apt install build-essential, gfortran, libxm2-dev, mesa-utils

# Back out of UM2 directory
cd ../../

# Clone spack to version 0.20
echo "git clone --depth=100 --branch=releases/v0.20 https://github.com/spack/spack.git"
git clone --depth=100 --branch=releases/v0.20 https://github.com/spack/spack.git

# Source spack upon login (and again now)
echo "source ${PWD}/spack/share/spack/setup-env.sh" >> ~/.bashrc && source ~/.bashrc

# Find the currently installed compilers
spack compiler find

# Make a local tmp directory to build in
mkdir tmp

# Create a spack environment just for GCC 12.3, so it can't get
# garbage collected
spack env create gcc12
spack env activate -p gcc12
spack add gcc@12.3
spack spec
echo "Installing GCC 12.3.0"
TMP=${PWD}/tmp spack install
spack compiler find
despacktivate

# Now, when we install GCC 12.3 in the global environment, it will use
# the one we just built and be safe from garbage collection.
spack install gcc@12.3
spack load gcc@12.3
spack compiler find

# Create a spack environment for UM2, using the GCC 12.3 compiler
# to build all the dependencies. This environment will contain all
# dependencies, except GCC 12.3. This way we can build all the dependencies
# with GCC 12.3, and then build UM2 with GCC 12.3. Note, if GCC 12.3 is 
# added to the environment, spack errors, likely due to a circular dependency.
spack env create um2 ${PWD}/UM2/dependencies/spack/user/server.yaml
spack env activate -p um2
spack spec
echo "Installing dependencies. This will take an hour or two."
time TMP=${PWD}/tmp spack install
rm -r tmp

# Make a copy of the environment and try to add gcc to it?
cd UM2
mkdir build
cd build
cmake ..
make -j
ctest && make install
cd ../..
rm -r tmp

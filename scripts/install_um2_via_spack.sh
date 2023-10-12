cd ../../

echo "git clone --depth=100 --branch=releases/v0.20 https://github.com/spack/spack.git"
git clone --depth=100 --branch=releases/v0.20 https://github.com/spack/spack.git
echo "source ${PWD}/spack/share/spack/setup-env.sh" >> ~/.bashrc && source ~/.bashrc

echo "spack compiler find"
spack compiler find

mkdir tmp
echo "TMP=${PWD}/tmp spack install gcc@12.3.0"
TMP=${PWD}/tmp spack install gcc@12.3.0

echo "spack load gcc@12.3.0"
spack load gcc@12.3.0

echo "spack compiler find"
spack compiler find

echo "check gcc version = 12.3.0: $(gcc --version)"

echo "spack env create um2 ./UM2/dependencies/user/desktop/server.yaml"
spack env create um2 ${PWD}/UM2/dependencies/spack/user/server.yaml

echo "spack env activate -p um2"
spack env activate -p um2

echo "spack load gcc@12.3.0" >> ~/.bashrc

echo "spack spec"
spack spec

echo "spack install"
spack install

cd UM2
mkdir build
cd build
cmake ..
make -j
ctest && make install

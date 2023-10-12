cd ../../

git clone --depth=100 --depth=100 --branch=UM2 git@code-nuram.miserver.it.umich.edu:MPACT/MPACT.git

cd ./MPACT
git clone --depth=100 git@code-nuram.miserver.it.umich.edu:VERA/Futility.git
git clone --depth=100 git@code-nuram.miserver.it.umich.edu:MPACT/MPACT_Extras.git
cd ..

spack env deactivate
mkdir tmp
TMP=${PWD}/tmp spack install gcc@8.5.0
spack load gcc@8.5.0
spack compiler find
TMP=${PWD}/tmp spack install mpich@3.3%gcc@8.5.0
spack load mpich@3.3%gcc@8.5.0
cd ./MPACT
spack env activate -p um2
mkdir build
cd build
cp ../build_scripts/build_with_um2.sh ./
cp ../build_scripts/build_with_hdf5.sh ./
cp ../build_scripts/build_mpi_release_all.sh ./
SPACK_VIEW_DIR=${SPACK_ENV}/.spack-env/view
HDF5_ROOT=$SPACK_VIEW_DIR UM2_ROOT=${PWD}/../../UM2/build ./build_with_hdf5.sh ./build_with_um2.sh ./build_mpi_release_all.sh ..
make -j MPACT
spack env deactivate
spack load gcc@12.3.0
cd ../..
TMP=${PWD}/tmp spack install mpich@3.3%gcc@12.3.0
spack load mpich@3.3%gcc@12.3.0
spack env activate -p um2
cd ./MPACT/build/MPACT_exe/src
make MPACT VERBOSE=1
echo "Now copy the command and use you local mpif90"

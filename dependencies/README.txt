For debian-based desktop machines for which the user is a member of the sudoers group,
dependencies may be installed using apt. See "apt_install.sh"

For all others, dependencies are installed via spack. 

To install spack:
  git clone -c feature.manyFiles=true https://github.com/spack/spack.git

Ensure that python version >=3.6 is available and setup the spack environment.
  . spack/share/spack/setup-env.sh

Install gcc-12 or clang-15 on the system or using spack. To install using spack: 
  spack compiler find
  spack install gcc@12 (or llvm@15)
    
Then, to install UM2 dependencies:
  spack load gcc@12 (or llvm@15)
  spack compiler find

Pick the appropriate spack.yaml file in dependencies/spack.
  spack env create um2 <choice of spack env>
  spack env activate -p um2 
  spack spec
  spack install

If you're using the default spack.yaml, you may need to add:
```
packages:
  opengl:
    buildable: false
    externals:
    - spec: opengl@<OpenGL version on your machine>
      prefix: <path to opengl, such as /usr/x86_64-linux-gnu>
```
in ~/.spack/packages.yaml 

To build and install UM2:
  cd UM2
  mkdir build && cd build
If you used the apt_install.sh script to install gmsh, you mat need to set
  export GMSH_ROOT=<path/to/gmsh>
  cmake ..
  make -j
  ctest (Ensure the tests pass!)
  make install

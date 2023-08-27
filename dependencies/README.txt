Dependencies are installed via spack.
In spack.yaml,
    Change +fltk to ~fltk if a mesh viewer is not needed
    Delete the cuda spec if cuda is not desired supported

To install spack:
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh

Install gcc-12 or clang-15 on the system or using spack.
To install using spack: 
    spack compiler find
    spack install gcc@12 or spack install llvm@15
    spack load <choice above>
    
Then, to install UM2 dependencies:
spack compiler find
spack env create um2 spack.yaml
spack env activate -p um2
spack spec
spack install

You may need to place:
```
packages:
  opengl:
    buildable: False
    externals:
    - spec: opengl@4.6.0
      prefix: /usr/x86_64-linux-gnu
```
in ~/.spack/packages.yaml 
if using a configuration which requires gmsh

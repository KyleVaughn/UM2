#!/bin/bash
JULIA_VERSION="1.7.2"
FLTK_VERSION="1.3.8"
OCC_VERSION="V7_6_0"
GMSH_VERSION="4.9.5"

# Check operating system. Exit if not linux 
if [ "$OSTYPE" != "linux-gnu" ]; then
    echo "Operating system: $OSTYPE"
    echo "This script currently only supports linux-gnu operating systems."
    exit 1
fi

# Process options
PREFIX=/usr/local
JOBS=1
for i in "$@"; do
  case $1 in
    -h|--help)
      echo "Usage: ./install.sh [options]"
      echo " "
      echo "Examples:"
      echo "  ./install.sh                      Install to the default [/usr/local]." 
      echo "  ./install.sh -j8 --prefix=/lib    Install to /lib, with 8 jobs at once."
      echo " "
      echo "Options:"
      echo "  -h, --help          Show brief help."
      echo "  --jobs[=N]          Allow N jobs at once." 
      echo "  --prefix=PREFIX     Install files to PREFIX [/usr/local by default]"
      exit 0
      ;;
    --prefix=*)
      PREFIX="${i#*=}"
      shift
      ;;
    --jobs=*)
      JOBS="${i#*=}"
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Ensure that script is run as sudoer
if [[ $(id -u) != 0 ]]; then
    echo "Run as superuser, i.e. sudo ./install.sh"
    exit 1
fi

# Begin install
start_dir=$(pwd)

# Install curl if needed
if ! [ -x "$(command -v curl)" ]; then
  apt install curl
fi

# Install Julia
while true; do
  read -p "Install Julia? " yn
  case $yn in
    Yes ) 
      cd $PREFIX
      curl -O https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION:0:3}/julia-$JULIA_VERSION-linux-x86_64.tar.gz 
      tar -xzvf julia-$JULIA_VERSION-linux-x86_64.tar.gz
      rm -f $PREFIX/bin/julia
      ln -s $PREFIX/julia-$JULIA_VERSION/bin/julia $PREFIX/bin/julia
      rm julia-$JULIA_VERSION-linux-x86_64.tar.gz
      cd $start_dir
      break
      ;;
    No ) 
      break
      ;;
    * ) 
      echo "Please answer Yes or No."
      ;;
  esac
done

# Install gmsh
while true; do
  read -p "Install gmsh? " yn
  case $yn in
    Yes ) 
      cd $PREFIX

      # Prerequisites 
      apt install software-properties-common libtool autoconf automake gfortran gdebi -y
      apt install gcc-multilib libxmu-headers -y
      apt install libx11-dev mesa-common-dev libglu1-mesa-dev -y
      apt install libfontconfig1-dev -y
      apt install tcllib tklib tcl-dev tk-dev libfreetype-dev libxt-dev libxmu-dev libxi-dev -y
      apt install libgl1-mesa-dev libfreeimage-dev rapidjson-dev -y
      apt install libtbb-dev libxi-dev libxmu-dev -y
      apt install cmake -y
      apt install doxygen -y
      apt install libx11-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev libxft-dev -y
      apt install libomp-dev libalglib-dev -y

      # Install OCC
      while true; do
        read -p "Install gmsh dependency: OpenCASCADE? " yn
        case $yn in
          Yes ) 
            curl -L -o occt.tgz "http://git.dev.opencascade.org/gitweb/?p=occt.git;a=snapshot;h=refs/tags/$OCC_VERSION;sf=tgz" 
            tar -xzvf occt.tgz
            cd occt-$OCC_VERSION
            mkdir build
            cd build
            cmake -DCMAKE_BUILD_TYPE=Release -DUSE_TBB=1 -DBUILD_MODULE_Draw=0 \
                  -DBUILD_MODULE_Visualization=0 -DBUILD_MODULE_ApplicationFramework=0 ..
            make --jobs=$JOBS
            make install
            cd $PREFIX
            rm occt.tgz
            break;;
          No )
            break;;
          * )
            echo "Please answer Yes or No.";;
        esac
      done

      # FLTK
      while true; do
        read -p "Install gmsh dependency: FLTK? " yn
        case $yn in
          Yes ) 
            curl -O https://www.fltk.org/pub/fltk/$FLTK_VERSION/fltk-$FLTK_VERSION-source.tar.gz 
            tar -xzvf fltk-$FLTK_VERSION-source.tar.gz
            cd fltk-$FLTK_VERSION
            mkdir build
            cd build
            cmake -DCMAKE_POSITION_INDEPENDENT_CODE=0 -DOPTION_BUILD_SHARED_LIBS=0 ..
            make --jobs=$JOBS
            make install
            cd $PREFIX
            rm fltk-$FLTK_VERSION-source.tar.gz
            break;;
          No )
            break;;
          * )
            echo "Please answer Yes or No.";;
        esac
      done

      # Gmsh
      curl -O https://gmsh.info/src/gmsh-$GMSH_VERSION-source.tgz
      tar -xzvf gmsh-$GMSH_VERSION-source.tgz
      cd gmsh-$GMSH_VERSION-source
      mkdir build
      cd build
      cmake -DENABLE_BUILD_DYNAMIC=1 -DENABLE_OCC_TBB=1 -DENABLE_OPENMP=1 ..
      make --jobs=$JOBS
      make install
      cd $PREFIX
      rm gmsh-$GMSH_VERSION-source.tgz
      cd $start_dir
      break
      ;;
    No ) 
      break
      ;;
    * ) 
      echo "Please answer Yes or No."
      ;;
  esac
done

echo "Installation finished."
echo " "
echo "Be sure to add the following to your environment:"
echo ""export JULIA_LOAD_PATH=\$\{JULIA_LOAD_PATH\}:$PREFIX/lib""

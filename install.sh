#!/bin/bash
JULIA_VERSION="1.7.2"
FLTK_VERSION="1.3.8"
OCC_VERSION="V7_6_0"
GMSH_VERSION="4.9.5"
if [[ $(id -u) == 0 ]]; then
  start_dir=$(pwd)
  # Install curl if needed
  if ! [ -x "$(command -v curl)" ]; then
    apt install curl
  fi
  
  while true; do
    read -p "Do you wish to install Julia? " yn
    case $yn in
      Yes ) 
        echo "Installing to /usr/local"
        cd /usr/local
        echo "Fetching Julia"
        curl -O https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION:0:3}/julia-$JULIA_VERSION-linux-x86_64.tar.gz 
        tar -xzvf julia-$JULIA_VERSION-linux-x86_64.tar.gz
        rm -f /usr/local/bin/julia
        ln -s /usr/local/julia-$JULIA_VERSION/bin/julia /usr/local/bin/julia
        rm julia-$JULIA_VERSION-linux-x86_64.tar.gz
        cd $start_dir
        break;;
      No ) 
        break;;
      * ) 
        echo "Please answer Yes or No.";;
    esac
  done
  
  while true; do
    read -p "Do you wish to install Gmsh? " yn
    case $yn in
      Yes ) 
        echo "Installing to /usr/local"
        cd /usr/local

        # OCC
        read -p "Do you wish to install Gmsh dependency: OpenCASCADE? " yn
        case $yn in
          Yes ) 
            echo "Fetching Gmsh dependency: OpenCASCADE"
            curl -L -o occt.tgz "http://git.dev.opencascade.org/gitweb/?p=occt.git;a=snapshot;h=refs/tags/$OCC_VERSION;sf=tgz" 
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
            tar -xzvf occt.tgz
            cd occt-$OCC_VERSION
            mkdir build
            cd build
            cmake -DCMAKE_BUILD_TYPE=Release -DUSE_TBB=1 -DBUILD_MODULE_Draw=0 \
                  -DBUILD_MODULE_Visualization=0 -DBUILD_MODULE_ApplicationFramework=0 ..
            make -j8
            make install
            cd /usr/local
            rm occt.tgz
            break;;
          No )
            break;;
          * )
            echo "Please answer Yes or No.";;
        esac

        # FLTK
        read -p "Do you wish to install Gmsh dependency FLTK? " yn
        case $yn in
          Yes ) 
            echo "Fetching Gmsh dependency: FLTK"
            curl -O https://www.fltk.org/pub/fltk/$FLTK_VERSION/fltk-$FLTK_VERSION-source.tar.gz 
            tar -xzvf fltk-$FLTK_VERSION-source.tar.gz
            cd fltk-$FLTK_VERSION
            mkdir build
            cd build
            cmake -DCMAKE_POSITION_INDEPENDENT_CODE=0 -DOPTION_BUILD_SHARED_LIBS=0 ..
            #./configure --enable-shared --enable-threads
            make -j8
            make install
            cd /usr/local
            rm fltk-$FLTK_VERSION-source.tar.gz
            break;;
          No )
            break;;
          * )
            echo "Please answer Yes or No.";;
        esac

        # Gmsh
        echo "Fetching Gmsh"
        curl -O https://gmsh.info/src/gmsh-$GMSH_VERSION-source.tgz
        tar -xzvf gmsh-$GMSH_VERSION-source.tgz
        cd gmsh-$GMSH_VERSION-source
        mkdir build
        cd build
        cmake -DENABLE_BUILD_DYNAMIC=1 -DENABLE_OCC_TBB=1 -DENABLE_OPENMP=1 ..
        make -j8
        make install
        cd /usr/local
        rm gmsh-$GMSH_VERSION-source.tgz
        # Append the Gmsh julia API install location to bashrc
        echo ""export JULIA_LOAD_PATH=\$\{JULIA_LOAD_PATH\}:/usr/local/lib"" >> ~/.bashrc
        cd $start_dir
        break;;
      No ) 
        break;;
      * ) 
        echo "Please answer Yes or No.";;
    esac
  done
  
  while true; do
    read -p "Do you wish to install Paraview? " yn
    case $yn in
      Yes ) 
        apt install paraview
        break;;
      No ) 
        break;;
      * ) 
        echo "Please answer Yes or No.";;
    esac
  done
  
  echo "Adding package to Julia's path and precompiling"
  julia -e 'using Pkg; Pkg.develop(PackageSpec(path = "/home/kcvaughn/Desktop/MOCNeutronTransport"))'
  julia -e 'using MOCNeutronTransport'
  echo "NOTE: You may need to source your ~/.bashrc in order for the Gmsh Julia API to work"
  echo "This may affect the next question."
  while true; do
    read -p "Do you wish to run tests to ensure the package was installed correctly? " yn
    case $yn in
      Yes ) 
        julia -e 'using Pkg; Pkg.activate("MOCNeutronTransport"); Pkg.test()'
        break;;
      No ) 
        break;;
      * ) 
        echo "Please answer Yes or No.";;
    esac
  done
  echo "Installation finished."
else
  echo "Use 'sudo ./install.sh'"
fi

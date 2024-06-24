find_library(GMSH_LIB "gmsh" REQUIRED HINTS $ENV{GMSH_ROOT}/lib)
find_path(GMSH_INC "gmsh.h" REQUIRED HINTS $ENV{GMSH_ROOT}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gmsh
  "Gmsh could not be found. Set GMSH_ROOT in your environment variables to manually specify the location."
  GMSH_LIB GMSH_INC)

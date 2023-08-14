## Update git submodules #########################
##################################################
include(cmake/update-git-submodules.cmake)

## System ########################################
##################################################
# No Windows support
if (WIN32)
  message(FATAL_ERROR "Windows is not supported")
endif()
# No Apple support
if (APPLE)
  message(FATAL_ERROR "Apple is not supported")
endif()

## Compiler ######################################
##################################################
# Check for gcc or clang
if (NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
         CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

## Standard ######################################
##################################################
set(UM2_CXX_STANDARD "20" CACHE STRING "C++ standard")
set_target_properties(um2 PROPERTIES CXX_STANDARD ${UM2_CXX_STANDARD})
set_target_properties(um2 PROPERTIES CXX_STANDARD_REQUIRED ON)

## OpenMP ########################################
##################################################
if (UM2_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(um2 PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

## CUDA ##########################################
##################################################
if (UM2_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  # nvcc will default to gcc and g++ if the host compiler is not set using CUDAHOSTCXX. 
  # To prevent unintentional version/compiler mismatches, we set the host compiler to the 
  # same compiler used to build the project.
  if (NOT DEFINED ENV{CUDAHOSTCXX})
    message(STATUS "Setting CMAKE_CUDA_HOST_COMPILER to ${CMAKE_CXX_COMPILER}." 
      " Consider setting the CUDAHOSTCXX environment variable if this is not desired.")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  set(UM2_CUDA_STANDARD "20" CACHE STRING "CUDA standard")
  macro(set_cuda_properties CUDA_TARGET)
    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_STANDARD ${UM2_CUDA_STANDARD})
    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${CUDA_TARGET} PROPERTIES CUDA_ARCHITECTURES native)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)    
  endmacro()
  set_cuda_properties(um2 ${UM2_SOURCES})
  target_include_directories(um2 SYSTEM PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")    
endif()

## hdf5 ##########################################
##################################################
find_library(HDF5_LIB "hdf5")
if (NOT HDF5_LIB)
  message(FATAL_ERROR "Could not find hdf5")
endif()
find_library(HDF5_CPP_LIB "hdf5_cpp")
if (NOT HDF5_CPP_LIB)
  message(FATAL_ERROR "Could not find hdf5_cpp")
endif()
find_path(HDF5_CPP_INC "H5Cpp.h")
if (NOT HDF5_CPP_INC)
  message(FATAL_ERROR "Could not find H5Cpp.h")
endif()
target_link_libraries(um2 PUBLIC "${HDF5_LIB}" "${HDF5_CPP_LIB}")
target_include_directories(um2 SYSTEM PUBLIC "${HDF5_CPP_INC}")

## pugixml #######################################
##################################################
find_library(PUGIXML_LIB "pugixml")
if (NOT PUGIXML_LIB)
  message(FATAL_ERROR "Could not find pugixml")
endif()
find_path(PUGIXML_INC "pugixml.hpp")
if (NOT PUGIXML_INC)
  message(FATAL_ERROR "Could not find pugixml.hpp")
endif()
target_link_libraries(um2 PUBLIC "${PUGIXML_LIB}")
target_include_directories(um2 SYSTEM PUBLIC "${PUGIXML_INC}")

## gmsh ##########################################
##################################################
if (UM2_ENABLE_GMSH)
  find_library(GMSH_LIB "gmsh")
  if (NOT GMSH_LIB)
    message(FATAL_ERROR "Could not find gmsh")
  endif()
  find_path(GMSH_INC "gmsh.h")
  if (NOT GMSH_INC)
    message(FATAL_ERROR "Could not find gmsh.h")
  endif()
  target_link_libraries(um2 PUBLIC "${GMSH_LIB}")
  target_include_directories(um2 SYSTEM PUBLIC "${GMSH_INC}")
endif()

### visualization #################################
###################################################
#if (UM2_ENABLE_VIS)
#  set(UM2_VIS_LIBRARIES
#    "OpenGL::GL"
#    "glfw"
#    "glad"
#    CACHE STRING "Visualization libraries")
#  # OpenGL
#  find_package(OpenGL REQUIRED)
#  # GLFW
#  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
#  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
#  add_subdirectory("${PROJECT_SOURCE_DIR}/dependencies/glfw" SYSTEM)
#  # GLAD
#  add_subdirectory("${PROJECT_SOURCE_DIR}/dependencies/glad" SYSTEM)
#  target_link_libraries(um2 PRIVATE ${UM2_VIS_LIBRARIES}) 
#endif()

## config.hpp ####################################
##################################################
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
  "${PROJECT_SOURCE_DIR}/include/um2/config.hpp")

## clang-format ##################################
##################################################
if (UM2_ENABLE_CLANG_FORMAT)
  include(cmake/clang-format.cmake)
endif()

## clang-tidy ####################################
##################################################
if (UM2_ENABLE_CLANG_TIDY)
  macro(set_clang_tidy_properties TIDY_TARGET)
    set_target_properties(${TIDY_TARGET} PROPERTIES
                          CXX_CLANG_TIDY
                          "clang-tidy;--extra-arg=-Wno-unknown-warning-option")
  endmacro()
  set_clang_tidy_properties(um2)
endif()

## cppcheck ######################################
##################################################
if (UM2_ENABLE_CPPCHECK)
  # Concatenate the cppcheck arguments into a single string
  string(CONCAT CPPCHECK_ARGS
    "cppcheck"
    ";--enable=warning,style,information,missingInclude"
    ";--std=c++20"
    ";--language=c++"
    ";--suppress=missingIncludeSystem"
    ";--suppress=invalidPrintfArgType_float"
    ";--suppress=unmatchedSuppression"
    ";--inconclusive"
    ";--inline-suppr"
    ";--error-exitcode=10")
  set_target_properties(um2 PROPERTIES CXX_CPPCHECK "${CPPCHECK_ARGS}")
endif()

## flags #########################################
##################################################
# Set compiler-specific warning flags 
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include(cmake/clang-cxx-flags.cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UM2_CLANG_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  include(cmake/gnu-cxx-flags.cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UM2_GNU_FLAGS}")
endif()

# Set fast math flags if enabled
if (UM2_ENABLE_FASTMATH)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
  if (UM2_ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
  endif()
endif()

# Set release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Set debug flags
if (!UM2_ENABLE_CUDA)
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()

# If OpenMP and parallel STL are enabled
if (UM2_ENABLE_OPENMP AND UM2_ENABLE_PAR_STL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_PARALLEL")
endif()

# If CUDA is enabled, pass the CXX flags via -Xcompiler
if (UM2_ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CMAKE_CXX_FLAGS}\"")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler \"${CMAKE_CXX_FLAGS_RELEASE}\"")
  set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler \"${CMAKE_CXX_FLAGS_DEBUG}\"")
  # If OpenMP is enabled, -fopenmp is passed via -Xcompiler
  if (UM2_ENABLE_OPENMP)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  endif()
endif()

# If COVERAGE is enabled, set the flags
if (UM2_ENABLE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0")
  target_link_libraries(um2 PUBLIC gcov)
endif ()

## Tests #########################################
##################################################
if (UM2_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

## Examples ###################################### 
##################################################
if (UM2_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

## Benchmarks #################################### 
##################################################
if (UM2_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

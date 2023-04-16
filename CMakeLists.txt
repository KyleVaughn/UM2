cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(
      FATAL_ERROR
      "In-source builds are not supported. "
  )
endif()

project(
    UM2
    VERSION 0.1.0
    DESCRIPTION "An unstructured mesh library for automated method of characteristic mesh generation"
    HOMEPAGE_URL "https://github.com/KyleVaughn/UM2"
    LANGUAGES CXX
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    #    include(CTest)

endif()

option(UM2_ENABLE_CUDA "Enable CUDA support" OFF)

add_subdirectory(src)

# CUDA is currently only supported via Clang
#if(UM2_ENABLE_CUDA)
#  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#    message(
#        FATAL_ERROR
#        "CUDA supported requires Clang as the C++ compiler."
#    )
#  endif()
#  #  include(Compiler/
#  #
#  #  set(CMAKE_CUDA_COMPILER ${CMAKE_CXX_COMPILER})
#  #  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
#  enable_language(CUDA)
#  set_source_files_properties(test.cpp PROPERTIES LANGUAGE CUDA)
#endif()

#add_library(
#    um2
#    source/UM2.cpp
#)

#if(NOT BUILD_SHARED_LIBS)
#  target_compile_definitions(UM2_UM2 PUBLIC UM2_STATIC_DEFINE)
#endif()
#
#set_target_properties(
#    UM2_UM2 PROPERTIES
#    CXX_VISIBILITY_PRESET hidden
#    VISIBILITY_INLINES_HIDDEN YES
#    VERSION "${PROJECT_VERSION}"
#    SOVERSION "${PROJECT_VERSION_MAJOR}"
#    EXPORT_NAME UM2
#    OUTPUT_NAME UM2
#)
#
#target_include_directories(
#    UM2_UM2 ${warning_guard}
#    PUBLIC
#    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
#)
#
#target_include_directories(
#    UM2_UM2 SYSTEM
#    PUBLIC
#    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
#)
##
##target_compile_features(UM2_UM2 PUBLIC cxx_std_23)
##
### ---- Install rules ----
##
##if(NOT CMAKE_SKIP_INSTALL_RULES)
##  include(cmake/install-rules.cmake)
##endif()
##
### ---- Examples ----
##
##if(PROJECT_IS_TOP_LEVEL)
##  option(BUILD_EXAMPLES "Build examples tree." "${UM2_DEVELOPER_MODE}")
##  if(BUILD_EXAMPLES)
##    add_subdirectory(example)
##  endif()
#endif()
#
#
#include(cmake/dev-mode.cmake)

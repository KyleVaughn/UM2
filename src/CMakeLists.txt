file(GLOB HEADERS CONFIGURE_DEPENDS "${UM2_SOURCE_DIR}/include/*")

add_library(um2 um2c.cpp ${HEADERS})
add_library(UM2::UM2_CXX ALIAS um2)

set_target_properties(um2 PROPERTIES CXX_CLANG_TIDY clang-tidy)

target_include_directories(um2 PUBLIC ../include)

# Don't use -std=c++23, use -std=c++2b instead.
# Change this for future compiler versions.
set(CMAKE_CXX23_STANDARD_COMPILE_OPTION "-std=c++2b")
set(CMAKE_CXX23_EXTENSION_COMPILE_OPTION "-std=gnu++2b")

target_compile_features(um2 PUBLIC cxx_std_23)

target_link_libraries(um2 PRIVATE Thrust)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADERS})

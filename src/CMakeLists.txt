file(GLOB HEADERS CONFIGURE_DEPENDS "${UM2_SOURCE_DIR}/include/*")

add_library(um2 um2c.cpp ${HEADERS})

target_include_directories(um2 PUBLIC ../include)

set(Thrust_DIR "${CMAKE_CURRENT_LIST_DIR}/../tpls/thrust/thrust/cmake")
message(STATUS "Thrust_DIR: ${Thrust_DIR}")
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)
target_link_libraries(um2 PRIVATE Thrust)

target_compile_features(um2 PUBLIC cxx_std_23)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADERS})

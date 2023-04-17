file(GLOB HEADERS CONFIGURE_DEPENDS "${UM2_SOURCE_DIR}/include/*")

add_library(um2 um2c.cpp ${HEADERS})

target_include_directories(um2 PUBLIC ../include)

target_compile_features(um2 PUBLIC cxx_std_23)

target_link_libraries(um2 PRIVATE Thrust)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADERS})

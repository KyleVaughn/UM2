include(${PROJECT_SOURCE_DIR}/cmake/add_um2_example.cmake)

add_um2_example(./vera/1/1a_make_cad_model.cpp)

if (UM2_ENABLE_VIS)
  add_um2_example(./visualization/window.cpp)
endif()

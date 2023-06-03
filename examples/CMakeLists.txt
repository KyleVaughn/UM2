include(${PROJECT_SOURCE_DIR}/cmake/add_um2_example.cmake)

if (UM2_ENABLE_VIS)
  add_um2_example(ex_window ./visualization/window.cpp)
endif()

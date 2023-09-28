include(${PROJECT_SOURCE_DIR}/cmake/add_um2_example.cmake)

#==============================================================================
# C5G7
#==============================================================================
add_um2_example(./c5g7/single_pin/c5g7_pin_make_cad_model.cpp)
add_um2_example(./c5g7/single_pin/c5g7_uo2_pin.cpp)

if (UM2_ENABLE_VIS)
  add_um2_example(./visualization/window.cpp)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/add_um2_example.cmake)

#==============================================================================
# VERA
#==============================================================================

add_um2_example(./vera/1/1a_make_cad_model.cpp)
add_um2_example(./vera/1/1a.cpp)
add_um2_example(./vera/2/2a_make_cad_model.cpp)
add_um2_example(./vera/2/2a.cpp)
add_um2_example(./vera/2/2a_nogap_make_cad_model.cpp)
add_um2_example(./vera/2/2a_nogap.cpp)

#==============================================================================
# C5G7
#==============================================================================
add_um2_example(./c5g7/single_pin/c5g7_pin_make_cad_model.cpp)
add_um2_example(./c5g7/single_pin/c5g7_uo2_pin.cpp)

if (UM2_ENABLE_VIS)
  add_um2_example(./visualization/window.cpp)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/add_um2_test.cmake)

add_subdirectory(./stdlib)
add_subdirectory(./common)
#add_subdirectory(./math)
#add_subdirectory(./geometry)
#add_subdirectory(./physics)
#add_subdirectory(./mesh)
#add_subdirectory(./mpact)
#add_subdirectory(./gmsh)
#add_subdirectory(./api)
###==============================================================================
### gmsh
###==============================================================================
##
##add_um2_test(./gmsh/base_gmsh_api.cpp)
##add_um2_test(./gmsh/gmsh_io.cpp)
##add_um2_test(./gmsh/gmsh_model.cpp)
##
###==============================================================================
### visualization
###==============================================================================
##
##add_um2_test(./visualization/Image2D.cpp)
##
#####if (UM2_ENABLE_VIS)
#####  add_um2_test(shader ./visualization/shader.cpp)
#####endif()
##
###==============================================================================
### api
###==============================================================================
##
### Copy the mesh files to the built test directory
##file(COPY ${PROJECT_SOURCE_DIR}/tests/api/api_mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
##
##add_um2_test(./api/um2_cpp_api.cpp)
##add_um2_test(./api/um2_c_api.cpp)

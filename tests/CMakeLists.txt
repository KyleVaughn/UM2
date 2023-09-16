include(${PROJECT_SOURCE_DIR}/cmake/add_um2_test.cmake)

#===============================================================================
# stdlib
#===============================================================================

add_um2_test(./stdlib/algorithm/copy.cpp)
add_um2_test(./stdlib/algorithm/fill.cpp)
add_um2_test(./stdlib/algorithm/sort.cpp)
add_um2_test(./stdlib/memory/addressof.cpp)
add_um2_test(./stdlib/memory/construct_at.cpp)
add_um2_test(./stdlib/numeric/iota.cpp)
add_um2_test(./stdlib/Vector.cpp)
add_um2_test(./stdlib/sto.cpp)
add_um2_test(./stdlib/String.cpp)

#===============================================================================
# common
#===============================================================================

add_um2_test(./common/Log.cpp)
add_um2_test(./common/ShortString.cpp)
add_um2_test(./common/Color.cpp)
add_um2_test(./common/permutation.cpp)
add_um2_test(./common/to_vecvec.cpp)
#add_um2_test(./common/histogram.cpp)

#===============================================================================
# math
#===============================================================================

add_um2_test(./math/morton.cpp)
add_um2_test(./math/Vec.cpp)
add_um2_test(./math/Mat.cpp)
add_um2_test(./math/stats.cpp)
if (UM2_USE_TBB)
  add_um2_test(./parallel/math/parallel_stats.cpp)
endif()
add_um2_test(./math/AngularQuadrature.cpp)

#===============================================================================
# geometry
#===============================================================================

add_um2_test(./geometry/Point.cpp)
add_um2_test(./geometry/morton_sort_points.cpp)
if (UM2_USE_TBB OR UM2_USE_CUDA)
  add_um2_test(./parallel/geometry/parallel_morton_sort_points.cpp)
endif()
add_um2_test(./geometry/AxisAlignedBox.cpp)
add_um2_test(./geometry/LineSegment.cpp)
add_um2_test(./geometry/Triangle.cpp)
add_um2_test(./geometry/Quadrilateral.cpp)
add_um2_test(./geometry/QuadraticSegment.cpp)
add_um2_test(./geometry/QuadraticTriangle.cpp)
add_um2_test(./geometry/QuadraticQuadrilateral.cpp)

#==============================================================================
# mesh
#==============================================================================

# Copy the mesh files to the built test directory
file(COPY ${PROJECT_SOURCE_DIR}/tests/mesh/mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_um2_test(./mesh/RegularGrid.cpp)
add_um2_test(./mesh/RegularPartition.cpp)
add_um2_test(./mesh/RectilinearGrid.cpp)
add_um2_test(./mesh/RectilinearPartition.cpp)
add_um2_test(./mesh/MeshFile.cpp)
add_um2_test(./mesh/TriMesh.cpp)
add_um2_test(./mesh/QuadMesh.cpp)
add_um2_test(./mesh/QuadraticTriMesh.cpp)
add_um2_test(./mesh/QuadraticQuadMesh.cpp)
add_um2_test(./mesh/io_abaqus.cpp)
add_um2_test(./mesh/io_xdmf.cpp)

#==============================================================================
# mpact
#==============================================================================

# Copy the mesh files to the built test directory
file(COPY ${PROJECT_SOURCE_DIR}/tests/mpact/mpact_mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_um2_test(./mpact/SpatialPartition.cpp)

##==============================================================================
## ray_casting
##==============================================================================


#==============================================================================
# gmsh
#==============================================================================

add_um2_test(./gmsh/base_gmsh_api.cpp)
add_um2_test(./gmsh/gmsh_io.cpp)
add_um2_test(./gmsh/gmsh_model.cpp)

#==============================================================================
# visualization
#==============================================================================

add_um2_test(./visualization/Image2D.cpp)

###if (UM2_ENABLE_VIS)
###  add_um2_test(shader ./visualization/shader.cpp)
###endif()

#==============================================================================
# api
#==============================================================================

# Copy the mesh files to the built test directory
file(COPY ${PROJECT_SOURCE_DIR}/tests/api/api_mesh_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_um2_test(./api/um2_cpp_api.cpp)
add_um2_test(./api/um2_c_api.cpp)

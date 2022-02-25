module MOCNeutronTransport

const path_to_gmsh_api = "/usr/local/lib/gmsh.jl"
const enable_visualization = true
const visualize_ray_tracing = false 

using AbstractTrees
using CUDA
using Logging
using HDF5
using LightXML
using LinearAlgebra
using StaticArrays
using Dates: now, format
using LoggingExtras: TransformerLogger, global_logger

import Base: +, -, *, /, ==, ≈, intersect, sort, sort!
import LinearAlgebra: ×, ⋅, norm, inv

include(path_to_gmsh_api)

include("SVector.jl")
include("./primitives/Edge.jl")
include("./primitives/Face.jl")
include("./primitives/Point.jl")
include("./primitives/LineSegment.jl")
include("./primitives/QuadraticSegment.jl")
include("./primitives/Hyperplane.jl")
include("./primitives/AABox.jl")
include("./primitives/Polygon.jl")
include("./primitives/QuadraticPolygon.jl")
#
#include("constants.jl")
#include("log.jl")


#include("operators.jl")
#include("./gmsh/gmsh_generate_rectangular_grid.jl")
#include("./gmsh/gmsh_group_preserving_fragment.jl")
#include("./gmsh/gmsh_overlay_rectangular_grid.jl")


###include("L_system.jl")
#include("./mesh/UnstructuredMesh.jl")
#include("./mesh/PolygonMesh.jl")
#include("./mesh/QuadraticPolygonMesh.jl")
#include("./mesh/HierarchicalMeshPartition.jl")
#include("./mesh/IO_abaqus.jl")
###include("./mesh/IO_vtk.jl")
#include("./mesh/IO_xdmf.jl")
#include("gauss_legendre_quadrature.jl")
##include("./ray_tracing/AngularQuadrature.jl")
##include("./raytracing/raytrace.jl")
##include("./ray_tracing/ray_trace_low_level.jl")

# SVector.jl
export distance, inv, norm², normalize
# Edge.jl
export Edge, Edge2D, Edge3D
# Face.jl
export Face, Face2D, Face3D
# Point.jl
export Point, Point1D, Point2D, Point3D, +, -, *, /, ⋅, ×, ==, ≈, distance,
       distance², midpoint, nan, norm, norm²
# LineSegment
export LineSegment, LineSegment2D, LineSegment3D
# QuadraticSegment
export  QuadraticSegment, QuadraticSegment2D, QuadraticSegment3D, jacobian,
        isstraight
# Hyperplane
export Hyperplane, Hyperplane2D, Hyperplane3D 
# AABox
export AABox, AABox2D, AABox3D, Δx, Δy, Δz
# Polygon
export Polygon, Triangle, Triangle2D, Triangle3D, Quadrilateral, Quadrilateral2D,
       Quadrilateral3D
# QuadraticPolygon
export QuadraticPolygon, QuadraticTriangle, QuadraticTriangle2D, QuadraticTriangle3D,
       QuadraticQuadrilateral, QuadraticQuadrilateral2D, QuadraticQuadrilateral3D

# Structs/Types
#export 
#        HierarchicalMeshPartition, 
#       MeshPartitionTree,
#        PolygonMesh, 
#        QuadraticPolygonMesh,
#       QuadraticTriangleMesh, QuadraticQuadrilateralMesh, 
#       Tree,  TriangleMesh,
#       UnstructuredMesh, UnstructuredMesh2D, UnstructuredMesh3D

# Convenience operators
#const 𝗗 = derivative
#const ∇ = gradient
#const ∇² = laplacian
#const 𝗝 = jacobian

# Operators
#export 𝗝

# Methods
#export arclength, area, 
#       boundingbox, 
#       centroid, 
#       depth, derivative, distance, distance², 
#       edgepoints, edges, edge_face_connectivity, 
#       facepoints, face_edge_connectivity,
#       gauss_legendre_quadrature, 
#       height, 
#       intersect, intersect_edges, intersect_edges_CUDA, inv, isleft, isstraight, isroot, 
#       in_halfspace,
#       jacobian, 
#       leaves, linear_edges, log_timestamps, 
#       materialize_edge, materialize_edges, materialize_face, materialize_faces,
#       materialize_polygon, materialize_quadratic_polygon, midpoint, 
#       nan, nearest_point, norm, norm², num_edges, 
#       partition_mesh,
#       quadratic_edges,
#       rand, read_abaqus2d, real_to_parametric, 
#       sort, sort!, sort_intersection_points!, submesh,
#       triangulate, triangulate_nonconvex,
#       union, 
#       width, write_xdmf2d
#
## Gmsh
#export gmsh,
#       gmsh_generate_rectangular_grid,
#       gmsh_group_preserving_fragment,
#       gmsh_overlay_rectangular_grid
# 
## Plot
#if enable_visualization
#    export Figure, Axis, Axis3
#    export scatter, linesegments, mesh,
#           scatter!, linesegments!, mesh!
#end
#if enable_visualization 
#    using GLMakie: Axis, Axis3, Figure, LineSegments, Mesh, Scatter, current_axis, record
#    import GLMakie: linesegments, linesegments!, mesh, mesh!, scatter, scatter!, 
#                    convert_arguments
#end

#
end

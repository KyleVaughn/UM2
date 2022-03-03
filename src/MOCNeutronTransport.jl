module MOCNeutronTransport

const minimum_ray_segment_length = 1e-4 # 1μm
const path_to_gmsh_api = "/usr/local/lib/gmsh.jl"
const enable_visualization = true
const visualize_ray_tracing = false 

using CUDA
using Logging
using HDF5
using LightXML
using LinearAlgebra
using StaticArrays
using Dates: now, format
using LoggingExtras: TransformerLogger, global_logger

import Base: +, -, *, /, ==, ≈, intersect, sort, sort!, split
import LinearAlgebra: ×, ⋅, norm, inv

include(path_to_gmsh_api)

include("log.jl")
include("SVector.jl")
include("primitives/Edge.jl")
include("primitives/Face.jl")
include("primitives/Cell.jl")
include("primitives/Point.jl")
include("primitives/LineSegment.jl")
include("primitives/QuadraticSegment.jl")
include("primitives/Hyperplane.jl")
include("primitives/AABox.jl")
include("primitives/ConvexPolygon.jl")
include("primitives/QuadraticPolygon.jl")
include("primitives/ConvexPolyhedron.jl")
include("primitives/QuadraticPolyhedron.jl")
include("mesh/UnstructuredMesh.jl")
include("mesh/ConvexPolygonMesh.jl")
include("mesh/QuadraticPolygonMesh.jl")
include("MPACT/MPACTCoarseCell.jl")
include("MPACT/MPACTRayTracingModule.jl")
include("MPACT/MPACTLattice.jl")
include("MPACT/MPACTCore2D.jl")

# gmsh
include("rand.jl")
include("interpolation.jl")
include("jacobian.jl")
include("gauss_legendre_quadrature.jl")
include("triangulate.jl")
include("measure.jl")
# only need to worry about dampening for intersection with 
# quadratic faces in 3D
#
# TODO: centroid.jl, boundingbox.jl, cartesian_to_parametric.jl

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

# log
export log_timestamps
# constants
export minimum_ray_segment_length
# SVector
export distance, inv, norm², normalize
# Edge
export Edge, Edge2D, Edge3D
# Face
export Face, Face2D, Face3D
# Cell
export Cell
# Point
export Point, Point1D, Point2D, Point3D, +, -, *, /, ⋅, ×, ==, ≈, distance,
       distance², midpoint, nan, norm, norm²
# LineSegment
export LineSegment, LineSegment2D, LineSegment3D
# QuadraticSegment
export QuadraticSegment, QuadraticSegment2D, QuadraticSegment3D, isstraight
# Hyperplane
export Hyperplane, Hyperplane2D, Hyperplane3D 
# AABox
export AABox, AABox2D, AABox3D, Δx, Δy, Δz, split
# ConvexPolygon
export ConvexPolygon, Triangle, Triangle2D, Triangle3D, Quadrilateral, Quadrilateral2D,
       Quadrilateral3D
# QuadraticPolygon
export QuadraticPolygon, QuadraticTriangle, QuadraticTriangle2D, QuadraticTriangle3D,
       QuadraticQuadrilateral, QuadraticQuadrilateral2D, QuadraticQuadrilateral3D
# ConvexPolyhedron
export ConvexPolyhedron, Tetrahedron, Hexahedron
# QuadraticPolyhedron
export QuadraticPolyhedron, QuadraticTetrahedron, QuadraticHexahedron
# UnstructuredMesh
export UnstructuredMesh, UnstructuredMesh2D, UnstructuredMesh3D, 
       LinearUnstructuredMesh, LinearUnstructuredMesh2D, LinearUnstructuredMesh3D,
       QuadraticUnstructuredMesh, QuadraticUnstructuredMesh2D, 
       QuadraticUnstructuredMesh3D 
# ConvexPolygonMesh
export ConvexPolygonMesh, TriangleMesh, QuadrilateralMesh
# QuadraticPolygonMesh
export QuadraticPolygonMesh, QuadraticTriangleMesh, QuadraticQuadrilateralMesh
# Gmsh
export gmsh
# MPACTCoarseCell
export MPACTCoarseCell, MPACTCoarseCells
# MPACTRayTracingModule
export MPACTRayTracingModule, MPACTRayTracingModules
# MPACTLattice
export MPACTLattice, MPACTLattices
# MPACTCore2D
export MPACTCore2D, validate_core
# jacobian
const 𝗝 = jacobian
export jacobian, 𝗝
# gauss_legendre_quadrature
export gauss_legendre_quadrature, triangular_gauss_legendre_quadrature
# triangulate
export triangulate
# measure
export measure

# Structs/Types
#export 
#        HierarchicalMeshPartition, 
#       MeshPartitionTree,
#        PolygonMesh, 
#        QuadraticPolygonMesh,
#       QuadraticTriangleMesh, QuadraticQuadrilateralMesh, 
#       Tree,  TriangleMesh,
#       

# Convenience operators
#const 𝗗 = derivative
#const ∇ = gradient
#const ∇² = laplacian
#const 𝗝 = jacobian

# Operators
#export 𝗝

# Methods
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


#       gmsh_generate_rectangular_grid,
#       gmsh_group_preserving_fragment,
#       gmsh_overlay_rectangular_grid
# 
# Plot
if enable_visualization
    using GLMakie: Axis, Axis3, Figure, LineSegments, Mesh, Scatter, current_axis, 
                   record
    import GLMakie: linesegments, linesegments!, mesh, mesh!, scatter, scatter!, 
                    convert_arguments
    include("plot/Point.jl")
    include("plot/LineSegment.jl")
    include("plot/QuadraticSegment.jl")
    include("plot/ConvexPolygon.jl")
    include("plot/AABox.jl")
    include("plot/QuadraticPolygon.jl")

    export Figure, Axis, Axis3
    export scatter, linesegments, mesh,
           scatter!, linesegments!, mesh!
end

#
end

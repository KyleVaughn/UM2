module MOCNeutronTransport

const minimum_ray_segment_length = 1e-4 # 1μm
const plot_nonlinear_subdivisions = 2
const enable_visualization = false
const visualize_ray_tracing = false 

using CUDA
using Logging
using HDF5
using LightXML
using LinearAlgebra
using StaticArrays
using LinearAlgebra
using Dates: now, format
using LoggingExtras: TransformerLogger, global_logger

import Base: +, -, *, /, ==, ≈, convert, hypot, intersect, issubset, sort, 
             sort!, zero
import LinearAlgebra: ×, ⋅, norm, inv

# Include gmsh
for path in Base.load_path()
    gmsh_api_path = path*"/gmsh.jl" 
    if isfile(gmsh_api_path)
        include(gmsh_api_path)
        break
    end
end
if !@isdefined(gmsh)
    @warn "Gmsh API was not found"
end

include("log.jl")
include("SVector.jl")
include("Material.jl")
include("primitives/Edge.jl")
include("primitives/Face.jl")
include("primitives/Cell.jl")
include("primitives/Point.jl")
include("primitives/LineSegment.jl")
include("primitives/QuadraticSegment.jl")
include("primitives/Hyperplane.jl")
include("primitives/AABox.jl")
include("primitives/Polygon.jl")
include("primitives/QuadraticPolygon.jl")
include("primitives/Polyhedron.jl")
include("primitives/QuadraticPolyhedron.jl")
include("mesh/RectilinearGrid.jl")
include("mesh/UnstructuredMesh.jl")
include("mesh/PolygonMesh.jl")
include("mesh/QuadraticPolygonMesh.jl")
include("mesh/PolyhedronMesh.jl")
include("mesh/QuadraticPolyhedronMesh.jl")
#include("mesh/IO_abaqus.jl")
#include("mesh/mesh_IO.jl")
include("MPACT/MPACTGridHierarchy.jl")
#include("MPACT/MPACTGeometryHierarchy.jl")
#include("MPACT/MPACTCoarseCell.jl")
#include("MPACT/MPACTRayTracingModule.jl")
#include("MPACT/MPACTLattice.jl")
#include("MPACT/MPACTCore2D.jl")
#
include("gmsh_extensions/model/add_cad_entity_names_to_physical_groups.jl")
include("gmsh_extensions/model/get_entities_by_color.jl")
include("gmsh_extensions/model/add_materials_to_physical_groups_by_color.jl")
include("gmsh_extensions/model/import_model.jl")
include("gmsh_extensions/model/physical_group_preserving_fragment.jl")
#include("gmsh_extensions/overlay_mpact_grid_hierarchy.jl")




# gmsh
#include("rand.jl")
#include("interpolation.jl")
#include("jacobian.jl")
##include("boundingbox.jl")
#include("gauss_legendre_quadrature.jl")
#include("triangulate.jl")
#include("measure.jl")
# only need to worry about dampening for intersection with 
# quadratic faces in 3D
#

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
# SVector
export distance, inv, norm², hypot
# Material
export Material
# Edge
export Edge, Edge2D, Edge3D
# Face
export Face, Face2D, Face3D
# Cell
export Cell
# Point
export Point, Point1D, Point2D, Point3D, +, -, *, /, ⋅, ×, ==, ≈, distance,
       distance², isCCW, midpoint, nan, norm, norm²
# LineSegment
export LineSegment, LineSegment2D, LineSegment3D
# QuadraticSegment
export QuadraticSegment, QuadraticSegment2D, QuadraticSegment3D, isstraight
# Hyperplane
export Hyperplane, Hyperplane2D, Hyperplane3D 
# AABox
export AABox, AABox2D, AABox3D, Δx, Δy, Δz
# Polygon
export Polygon, Polygon2D, Polygon3D, Triangle, Triangle2D, Triangle3D, Quadrilateral, 
       Quadrilateral2D, Quadrilateral3D
# QuadraticPolygon
export QuadraticPolygon, QuadraticTriangle, QuadraticTriangle2D, QuadraticTriangle3D,
       QuadraticQuadrilateral, QuadraticQuadrilateral2D, QuadraticQuadrilateral3D
# Polyhedron
export Polyhedron, Tetrahedron, Hexahedron
# QuadraticPolyhedron
export QuadraticPolyhedron, QuadraticTetrahedron, QuadraticHexahedron
# UnstructuredMesh
export UnstructuredMesh, UnstructuredMesh2D, UnstructuredMesh3D, 
       LinearUnstructuredMesh, LinearUnstructuredMesh2D, LinearUnstructuredMesh3D,
       QuadraticUnstructuredMesh, QuadraticUnstructuredMesh2D, 
       QuadraticUnstructuredMesh3D 
# PolygonMesh
export PolygonMesh, TriangleMesh, QuadrilateralMesh
# QuadraticPolygonMesh
export QuadraticPolygonMesh, QuadraticTriangleMesh, QuadraticQuadrilateralMesh
# RectilinearGrid
export RectilinearGrid, RectilinearGrid2D, issubset
# MPACTGridHierarchy
export MPACTGridHierarchy
## mesh_IO
#export import_mesh
# gmsh
export gmsh
# add_cad_entity_names_to_physical_groups
export add_cad_entity_names_to_physical_groups
# get_entities_by_color
export get_entities_by_color
# add_materials_to_physical_groups_by_color
export add_materials_to_physical_groups_by_color
# import_model
export import_model



## MPACTCoarseCell
#export MPACTCoarseCell, MPACTCoarseCells
## MPACTRayTracingModule
#export MPACTRayTracingModule, MPACTRayTracingModules
## MPACTLattice
#export MPACTLattice, MPACTLattices
## MPACTCore2D
#export MPACTCore2D, validate_core_partition
## jacobian
#const 𝗝 = jacobian
#export jacobian, 𝗝
## gauss_legendre_quadrature
#export gauss_legendre_quadrature, triangular_gauss_legendre_quadrature
## triangulate
#export triangulate
## measure
#export measure

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
    include("plot/Polygon.jl")
    include("plot/AABox.jl")
    include("plot/QuadraticPolygon.jl")
    export Figure, Axis, Axis3
    export scatter, linesegments, mesh,
           scatter!, linesegments!, mesh!
end

#
end

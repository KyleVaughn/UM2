module MOCNeutronTransport

#const minimum_ray_segment_length = 1e-4 # 1μm
#const visualization_enabled = true
#const visualize_ray_tracing = false 
#const plot_nonlinear_subdivisions = 3
#
#using CUDA, Colors, FixedPointNumbers, HDF5, Logging, LightXML, LinearAlgebra, 
#      StaticArrays, Statistics
using Reexport
using Pkg.Artifacts
#using Dates: now, format
#using LoggingExtras: TransformerLogger, global_logger
#
#import Base: +, -, *, /, ==, ≈, convert, hypot, intersect, issubset, sort, 
#             sort!, zero
#import LinearAlgebra: ×, ⋅, norm, inv

@reexport using LinearAlgebra
@reexport using StaticArrays

# include gmsh
# Check if there is a local install on JULIA_LOAD_PATH
for path in Base.load_path()
    gmsh_jl = joinpath(path, "gmsh.jl")
    if isfile(gmsh_jl)
        @info "MOCNeutronTransport is using the gmsh API found at: "*gmsh_jl
        include(gmsh_jl)
        break
    end
end
# Fallback on the SDK
if !@isdefined(gmsh)
    gmsh_dir = readdir(artifact"gmsh", join=true)[1]
    gmsh_jl = joinpath(gmsh_dir, "lib", "gmsh.jl") 
    if isfile(gmsh_jl)
        @info "MOCNeutronTransport is using the gmsh API found at: "*gmsh_jl
        include(gmsh_jl)
    else
        error("Could not find gmsh API.")
    end
end


include("quadrature/Quadrature.jl")
@reexport using .Quadrature
include("geometry/Geometry.jl")
@reexport using .Geometry
#include("Material.jl")
#include("log.jl")
#
#include("mesh/RectilinearGrid.jl")
#include("mesh/UnstructuredMesh.jl")
#include("mesh/PolygonMesh.jl")
#include("mesh/QuadraticPolygonMesh.jl")
#include("mesh/PolyhedronMesh.jl")
#include("mesh/QuadraticPolyhedronMesh.jl")
#include("mesh/submesh.jl")
#include("mesh/edges.jl")
#include("mesh/io_abaqus.jl")
#include("mesh/mesh_io.jl")
#include("mesh/materialize.jl")
#
#include("MPACT/MPACTGridHierarchy.jl")
#
#include("gmsh_extensions/model/add_physical_group.jl")
#include("gmsh_extensions/model/add_cad_names_to_physical_groups.jl")
#include("gmsh_extensions/model/get_entities_by_color.jl")
#include("gmsh_extensions/model/color_material_physical_group_entities.jl")
#include("gmsh_extensions/model/add_materials_to_physical_groups_by_color.jl")
#include("gmsh_extensions/model/import_model.jl")
#include("gmsh_extensions/model/physical_group_preserving_fragment.jl")
#include("gmsh_extensions/model/overlay_mpact_grid_hierarchy.jl")
#include("gmsh_extensions/mesh/set_mesh_field_using_materials.jl")
#include("gmsh_extensions/mesh/generate_mesh.jl")
#include("gmsh_extensions/mesh/get_cad_to_mesh_error.jl")
#
## Material
#export Material
## linalg
#export ⊙, ⊘, inv, norm²
## log
#export add_timestamps_to_logger
## Edge
#export Edge, Edge2D, Edge3D
## Face
#export Face, Face2D, Face3D
## Cell
#export Cell
## Point
#export Point, Point1D, Point2D, Point3D, +, -, *, /, ⋅, ×, ⊙, ⊘, ==, ≈, distance,
#       distance², isCCW, midpoint, nan, norm, norm²
## LineSegment
#export LineSegment, LineSegment2D, LineSegment3D
## QuadraticSegment
#export QuadraticSegment, QuadraticSegment2D, QuadraticSegment3D, isstraight
## Hyperplane
#export Hyperplane, Hyperplane2D, Hyperplane3D 
## AABox
#export AABox, AABox2D, AABox3D, Δx, Δy, Δz
## Polygon
#export Polygon, Polygon2D, Polygon3D, Triangle, Triangle2D, Triangle3D, Quadrilateral, 
#       Quadrilateral2D, Quadrilateral3D
## QuadraticPolygon
#export QuadraticPolygon, QuadraticTriangle, QuadraticTriangle2D, QuadraticTriangle3D,
#       QuadraticQuadrilateral, QuadraticQuadrilateral2D, QuadraticQuadrilateral3D
## Polyhedron
#export Polyhedron, Tetrahedron, Hexahedron
## QuadraticPolyhedron
#export QuadraticPolyhedron, QuadraticTetrahedron, QuadraticHexahedron
## triangulate
#export triangulate
## measure
#export measure
#
#
## RectilinearGrid
#export RectilinearGrid, RectilinearGrid2D, issubset
## UnstructuredMesh
#export UnstructuredMesh, UnstructuredMesh2D, UnstructuredMesh3D, 
#       LinearUnstructuredMesh, LinearUnstructuredMesh2D, LinearUnstructuredMesh3D,
#       QuadraticUnstructuredMesh, QuadraticUnstructuredMesh2D, 
#       QuadraticUnstructuredMesh3D 
## PolygonMesh
#export PolygonMesh, TriangleMesh, QuadrilateralMesh, MixedPolygonMesh
## QuadraticPolygonMesh
#export QuadraticPolygonMesh, QuadraticTriangleMesh, QuadraticQuadrilateralMesh,
#       MixedQuadraticPolygonMesh
## edges
#export edges
## submesh
#export submesh
## mesh_io
#export import_mesh
## materialize
#export getpoints, materialize_edge, materialize_edges, materialize_face, materialize_faces
#
## MPACTGridHierarchy
#export MPACTGridHierarchy
#
## gmsh
#export gmsh
## add_physical_group
#export add_physical_group
## add_cad_names_to_physical_groups
#export add_cad_names_to_physical_groups
## get_entities_by_color
#export get_entities_by_color
## color_material_physical_group_entities
#export color_material_physical_group_entities
## add_materials_to_physical_groups_by_color
#export add_materials_to_physical_groups_by_color
## import_model
#export import_model
## physical_group_preserving_fragment
#export physical_group_preserving_fragment
## overlay_mpact_grid_hierarchy
#export overlay_mpact_grid_hierarchy
## set_mesh_field_using_materials
#export set_mesh_field_using_materials
## generate_mesh
#export generate_mesh
## get_cad_to_mesh_error 
#export get_cad_to_mesh_error 
#
## Plot
#if visualization_enabled
#    using GLMakie: Axis, Axis3, Figure, LineSegments, Mesh, Scatter, current_axis, 
#                   record, hist, hist!
#    import GLMakie: linesegments, linesegments!, mesh, mesh!, scatter, scatter!, 
#                    convert_arguments
#    include("plot/Point.jl")
#    include("plot/LineSegment.jl")
#    include("plot/QuadraticSegment.jl")
#    include("plot/AABox.jl")
#    include("plot/Polygon.jl")
#    include("plot/QuadraticPolygon.jl")
#    include("plot/UnstructuredMesh.jl")
#    export Figure, Axis, Axis3
#    export hist, scatter, linesegments, mesh,
#           hist!, scatter!, linesegments!, mesh!
#end

end

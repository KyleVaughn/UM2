module MOCNeutronTransport
using ColorSchemes
using HDF5
using LightXML
using LinearAlgebra
using StaticArrays
using Dates: now, format
using GLMakie: Axis, Figure, LineSegments, Mesh, Scatter
using LoggingExtras: TransformerLogger, global_logger
try
    # Use local gmsh install
    using gmsh
catch e
    # Fall back on Gmsh package
    @warn "Using Gmsh package instead of install from source"
    using Gmsh: gmsh
end

# Make logger give time stamps
const date_format = "yyyy-mm-dd HH:MM:SS"
timestamp_logger(logger) = TransformerLogger(logger) do log
  merge(log, (; message = "$(format(now(), date_format)) $(log.message)"))
end

function log_timestamps()
    logger = global_logger()
    logger |> timestamp_logger |> global_logger
end

import Base: +, -, *, /, ≈, ==, intersect, in
import GLMakie: linesegments!, mesh!, scatter!, convert_arguments
include("AbstractTypes.jl")
include("AngularQuadrature.jl")
include("Tree.jl")
include("Point_2D.jl")
include("LineSegment_2D.jl")
include("QuadraticSegment_2D.jl")
include("Triangle_2D.jl")
include("Quadrilateral_2D.jl")
include("Triangle6_2D.jl")
include("Quadrilateral8_2D.jl")
include("constants.jl")
include("gauss_legendre_quadrature.jl")
include("UnstructuredMesh_2D.jl")
include("UnstructuredMesh_2D-low_level_functions.jl")
include("HierarchicalRectangularlyPartitionedMesh.jl")
include("abaqus.jl")
include("ray_trace.jl")
include("vtk.jl")
include("xdmf.jl")

include("gmsh_rectangular_grid.jl")
include("gmsh_group_preserving_fragment.jl")
include("gmsh_overlay_rectangular_grid.jl")

# Structs/Types
export  AngularQuadrature,
        Edge_2D,
        Face_2D,
        HierarchicalRectangularlyPartitionedMesh,
        LineSegment_2D,
        Point_2D,
        ProductAngularQuadrature,
        QuadraticSegment_2D,
        Quadrilateral_2D,
        Quadrilateral8_2D,
        Triangle_2D,
        Triangle6_2D,
        UnstructuredMesh_2D,
        Tree
# Functions
export  ×,
        ⋅,
        AABB,
        add_boundary_edges,
        add_connectivity,
        add_edges,
        add_edges_materialized,
        add_everything,
        add_faces_materialized,
        add_edge_face_connectivity,
        add_face_edge_connectivity,
        angular_quadrature,
        arc_length,
        area,
        boundary_edges,
        derivative,
        distance,
        edges,
        edges_materialized,
        edge_face_connectivity,
        edge_materialized,        
        faces_sharing_vertex,
        face_edge_connectivity,
        find_face,
        find_segment_faces,
        gauss_legendre_quadrature,
        get_adjacent_faces,
        get_edge_points,
        get_level,
        get_shared_edge,
        height,
        intersect,
        intersect_edges,
        intersect_faces,
        intersect_iterative,
        jacobian,
        levels,
        log_timestamps,
        materialize_edge,
        materialize_face,
        midpoint,
        next_edge_and_face_explicit,
        next_face_fallback_explicit,
        norm,
        partition_rectangularly,
        ray_trace,
        ray_trace_edge_to_edge,
        ray_trace_edge_to_edge_explicit!,
        read_abaqus_2d,
        read_vtk_2d,
        real_to_parametric,
        segmentize,
        submesh,
        tracks,
        triangulate,
        width,
        write_vtk_2d,
        write_xdmf_2d

# Gmsh
export gmsh,
       gmsh_rectangular_grid,
       gmsh_group_preserving_fragment,
       gmsh_overlay_rectangular_grid

# Plot
export Figure, Axis
export scatter, linesegments, mesh,
       scatter!, linesegments!, mesh!

end # module

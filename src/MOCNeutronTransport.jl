module MOCNeutronTransport

# Compilation options
const path_to_gmsh_api = "/usr/local/lib/gmsh.jl"
const enable_visualization = false
const visualize_ray_tracing = false 

# using
using ColorSchemes
using Logging
using HDF5
using LightXML
using LinearAlgebra
using StaticArrays
using Dates: now, format
using LoggingExtras: TransformerLogger, global_logger

# import
import Base: @propagate_inbounds
import Base: broadcastable, getindex, getproperty, +, -, *, /, in, intersect, isapprox, union

# Optional compilation/local dependencies
if enable_visualization 
    using GLMakie: Axis, Figure, LineSegments, Mesh, Scatter, current_axis, record
    import GLMakie: linesegments!, mesh!, scatter!, convert_arguments
    include(path_to_gmsh_api)
end

# Setup logger to have time stamps 
const date_format = "HH:MM:SS.sss"
timestamp_logger(logger) = TransformerLogger(logger) do log
  merge(log, (; message = "$(format(now(), date_format)) $(log.message)"))
end

MOCNeutronTransport_timestamps_on = false
function log_timestamps()
    if !MOCNeutronTransport_timestamps_on
        logger = global_logger()
        logger |> timestamp_logger |> global_logger
        global MOCNeutronTransport_timestamps_on = true
    end
end

include("Tree.jl")
# include("constants.jl")
#include("operators.jl")
#include("./gmsh/gmsh_generate_rectangular_grid.jl")
#include("./gmsh/gmsh_group_preserving_fragment.jl")
#include("./gmsh/gmsh_overlay_rectangular_grid.jl")
#include("./primitives/Edge_2D.jl")
#include("./primitives/Face_2D.jl")
include("./primitives/Point_2D.jl")
#include("./primitives/LineSegment_2D.jl")
#include("./primitives/Rectangle_2D.jl")
#include("./primitives/QuadraticSegment_2D.jl")
#include("./primitives/Triangle_2D.jl")
#include("./primitives/Quadrilateral_2D.jl")
#include("./primitives/Triangle6_2D.jl")
#include("./primitives/Quadrilateral8_2D.jl")
#include("L_system.jl")
#include("./mesh/UnstructuredMesh_2D.jl")
#include("./mesh/LinearUnstructuredMesh_2D.jl")
#include("./mesh/QuadraticUnstructuredMesh_2D.jl")
#include("./mesh/HierarchicalRectangularlyPartitionedMesh.jl")
#include("./mesh/IO_abaqus.jl")
##include("./mesh/IO_vtk.jl")
#include("./mesh/IO_xdmf.jl")
#include("gauss_legendre_quadrature.jl")
#include("./ray_tracing/AngularQuadrature.jl")
#include("./ray_tracing/ray_trace.jl")
#include("./ray_tracing/ray_trace_low_level.jl")
#
#
# Convenience operators
#const ∇ = gradient
#const ∇² = laplacian
#const J = jacobian
const × = cross
const ⋅ = dot



# Structs/Types
export  Edge_2D,
        Face_2D,
        LinearUnstructuredMesh_2D,
        LineSegment_2D,
        Point_2D,
        QuadraticSegment_2D,
        QuadraticUnstructuredMesh_2D,
        Quadrilateral_2D,
        Quadrilateral8_2D,
        QuadrilateralMesh_2D,
        Rectangle_2D,
        Tree,
        Triangle_2D,
        Triangle6_2D,
        TriangleMesh_2D,
        UnstructuredMesh_2D
# Functions
export  +, -, *, /, ×, ⋅, ⪇ , ⪉ , ∇ , ∇²,
        add_boundary_edges,
        add_boundary_edges!,
        add_connectivity,
        add_connectivity!,
        add_edges,
        add_edges!,
        add_everything,
        add_everything!,
        add_edge_face_connectivity,
        add_face_edge_connectivity,
        add_materialized_edges,
        add_materialized_edges!,
        add_materialized_faces,
        add_materialized_faces!,
        arclength,
        area,
        boundary_edges,
        boundingbox,
        centroid,
        closest_point,
        cross,
        distance,
        distance²,
        dot,
        edges,
        edge_face_connectivity,
        edge_points,
        faces_sharing_vertex,
        face_edge_connectivity,
        face_points,
        find_face,
        find_segment_faces,
        generate_angular_quadrature,
        generate_tracks,
        height,
        hilbert_curve,
        intersect,
        intersect_edges,
        intersect_faces,
        isleft,
        isstraight,
        jacobian,
        J,
        log_timestamps,
        materialize_edge,
        materialize_edges,
        materialize_face,
        materialize_faces,
        midpoint,
        norm,
        norm²,
        partition_rectangularly,
        plot_track_edge_to_edge,
        ray_trace,
        ray_trace_edge_to_edge,
        ray_trace_angle_edge_to_edge!,
        ray_trace_track_edge_to_edge,
        read_abaqus_2d,
        read_vtk_2d,
        real_to_parametric,
        reorder_faces_to_hilbert!,
        reorder_points_to_hilbert!,
        reorder_to_hilbert!,
        segmentize,
        shared_edge,
        sortpoints,
        sortpoints!,
        sort_intersection_points!,
        submesh,
        to_lines,
        triangulate,
        union,
        validate_ray_tracing_data,
        width,
        write_vtk_2d,
        write_xdmf_2d

# Gmsh
export gmsh,
       gmsh_generate_rectangular_grid,
       gmsh_group_preserving_fragment,
       gmsh_overlay_rectangular_grid

# Plot
if enable_visualization
    export Figure, Axis
    export scatter, linesegments, mesh,
           scatter!, linesegments!, mesh!
end

end # module

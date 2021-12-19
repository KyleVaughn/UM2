module MOCNeutronTransport
# Compilation options
const enable_local_gmsh = true
const enable_visualization = false 
const visualize_ray_tracing = false 
# using
using ColorSchemes
using Logging
if enable_local_gmsh
    # Use local gmsh install
    # Temporarily turn off warnings, since gmsh isn't in dependencies
    try
        Logging.disable_logging(Logging.Error)
        using gmsh
        Logging.disable_logging(Logging.Debug)
        @info "MOCNeutronTransport is using the locally installed gmsh API instead of the Gmsh package"
    catch
        Logging.disable_logging(Logging.Debug)
        @warn "MOCNeutronTransport is using the Gmsh package instead of the locally installed gmsh API"
        using Gmsh: gmsh
    end
else
    # Fallback on Gmsh package
    @warn "MOCNeutronTransport is using the Gmsh package instead of the locally installed gmsh API"
    using Gmsh: gmsh
end
using HDF5
using LightXML
using LinearAlgebra
using StaticArrays
using Dates: now, format
if enable_visualization 
    using GLMakie: Axis, Figure, LineSegments, Mesh, Scatter, current_axis, record
end
using LoggingExtras: TransformerLogger, global_logger

# import
import Base: +, -, *, /, ≈, ≉, ==, intersect, in
if enable_visualization 
    import GLMakie: linesegments!, mesh!, scatter!, convert_arguments
end

# logging
# Make logger give time stamps
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


include("AbstractTypes.jl")
#include("Tree.jl")
#include("operators.jl")
#include("./gmsh/gmsh_generate_rectangular_grid.jl")
#include("./gmsh/gmsh_group_preserving_fragment.jl")
#include("./gmsh/gmsh_overlay_rectangular_grid.jl")
include("./primitives/Point_2D.jl")
include("./primitives/LineSegment_2D.jl")
include("./primitives/QuadraticSegment_2D.jl")
include("./primitives/Triangle_2D.jl")
include("./primitives/Quadrilateral_2D.jl")
include("./primitives/Triangle6_2D.jl")
include("./primitives/Quadrilateral8_2D.jl")
#include("L_system.jl")
include("./mesh/UnstructuredMesh_2D.jl")
include("./mesh/LinearUnstructuredMesh_2D.jl")
#include("./mesh/UnstructuredMesh_2D_low_level.jl")
#include("./mesh/HierarchicalRectangularlyPartitionedMesh.jl")
#include("./mesh/IO_abaqus.jl")
#include("./mesh/IO_vtk.jl")
#include("./mesh/IO_xdmf.jl")
include("constants.jl")
include("gauss_legendre_quadrature.jl")
#include("./ray_tracing/AngularQuadrature.jl")
#include("./ray_tracing/ray_trace.jl")
#include("./ray_tracing/ray_trace_low_level.jl")

# Structs/Types
export  AngularQuadrature,
        ProductAngularQuadrature,
        Edge_2D,
        Face_2D,
        UnstructuredMesh_2D,
        Point_2D,
        LineSegment_2D,
        QuadraticSegment_2D,
        Triangle_2D,
        Quadrilateral_2D,
        Triangle6_2D,
        Quadrilateral8_2D,
        LinearTriangleMesh_2D,
        HierarchicalRectangularlyPartitionedMesh,
        Tree
# Functions
export  ×,
        ⋅,
        ⪇ ,
        ⪉ ,
        add_boundary_edges,
        add_connectivity,
        add_edges,
        add_everything,
        add_edge_face_connectivity,
        add_face_edge_connectivity,
        add_materialized_edges,
        add_materialized_faces,
        adjacent_faces,
        arc_length,
        area,
        boundary_edges,
        bounding_box,
        classify_nesw,
        closest_point,
        derivative,
        distance,
        edges,
        edge_face_connectivity,
        edge_points,
        faces_sharing_vertex,
        face_edge_connectivity,
        face_points,
        find_face,
        find_face!,
        find_segment_faces,
        gauss_legendre_quadrature,
        generate_angular_quadrature,
        generate_tracks,
        get_start_edge_nesw,
        hilbert_curve,
        HRPM_height,
        HRPM_width,
        intersect,
        intersect_edges,
        intersect_faces,
        is_left,
        jacobian,
        log_timestamps,
        materialize_edge,
        materialize_edges,
        materialize_face,
        materialize_faces,
        midpoint,
        next_edge_and_face_linear,
        next_edge_and_face_fallback_linear,
        node_height,
        node_level,
        norm,
        partition_rectangularly,
        plot_track_edge_to_edge,
        ray_trace,
        ray_trace_edge_to_edge,
        ray_trace_angle_edge_to_edge!,
        ray_trace_track_edge_to_edge,
        read_abaqus_2d,
        read_vtk_2d,
        real_to_parametric,
        reorder_points_to_hilbert,
        segmentize,
        shared_edge,
        sort_points,
        submesh,
        to_lines,
        triangulate,
        validate_ray_tracing_data,
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

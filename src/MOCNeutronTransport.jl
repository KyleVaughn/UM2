module MOCNeutronTransport

# Compilation Options
# ---------------------------------------------------------------------------------------------
const path_to_gmsh_api = "/usr/local/lib/gmsh.jl"
const enable_visualization = true
const visualize_ray_tracing = false 

# using
# ---------------------------------------------------------------------------------------------
using Logging
using HDF5
using LightXML
using LinearAlgebra
using StaticArrays
using Dates: now, format
using LoggingExtras: TransformerLogger, global_logger

# import
# ---------------------------------------------------------------------------------------------
import Base: +, -, *, /, ==, ≈
import LinearAlgebra: ×, ⋅, norm, inv
# import Base: @propagate_inbounds
# import Base: broadcastable, getindex, getproperty, +, -, *, /, in, intersect, 
#              isapprox, rand, union

# Optional compilation/local dependencies
# ---------------------------------------------------------------------------------------------
if enable_visualization 
    using GLMakie: Axis, Figure, LineSegments, Mesh, Scatter, current_axis, record
    import GLMakie: linesegments!, mesh!, scatter!, convert_arguments
end
# Gmsh
include(path_to_gmsh_api)

# Setup logger to have time stamps 
# ---------------------------------------------------------------------------------------------
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
include("./primitives/Edge.jl")
include("./primitives/Face.jl")
include("./primitives/VectorND.jl")
include("./primitives/Point.jl")
include("./primitives/LineSegment.jl")
include("./primitives/AABB.jl")
include("./primitives/QuadraticSegment.jl")
include("./primitives/Polygon.jl")
#include("./primitives/Triangle.jl")
#include("./primitives/Quadrilateral.jl")
#include("./primitives/Triangle6.jl")
#include("./primitives/Quadrilateral8.jl")
##include("L_system.jl")
#include("./mesh/UnstructuredMesh.jl")
#include("./mesh/UnstructuredMesh2D.jl")
#include("./mesh/LinearUnstructuredMesh2D.jl")
#include("./mesh/QuadraticUnstructuredMesh2D.jl")
#include("./mesh/HierarchicalRectangularlyPartitionedMesh.jl")
#include("./mesh/IO_abaqus.jl")
##include("./mesh/IO_vtk.jl")
#include("./mesh/IO_xdmf.jl")
#include("gauss_legendre_quadrature.jl")
#include("./ray_tracing/AngularQuadrature.jl")
#include("./ray_tracing/ray_trace.jl")
#include("./ray_tracing/ray_trace_low_level.jl")


# Structs/Types
export AABB, AABB2D, AABB3D,
       Decagon,
       Edge, Edge2D, Edge3D, 
       Face, Face2D, Face3D, 
       Heptagon, Hexagon,
       LineSegment, LineSegment2D, LineSegment3D,
       Octagon, Nonagon,
       Pentagon,
       Point, Point2D, Point3D, 
       Polygon,
       QuadraticSegment, QuadraticSegment2D, QuadraticSegment3D, 
       Quadrilateral, Quadrilateral2D, Quadrilateral3D, 
       Quadrilateral8, Quadrilateral82D, Quadrilateral83D, 
       Tree, Triangle, Triangle2D, Triangle3D, 
       Triangle6, Triangle62D, Triangle63D, 
       UnstructuredMesh, UnstructuredMesh2D, UnstructuredMesh3D, 
       Vector2D, Vector3D
#         LinearUnstructuredMesh2D,
#         QuadraticUnstructuredMesh2D,
#         QuadrilateralMesh2D,
#         TriangleMesh2D,
#         UnstructuredMesh2D

# Convenience operators
const 𝗗 = derivative
#const ∇ = gradient
#const ∇² = laplacian
const 𝗝= jacobian

# Operators
export +, -, ⋅, ×, ==, ≈, 𝗗, 𝗝

# Methods
export arclength, area, centroid, depth, boundingbox, derivative, distance, distance², 
       gauss_legendre_quadrature, height, intersect, inv, isleft, isstraight, jacobian, 
       midpoint, nearest_point, norm, norm², rand, real_to_parametric, sortpoints, 
       sortpoints!, triangulate, union, width
# export  +, -, *, /, ×, ⋅, ⪇ , ⪉ , ∇ , ∇²,
#         add_boundary_edges,
#         add_boundary_edges!,
#         add_connectivity,
#         add_connectivity!,
#         add_edges,
#         add_edges!,
#         add_everything,
#         add_everything!,
#         add_edge_face_connectivity,
#         add_face_edge_connectivity,
#         add_materialized_edges,
#         add_materialized_edges!,
#         add_materialized_faces,
#         add_materialized_faces!,
#         arclength,
#         area,
#         boundary_edges,
#         boundingbox,
#         centroid,
#         closest_point,
#         cross,
#         distance,
#         distance²,
#         dot,
#         edges,
#         edge_face_connectivity,
#         edge_points,
#         faces_sharing_vertex,
#         face_edge_connectivity,
#         face_points,
#         find_face,
#         find_segment_faces,
#         generate_angular_quadrature,
#         generate_tracks,
#         height,
#         hilbert_curve,
#         intersect,
#         intersect_edges,
#         intersect_faces,
#         isleft,
#         isstraight,
#         jacobian,
#         J,
#         log_timestamps,
#         materialize_edge,
#         materialize_edges,
#         materialize_face,
#         materialize_faces,
#         midpoint,
#         norm,
#         norm²,
#         partition_rectangularly,
#         plot_track_edge_to_edge,
#         rand,
#         ray_trace,
#         ray_trace_edge_to_edge,
#         ray_trace_angle_edge_to_edge!,
#         ray_trace_track_edge_to_edge,
#         read_abaqus_2d,
#         read_vtk_2d,
#         real_to_parametric,
#         reorder_faces_to_hilbert!,
#         reorder_points_to_hilbert!,
#         reorder_to_hilbert!,
#         segmentize,
#         shared_edge,
#         sortpoints,
#         sortpoints!,
#         sort_intersection_points!,
#         submesh,
#         to_lines,
#         triangulate,
#         union,
#         validate_ray_tracing_data,
#         width,
#         write_vtk_2d,
#         write_xdmf_2d
# 
# # Gmsh
# export gmsh,
#        gmsh_generate_rectangular_grid,
#        gmsh_group_preserving_fragment,
#        gmsh_overlay_rectangular_grid
# 
# Plot
if enable_visualization
    export Figure, Axis
    export scatter, linesegments, mesh,
           scatter!, linesegments!, mesh!
end

end # module

module MOCNeutronTransport

# Compilation Options
# ---------------------------------------------------------------------------------------------
const path_to_gmsh_api = "/usr/local/lib/gmsh.jl"
const enable_visualization = true
const visualize_ray_tracing = false 

# using
# ---------------------------------------------------------------------------------------------
using CUDA
using Logging
using HDF5
using LightXML
using LinearAlgebra
using StaticArrays
using Dates: now, format
using LoggingExtras: TransformerLogger, global_logger

# import
# ---------------------------------------------------------------------------------------------
import Base: +, -, *, /, ==, ≈, intersect, sort, sort!
import LinearAlgebra: ×, ⋅, norm, inv

# Optional compilation/local dependencies
# ---------------------------------------------------------------------------------------------
if enable_visualization 
    using GLMakie: Axis, Axis3, Figure, LineSegments, Mesh, Scatter, current_axis, record
    import GLMakie: linesegments, linesegments!, mesh, mesh!, scatter, scatter!, 
                    convert_arguments
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

include("trees/Tree.jl")
include("trees/AnyTree.jl")
include("constants.jl")
#include("operators.jl")
include("./gmsh/gmsh_generate_rectangular_grid.jl")
include("./gmsh/gmsh_group_preserving_fragment.jl")
include("./gmsh/gmsh_overlay_rectangular_grid.jl")
include("./primitives/Edge.jl")
include("./primitives/Face.jl")
include("./primitives/VectorND.jl")
include("./primitives/Point.jl")
include("./primitives/LineSegment.jl")
include("./primitives/AABox.jl")
include("./primitives/QuadraticSegment.jl")
include("./primitives/Polygon.jl")
include("./primitives/Triangle.jl")
include("./primitives/QuadraticPolygon.jl")
##include("L_system.jl")
include("./mesh/UnstructuredMesh.jl")
include("./mesh/PolygonMesh.jl")
include("./mesh/QuadraticPolygonMesh.jl")
include("./mesh/HierarchicalMeshPartition.jl")
include("./mesh/IO_abaqus.jl")
##include("./mesh/IO_vtk.jl")
include("./mesh/IO_xdmf.jl")
include("gauss_legendre_quadrature.jl")
#include("./ray_tracing/AngularQuadrature.jl")
#include("./raytracing/raytrace.jl")
#include("./ray_tracing/ray_trace_low_level.jl")


# Structs/Types
export AABox, AABox2D, AABox3D, AnyTree,
       Edge, Edge2D, Edge3D,
       Face, Face2D, Face3D,
       Hexagon, HierarchicalMeshPartition,
       LineSegment, LineSegment1D, LineSegment2D, LineSegment3D,
       MeshPartitionTree,
       Point, Point1D, Point2D, Point3D, Polygon, PolygonMesh, 
       QuadraticPolygon, QuadraticPolygonMesh, QuadraticSegment, QuadraticSegment2D,
       QuadraticSegment3D, QuadraticTriangle, QuadraticTriangle2D, QuadraticTriangle3D,
       QuadraticQuadrilateral, QuadraticQuadrilateral2D, QuadraticQuadrilateral2D, 
       QuadraticTriangleMesh, QuadraticQuadrilateralMesh, Quadrilateral, Quadrilateral2D,
       Quadrilateral3D,
       Tree, Triangle, Triangle2D, Triangle3D, TriangleMesh,
       UnstructuredMesh, UnstructuredMesh2D, UnstructuredMesh3D,
       Vector2D, Vector3D

# Convenience operators
const 𝗗 = derivative
#const ∇ = gradient
#const ∇² = laplacian
const 𝗝= jacobian

# Operators
export +, -, ⋅, ×, ==, ≈, 𝗗, 𝗝

# Methods
export arclength, area, 
       boundingbox, 
       centroid, 
       depth, derivative, distance, distance², 
       edgepoints, edge_face_connectivity, 
       facepoints, face_edge_connectivity, findface, findface_implicit, findface_explicit, 
       gauss_legendre_quadrature, 
       height, 
       intersect, intersect_edges, intersect_edges_CUDA, inv, isleft, isstraight, isroot, 
       jacobian, 
       leaves, linear_edges,log_timestamps, 
       materialize_edge, materialize_edges, materialize_face, materialize_faces,
       materialize_polygon, materialize_quadratic_polygon, midpoint, 
       nan_point, nearest_point, norm, norm², num_edges, 
       partition_mesh,
       quadratic_edges,
       rand, read_abaqus2d, real_to_parametric, 
       sort, sort!, sort_intersection_points!, submesh,
       triangulate, triangulate_nonconvex,
       union, 
       width, write_xdmf2d

# Gmsh
export gmsh,
       gmsh_generate_rectangular_grid,
       gmsh_group_preserving_fragment,
       gmsh_overlay_rectangular_grid
 
# Plot
if enable_visualization
    export Figure, Axis, Axis3
    export scatter, linesegments, mesh,
           scatter!, linesegments!, mesh!
end

end # module

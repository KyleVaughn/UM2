struct GeneralUnstructuredMesh_2D <: UnstructuredMesh_2D
    name::String
    points::Vector{Point_2D}
    edges::Vector{<:SVector{L, UInt32} where {L}}
    materialized_edges::Vector{<:Edge_2D}
    faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}}
    materialized_faces::Vector{<:Face_2D}
    edge_face_connectivity::Vector{SVector{2, UInt32}}
    face_edge_connectivity::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}}
    boundary_edges::Vector{Vector{UInt32}}
    face_sets::Dict{String, Set{UInt32}}
end

function GeneralUnstructuredMesh_2D(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D} = Point_2D[],
        edges::Vector{<:SVector{L, UInt32} where {L}} = SVector{2, UInt32}[],
        materialized_edges::Vector{<:Edge_2D} = LineSegment_2D[],
        faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}} = SVector{4, UInt32}[],
        materialized_faces::Vector{<:Face_2D} = Triangle_2D[],
        edge_face_connectivity::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        face_edge_connectivity ::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}} = SVector{3, UInt32}[],
        boundary_edges::Vector{Vector{UInt32}} = Vector{UInt32}[],
        face_sets::Dict{String, Set{UInt32}} = Dict{String, Set{UInt32}}()
    ) where {F <: AbstractFloat, U <: Unsigned}
        return GeneralUnstructuredMesh_2D(name,
                                          points,
                                          edges,
                                          materialized_edges,
                                          faces,
                                          materialized_faces,
                                          edge_face_connectivity,
                                          face_edge_connectivity,
                                          boundary_edges,
                                          face_sets,
                                         )
end

Base.broadcastable(mesh::GeneralUnstructuredMesh_2D) = Ref(mesh)

# Return a mesh with boundary edges
function add_boundary_edges(mesh::M; bounding_shape="None"
    ) where {M <: UnstructuredMesh_2D}
    if 0 === length(mesh.edge_face_connectivity)
        mesh = add_connectivity(mesh)
    end
    return M(name = mesh.name,
             points = mesh.points,
             edges = mesh.edges,
             materialized_edges = mesh.materialized_edges,
             faces = mesh.faces,
             materialized_faces = mesh.materialized_faces,
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = mesh.face_edge_connectivity,
             boundary_edges = boundary_edges(mesh, bounding_shape),
             face_sets = mesh.face_sets
            )
end

# Return a mesh with face/edge connectivity and edge/face connectivity
function add_connectivity(mesh::M) where {M <: UnstructuredMesh_2D}
    return add_edge_face_connectivity(mesh)
end

# Return a mesh with its edges
function add_edges(mesh::M) where {M <: UnstructuredMesh_2D}
    return M(name = mesh.name,
             points = mesh.points,
             edges = edges(mesh),
             materialized_edges = mesh.materialized_edges,
             faces = mesh.faces,
             materialized_faces = mesh.materialized_faces,
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = mesh.face_edge_connectivity,
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Return a mesh with edge/face connectivity
function add_edge_face_connectivity(mesh::M) where {M <: UnstructuredMesh_2D}
    if 0 === length(mesh.face_edge_connectivity)
        mesh = add_face_edge_connectivity(mesh)
    end
    return M(name = mesh.name,
             points = mesh.points,
             edges = mesh.edges,
             materialized_edges = mesh.materialized_edges,
             faces = mesh.faces,
             materialized_faces = mesh.materialized_faces,
             edge_face_connectivity = edge_face_connectivity(mesh),
             face_edge_connectivity = mesh.face_edge_connectivity,
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Return a mesh with every field created
function add_everything(mesh::M) where {M <: UnstructuredMesh_2D}
    return add_materialized_faces(
             add_materialized_edges(
               add_boundary_edges(mesh, bounding_shape = "Rectangle")))
end


# Return a mesh with face/edge connectivity
function add_face_edge_connectivity(mesh::M) where {M <: UnstructuredMesh_2D}
    if 0 === length(mesh.edges)
        mesh = add_edges(mesh)
    end
    return M(name = mesh.name,
             points = mesh.points,
             edges = mesh.edges,
             materialized_edges = mesh.materialized_edges,
             faces = mesh.faces,
             materialized_faces = mesh.materialized_faces,
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = face_edge_connectivity(mesh),
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Return a mesh with materialized edges
function add_materialized_edges(mesh::M) where {M <: UnstructuredMesh_2D}
    if 0 === length(mesh.edges)
        mesh = add_edges(mesh)
    end
    return M(name = mesh.name,
             points = mesh.points,
             edges = mesh.edges,
             materialized_edges = materialize_edges(mesh),
             faces = mesh.faces,
             materialized_faces = mesh.materialized_faces,
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = mesh.face_edge_connectivity,
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Return a mesh with materialized faces
function add_materialized_faces(mesh::M) where {M <: UnstructuredMesh_2D}
    return M(name = mesh.name,
             points = mesh.points,
             edges = mesh.edges,
             materialized_edges = mesh.materialized_edges,
             faces = mesh.faces,
             materialized_faces = materialize_faces(mesh),
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = mesh.face_edge_connectivity,
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Area of face
function area(face::SVector{N, UInt32}, points::Vector{Point_2D}) where {N}
    return area(materialize_face(face, points))
end

# Return the area of a face set
function area(mesh::M, face_set::Set{UInt32}) where {M <: UnstructuredMesh_2D} 
    if 0 < length(mesh.materialized_faces)
        return mapreduce(x->area(mesh.materialized_faces[x]), +, face_set)
    else
        return mapreduce(x->area(mesh.faces[x], mesh.points), +, face_set)
    end 
end

# Return the area of a face set by name
function area(mesh::M, set_name::String) where {M <: UnstructuredMesh_2D} 
    return area(mesh, mesh.face_sets[set_name])
end

# Bounding box of a vector of points
function boundingbox(points::Vector{Point_2D})
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return Rectangle_2D(minimum(x), minimum(y), maximum(x), maximum(y))
end

# Bounding box of a vector of points
function boundingbox(points::SVector{L, Point_2D}) where {L}
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return Rectangle_2D(minimum(x), minimum(y), maximum(x), maximum(y))
end

# Return a vector containing vectors of the edges in each side of the mesh's bounding shape, e.g.
# For a rectangular bounding shape the sides are North, East, South, West. Then the output would
# be [ [e1, e2, e3, ...], [e17, e18, e18, ...], ..., [e100, e101, ...]]
function boundary_edges(mesh::M, bounding_shape::String) where {M <: UnstructuredMesh_2D}
    # edges which have face 0 in their edge_face connectivity are boundary edges
    the_boundary_edges = UInt32.(findall(x->x[1] === 0x00000000, mesh.edge_face_connectivity))
    if bounding_shape == "Rectangle"
        # Sort edges into NESW
        bb = boundingbox(mesh.points)
        y_north = bb.ymax
        x_east  = bb.xmax
        y_south = bb.ymin
        x_west  = bb.xmin
        p_NW = Point_2D(x_west, y_north)
        p_NE = bb.tr
        p_SE = Point_2D(x_east, y_south)
        p_SW = bb.bl
        edges_north = UInt32[] 
        edges_east = UInt32[] 
        edges_south = UInt32[] 
        edges_west = UInt32[] 
        # Insert edges so that indices move from NW -> NE -> SE -> SW -> NW
        for edge ∈  the_boundary_edges
            epoints = edge_points(mesh.edges[edge], mesh.points)
            if all(x->abs(x[2] - y_north) < 1e-4, epoints)
                insert_boundary_edge!(edge, p_NW, edges_north, mesh)
            elseif all(x->abs(x[1] - x_east) < 1e-4, epoints)
                insert_boundary_edge!(edge, p_NE, edges_east, mesh)
            elseif all(x->abs(x[2] - y_south) < 1e-4, epoints)
                insert_boundary_edge!(edge, p_SE, edges_south, mesh)
            elseif all(x->abs(x[1] - x_west) < 1e-4, epoints)
                insert_boundary_edge!(edge, p_SW, edges_west, mesh)
            else
                @error "Edge $edge could not be classified as NSEW"
            end
        end
        return [ edges_north, edges_east, edges_south, edges_west ]
    else
        return [ convert(Vector{UInt32}, the_boundary_edges) ]
    end 
end 

# SVector of MVectors of point IDs representing the 3 edges of a triangle
function edges(face::SVector{3, UInt32})
    edges = SVector( MVector{2, UInt32}(face[1], face[2]),
                     MVector{2, UInt32}(face[2], face[3]),
                     MVector{2, UInt32}(face[3], face[1]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 4 edges of a quadrilateral
function edges(face::SVector{4, UInt32})
    edges = SVector( MVector{2, UInt32}(face[1], face[2]),
                     MVector{2, UInt32}(face[2], face[3]),
                     MVector{2, UInt32}(face[3], face[4]),
                     MVector{2, UInt32}(face[4], face[1]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 3 edges of a quadratic triangle
function edges(face::SVector{6, UInt32})
    edges = SVector( MVector{3, UInt32}(face[1], face[2], face[4]),
                     MVector{3, UInt32}(face[2], face[3], face[5]),
                     MVector{3, UInt32}(face[3], face[1], face[6]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 4 edges of a quadratic quadrilateral
function edges(face::SVector{8, UInt32})
    edges = SVector( MVector{3, UInt32}(face[1], face[2], face[5]),
                     MVector{3, UInt32}(face[2], face[3], face[6]),
                     MVector{3, UInt32}(face[3], face[4], face[7]),
                     MVector{3, UInt32}(face[4], face[1], face[8]) )
    # Order th linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# The unique edges from a vector of triangles or quadrilaterals represented by point IDs
function edges(mesh::M) where {M <: UnstructuredMesh_2D}
    edges_filtered = sort(unique(reduce(vcat, edges.(mesh.faces))))
    return [ SVector(e.data) for e in edges_filtered ]
end

# A vector of length 2 SVectors, denoting the face ID each edge is connected to. If the edge
# is a boundary edge, face ID 0 is returned
function edge_face_connectivity(mesh::M) where {M <: UnstructuredMesh_2D}
    # Each edge should only border 2 faces if it is an interior edge, and 1 face if it is
    # a boundary edge.
    # Loop through each face in the face_edge_connectivity vector and mark each edge with
    # the faces that it borders.
    if length(mesh.edges) === 0
        @error "Does not have edges!"
    end
    if length(mesh.face_edge_connectivity) === 0
        @error "Does not have face/edge connectivity!"
    end
    edge_face = [MVector{2, UInt32}(0, 0) for _ in eachindex(mesh.edges)]
    for (iface, edges) in enumerate(mesh.face_edge_connectivity)
        for iedge in edges
            # Add the face id in the first non-zero position of the edge_face conn. vec.
            if edge_face[iedge][1] == 0
                edge_face[iedge][1] = iface
            elseif edge_face[iedge][2] == 0
                edge_face[iedge][2] = iface
            else
                @error "Edge $iedge seems to have 3 faces associated with it!"
            end
        end
    end
    return [SVector(sort(two_faces).data) for two_faces in edge_face]
end

# Return an SVector of the points in the edge (Linear)
function edge_points(edge::SVector{2, UInt32}, points::Vector{Point_2D})
    return SVector(points[edge[1]], points[edge[2]])
end

# Return an SVector of the points in the edge (Quadratic)
function edge_points(edge::SVector{3, UInt32}, points::Vector{Point_2D})
    return SVector(points[edge[1]], points[edge[2]], points[edge[3]])
end

# Return an SVector of the points in the edge
function edge_points(edge_id::UInt32, mesh::M) where {M <: UnstructuredMesh_2D}
    return edge_points(mesh.edges[edge_id], mesh.points)
end

# Return an SVector of the points in the face (Triangle)
function face_points(face::SVector{3, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[1]], points[face[2]], points[face[3]])
end

# Return an SVector of the points in the face (Quadrilateral)
function face_points(face::SVector{4, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[1]], points[face[2]], points[face[3]], points[face[4]])
end

# Return an SVector of the points in the face (Triangle6)
function face_points(face::SVector{6, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[1]], points[face[2]], points[face[3]],
                   points[face[4]], points[face[5]], points[face[6]])
end

# Return an SVector of the points in the face (Quadrilateral8)
function face_points(face::SVector{8, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[1]], points[face[2]], points[face[3]], points[face[4]],
                   points[face[5]], points[face[6]], points[face[7]], points[face[8]])
end

# Return an SVector of the points in the face
function face_points(edge_id::UInt32, mesh::M) where {M <: UnstructuredMesh_2D}
    return face_points(mesh.faces[edge_id], mesh.points)
end

# Insert the boundary edge into the correct place in the vector of edge indices, based on
# the distance from some reference point
function insert_boundary_edge!(edge_index::UInt32, p_ref::Point_2D, edge_indices::Vector{UInt32},
                               mesh::M) where {M <: UnstructuredMesh_2D}
    # Compute the minimum distance from the edge to be inserted to the reference point
    insertion_distance = minimum(distance.(Ref(p_ref), 
                                           edge_points(mesh.edges[edge_index], mesh.points)))
    # Loop through the edge indices until an edge with greater distance from the reference point
    # is found, then insert
    nindices = length(edge_indices)
    for i ∈ 1:nindices
        epoints = edge_points(mesh.edges[edge_indices[i]], mesh.points)
        edge_distance = minimum(distance.(Ref(p_ref), epoints))
        if insertion_distance < edge_distance
            insert!(edge_indices, i, edge_index)
            return nothing
        end
    end
    insert!(edge_indices, nindices+1, edge_index)
    return nothing
end

# Intersect a line with the mesh. Returns a vector of intersection points, sorted based
# upon distance from the line's start point
function intersect(l::LineSegment_2D, mesh::UnstructuredMesh_2D)
    # Edges are faster, so they are the default
    if length(mesh.edges) !== 0
        if 0 < length(mesh.materialized_edges)
            return intersect_edges_explicit(l, mesh.materialized_edges)
        else
            return intersect_edges_implicit(l, mesh.edges, mesh.points)
        end
    else
        if 0 < length(mesh.materialized_faces)
            return intersect_faces_explicit(l, mesh.materialized_faces)
        else
            return intersect_faces_implicit(l, mesh.faces, mesh.points)
        end
    end
end

# Intersect a line with an implicitly defined edge
function intersect_edge_implicit(l::LineSegment_2D,
                                 edge::SVector{L, UInt32} where {L},
                                 points::Vector{Point_2D})
    return l ∩ materialize_edge(edge, points)
end

# Intersect a line with materialized edges
function intersect_edges_explicit(l::LineSegment_2D, edges::Vector{LineSegment_2D})
    # A vector to hold all of the intersection points
    intersection_points = Point_2D[]
    for edge in edges
        npoints, point = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    sortpoints!(l[1], intersection_points)
    return intersection_points
end

# Intersect a line with implicitly defined edges
function intersect_edges_explicit(l::LineSegment_2D, edges::Vector{QuadraticSegment_2D})
    # A vector to hold all of the intersection points
    intersection_points = Point_2D[]
    for edge in edges
        npoints, points = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    sortpoints!(l[1], intersection_points)
    return intersection_points
end


# Intersect a line with a vector of implicitly defined linear edges
function intersect_edges_implicit(l::LineSegment_2D,
                                  edges::Vector{SVector{2, UInt32}},
                                  points::Vector{Point_2D})
    # A vector to hold all of the intersection points
    intersection_points = Point_2D[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, point = intersect_edge_implicit(l, edge, points)
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    sortpoints!(l[1], intersection_points)
    return intersection_points
end

# Intersect a line with a vector of implicitly defined quadratic edges
function intersect_edges_implicit(l::LineSegment_2D,
                                  edges::Vector{SVector{3, UInt32}},
                                  points::Vector{Point_2D})
    # An array to hold all of the intersection points
    intersection_points = Point_2D[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, ipoints = intersect_edge_implicit(l, edge, points)
        if 0 < npoints
            append!(intersection_points, ipoints[1:npoints])
        end
    end
    sortpoints!(l[1], intersection_points)
    return intersection_points
end

# Intersect a line with an implicitly defined face
function intersect_face_implicit(l::LineSegment_2D,
                                 face::SVector{L, UInt32} where {L},
                                 points::Vector{Point_2D})
    return l ∩ materialize_face(face, points)
end

# Intersect a line with explicitly defined linear faces
function intersect_faces_explicit(l::LineSegment_2D, faces::Vector{<:Face_2D} )
    # An array to hold all of the intersection points
    intersection_points = Point_2D[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    sortpoints!(l[1], intersection_points)
    return intersection_points
end

# Intersect a line with implicitly defined faces
function intersect_faces_implicit(l::LineSegment_2D,
                                  faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}},
                                  points::Vector{Point_2D})
    # An array to hold all of the intersection points
    intersection_points = Point_2D[]
    # Intersect the line with each of the faces
    for face in faces
        npoints, ipoints = intersect_face_implicit(l, face, points)
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, ipoints[1:npoints])
        end
    end
    sortpoints!(l[1], intersection_points)
    return intersection_points
end


# Return a LineSegment_2D from the point IDs in an edge
function materialize_edge(edge::SVector{2, UInt32}, points::Vector{Point_2D})
    return LineSegment_2D(edge_points(edge, points))
end

# Return a QuadraticSegment_2D from the point IDs in an edge
function materialize_edge(edge::SVector{3, UInt32}, points::Vector{Point_2D})
    return QuadraticSegment_2D(edge_points(edge, points))
end

# Return a LineSegment_2D or QuadraticSegment_2D
function materialize_edge(edge_id::UInt32, mesh::M) where {M <: UnstructuredMesh_2D}
    return materialize_edge(mesh.edges[edge_id], mesh.points)
end

# Return a materialized edge for each edge in the mesh
function materialize_edges(mesh::M) where {M <: UnstructuredMesh_2D}
    return materialize_edge.(mesh.edges, Ref(mesh.points))
end

# Return a Triangle_2D from the point IDs in a face
function materialize_face(face::SVector{3, UInt32}, points::Vector{Point_2D})
    return Triangle_2D(face_points(face, points))
end

# Return a Quadrilateral_2D from the point IDs in a face
function materialize_face(face::SVector{4, UInt32}, points::Vector{Point_2D})
    return Quadrilateral_2D(face_points(face, points))
end

# Return a Triangle6_2D from the point IDs in a face
function materialize_face(face::SVector{6, UInt32}, points::Vector{Point_2D})
    return Triangle6_2D(face_points(face, points))
end

# Return a Quadrilateral8_2D from the point IDs in a face
function materialize_face(face::SVector{8, UInt32}, points::Vector{Point_2D})
    return Quadrilateral8_2D(face_points(face, points))
end

# Return an SVector of the points in the edge
function materialize_face(face_id::UInt32, mesh::M) where {M <: UnstructuredMesh_2D}
    return materialize_face(mesh.faces[face_id], mesh.points)
end

# Return a materialized face for each face in the mesh
function materialize_faces(mesh::M) where {M <: UnstructuredMesh_2D}
    return materialize_face.(mesh.faces, Ref(mesh.points))
end

# Return the number of edges in a face
function num_edges(face::SVector{L, UInt32}) where {L}
    if L % 3 === 0 
        return 0x00000003
    elseif L % 4 === 0
        return 0x00000004
    else
        # Error
        return 0x00000000
    end
end

# How to display a mesh in REPL
function Base.show(io::IO, mesh::M) where {M <: UnstructuredMesh_2D}
    mesh_type = typeof(mesh)
    println(io, mesh_type)
    println(io, "  ├─ Name      : $(mesh.name)")
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        size_KB = size_MB*1000
        println(io, "  ├─ Size (KB) : $size_KB")
    else
        println(io, "  ├─ Size (MB) : $size_MB")
    end
    println(io, "  ├─ Points    : $(length(mesh.points))")
    nedges = length(mesh.edges)
    println(io, "  ├─ Edges     : $nedges")
    println(io, "  │  ├─ Linear         : $(count(x->x isa SVector{2, UInt32},  mesh.edges))")
    println(io, "  │  ├─ Quadratic      : $(count(x->x isa SVector{3, UInt32},  mesh.edges))")
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_edges) !== 0)")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{3, UInt32},  mesh.faces))")
    println(io, "  │  ├─ Quadrilateral  : $(count(x->x isa SVector{4, UInt32},  mesh.faces))")
    println(io, "  │  ├─ Triangle6      : $(count(x->x isa SVector{6, UInt32},  mesh.faces))")
    println(io, "  │  ├─ Quadrilateral8 : $(count(x->x isa SVector{8, UInt32},  mesh.faces))")
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_faces) !== 0)")
    println(io, "  ├─ Connectivity")
    println(io, "  │  ├─ Edge/Face : $(0 < length(mesh.edge_face_connectivity))")
    println(io, "  │  └─ Face/Edge : $(0 < length(mesh.face_edge_connectivity))")
    println(io, "  ├─ Boundary edges")
    nsides = length(mesh.boundary_edges)
    if nsides !== 0
        println(io, "  │  ├─ Edges : $(mapreduce(x->length(x), +, mesh.boundary_edges))")
    else
        println(io, "  │  ├─ Edges : 0") 
    end
    println(io, "  │  └─ Sides : $nsides")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end

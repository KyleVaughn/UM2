struct UnstructuredMesh_2D{T <: AbstractFloat, I <: Unsigned}
    name::String 
    points::Vector{Point_2D{T}}
    edges::Vector{<:Union{NTuple{2, I}, NTuple{3, I}}} 
    edges_materialized::Vector{<:Edge_2D{T}} 
    faces::Vector{<:Tuple{Vararg{I, N} where N}} 
    faces_materialized::Vector{<:Face_2D{T}} 
    edge_face_connectivity::Vector{NTuple{2, I}} 
    face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}} 
    boundary_edges::Vector{Vector{I}}
    face_sets::Dict{String, Set{I}} 
end

function UnstructuredMesh_2D{T, I}(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D{T}} = Point_2D{T}[],
        edges::Vector{<:Union{NTuple{2, I}, NTuple{3, I}}} = NTuple{2, I}[],
        edges_materialized::Vector{<:Edge_2D{T}} = LineSegment_2D{T}[],
        faces::Vector{<:Tuple{Vararg{I, N} where N}} = NTuple{4, I}[],
        faces_materialized::Vector{<:Face_2D{T}} = Triangle_2D{T}[],
        edge_face_connectivity::Vector{NTuple{2, I}} = NTuple{2, I}[], 
        face_edge_connectivity ::Vector{<:Tuple{Vararg{I, M} where M}} = NTuple{3, I}[],
        boundary_edges::Vector{Vector{I}} = Vector{I}[],
        face_sets::Dict{String, Set{I}} = Dict{String, Set{I}}()
    ) where {T <: AbstractFloat, I <: Unsigned}
        return UnstructuredMesh_2D{T, I}(name, 
                                         points, 
                                         edges, 
                                         edges_materialized, 
                                         faces, 
                                         faces_materialized, 
                                         edge_face_connectivity,
                                         face_edge_connectivity,
                                         boundary_edges,
                                         face_sets, 
                                        )
end

Base.broadcastable(mesh::UnstructuredMesh_2D) = Ref(mesh)

# Cell types are the same as VTK
const UnstructuredMesh_2D_linear_cell_types = UInt32[5, # Triangle 
                                                     9  # Quadrilateral
                                                    ]
const UnstructuredMesh_2D_quadratic_cell_types = UInt32[22, # Triangle6
                                                        23  # Quadrilateral8
                                                       ]
const UnstructuredMesh_2D_cell_types = vcat(UnstructuredMesh_2D_linear_cell_types,
                                            UnstructuredMesh_2D_quadratic_cell_types)

function add_boundary_edges(mesh::UnstructuredMesh_2D{T, I};
                          bounding_shape="Rectangle") where {T<:AbstractFloat, I <: Unsigned}
    if 0 < length(mesh.edge_face_connectivity)  
        return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                         points = mesh.points,
                                         edges = mesh.edges,
                                         edges_materialized = mesh.edges_materialized,
                                         faces = mesh.faces,
                                         faces_materialized = mesh.faces_materialized,
                                         edge_face_connectivity = mesh.edge_face_connectivity, 
                                         face_edge_connectivity = mesh.face_edge_connectivity, 
                                         boundary_edges = boundary_edges(mesh),
                                         face_sets = mesh.face_sets
                                        )
    else
        mesh_con = add_connectivity(mesh)
        return UnstructuredMesh_2D{T, I}(name = mesh_con.name,
                                         points = mesh_con.points,
                                         edges = mesh_con.edges,
                                         edges_materialized = mesh_con.edges_materialized,
                                         faces = mesh_con.faces,
                                         faces_materialized = mesh_con.faces_materialized,
                                         edge_face_connectivity = mesh_con.edge_face_connectivity, 
                                         face_edge_connectivity = mesh_con.face_edge_connectivity, 
                                         boundary_edges = boundary_edges(mesh_con),
                                         face_sets = mesh_con.face_sets
                                        )
    end
end

function add_connectivity(mesh::UnstructuredMesh_2D{T}) where {T<:AbstractFloat}
    if 0 < length(mesh.edges)  
        return add_edge_face_connectivity(add_face_edge_connectivity(mesh))
    else
        return add_edge_face_connectivity(add_face_edge_connectivity(add_edges(mesh)))
    end
end

function add_edges(mesh::UnstructuredMesh_2D{T, I}) where {T<:AbstractFloat, I <: Unsigned}
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                     points = mesh.points,
                                     edges = edges(mesh),
                                     edges_materialized = mesh.edges_materialized,
                                     faces = mesh.faces,
                                     faces_materialized = mesh.faces_materialized,
                                     edge_face_connectivity = mesh.edge_face_connectivity, 
                                     face_edge_connectivity = mesh.face_edge_connectivity, 
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
end

function add_edges_materialized(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    if 0 < length(mesh.edges)
        mat_edges = edges_materialized(mesh)
    else
        mat_edges = edges_materialized(add_edges(mesh))
    end
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                     points = mesh.points,
                                     edges = mesh.edges,
                                     edges_materialized = mat_edges,
                                     faces = mesh.faces,
                                     faces_materialized = mesh.faces_materialized,
                                     edge_face_connectivity = mesh.edge_face_connectivity, 
                                     face_edge_connectivity = mesh.face_edge_connectivity, 
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
end

function add_edge_face_connectivity(mesh::UnstructuredMesh_2D{T, I}) where {T<:AbstractFloat, I<:Unsigned}
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                    points = mesh.points,
                                    edges = edges(mesh),
                                    edges_materialized = mesh.edges_materialized,
                                    faces = mesh.faces,
                                    faces_materialized = mesh.faces_materialized,
                                    edge_face_connectivity = edge_face_connectivity(mesh),
                                    face_edge_connectivity = mesh.face_edge_connectivity, 
                                    boundary_edges = mesh.boundary_edges,
                                    face_sets = mesh.face_sets
                                   )
end

function add_everything(mesh::UnstructuredMesh_2D{T, I}) where {T<:AbstractFloat, I<:Unsigned}
    return add_boundary_edges(
           add_connectivity(
           add_faces_materialized(
           add_edges_materialized(
           add_edges(mesh)))))
end

function add_face_edge_connectivity(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, 
                                                                            I <: Unsigned}
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                    points = mesh.points,
                                    edges = edges(mesh),
                                    edges_materialized = mesh.edges_materialized,
                                    faces = mesh.faces,
                                    faces_materialized = mesh.faces_materialized,
                                    edge_face_connectivity = mesh.edge_face_connectivity,
                                    face_edge_connectivity = face_edge_connectivity(mesh), 
                                    boundary_edges = mesh.boundary_edges,
                                    face_sets = mesh.face_sets
                                   )
end

function add_faces_materialized(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                        I <: Unsigned}
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                     points = mesh.points,
                                     edges = mesh.edges,
                                     edges_materialized = mesh.edges_materialized,
                                     faces = mesh.faces,
                                     faces_materialized = faces_materialized(mesh),
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
end

function area(mesh::UnstructuredMesh_2D{T, I}, face_set::Set{I}) where {T <: AbstractFloat, I <: Unsigned} 
    unsupported::Int64 = sum(x->x[1] ∉  UnstructuredMesh_2D_cell_types, mesh.faces)
    if 0 < unsupported
        @error "Mesh contains an unsupported face type"
    end
    return mapreduce(x->area(mesh, mesh.faces[x]), +, face_set)
end

function area(mesh::UnstructuredMesh_2D{T, I}, set_name::String) where {T <: AbstractFloat, I <: Unsigned}
    return area(mesh, mesh.face_sets[set_name])
end

function area(mesh::UnstructuredMesh_2D{T, I}, face::NTuple{4, I}) where {T <: AbstractFloat, I <: Unsigned}
    return area(Triangle_2D(get_face_points(mesh, face)))
end

function area(mesh::UnstructuredMesh_2D{T, I}, face::NTuple{5, I}) where {T <: AbstractFloat, I <: Unsigned}
    return area(Quadrilateral_2D(get_face_points(mesh, face)))
end

function area(mesh::UnstructuredMesh_2D{T, I}, face::NTuple{7, I}) where {T <: AbstractFloat, I <: Unsigned}
    return area(Triangle6_2D(get_face_points(mesh, face)))
end

function area(mesh::UnstructuredMesh_2D{T, I}, face::NTuple{9, I}) where {T <: AbstractFloat, I <: Unsigned}
    return area(Quadrilateral8_2D(get_face_points(mesh, face)))
end

# Axis-aligned bounding box, in 2d a rectangle.
function AABB(mesh::UnstructuredMesh_2D{T, I}; 
              rectangular_boundary=false) where {T <: AbstractFloat, I <: Unsigned}
    # If the mesh does not have any quadratic faces, the AABB may be determined entirely from the 
    # points. If the mesh does have quadratic cells/faces, we need to find the bounding box of the edges
    # that border the mesh.
    if (any(x->x ∈  UnstructuredMesh_2D_quadratic_cell_types, getindex.(mesh.faces, 1)) && 
        !rectangular_boundary)
        @error "Cannot find AABB for a mesh with quadratic faces that does not have a rectangular boundary"
    else # Can use points
        x = map(p->p[1], mesh.points)
        y = map(p->p[2], mesh.points)
        xmin = minimum(x)
        xmax = maximum(x)
        ymin = minimum(y)
        ymax = maximum(y)
        return Quadrilateral_2D(Point_2D(xmin, ymin), 
                                Point_2D(xmax, ymin),
                                Point_2D(xmax, ymax),
                                Point_2D(xmin, ymax))
    end
end

function boundary_edges(mesh::UnstructuredMesh_2D{T, I}; 
                       bounding_shape="Rectangle") where {T<:AbstractFloat, I <: Unsigned}
    # edges which have face 0 in their edge_face connectivity are boundary edges
    boundary_edges = findall(x->x[1] == 0, mesh.edge_face_connectivity)
    if bounding_shape == "Rectangle"
        # Sort edges into NESW
        bb = AABB(mesh, rectangular_boundary=true)
        y_north = bb.points[3].x[2]
        x_east  = bb.points[3].x[1]
        y_south = bb.points[1].x[2]
        x_west  = bb.points[1].x[1]
        p_NW = bb.points[4]
        p_NE = bb.points[3]
        p_SE = bb.points[2]
        p_SW = bb.points[1]
        edges_north = I[]
        edges_east = I[]
        edges_south = I[]
        edges_west = I[]
        # Insert edges so that indices move from NW -> NE -> SE -> SW -> NW
        for i = 1:length(boundary_edges)
            iedge = I(boundary_edges[i])
            edge_points = get_edge_points(mesh, mesh.edges[iedge])
            if all(x->abs(x[2] - y_north) < 1e-4, edge_points) 
                insert_boundary_edge!(iedge, edges_north, p_NW, mesh)
            elseif all(x->abs(x[1] - x_east) < 1e-4, edge_points) 
                insert_boundary_edge!(iedge, edges_east, p_NE, mesh)
            elseif all(x->abs(x[2] - y_south) < 1e-4, edge_points) 
                insert_boundary_edge!(iedge, edges_south, p_SE, mesh)
            elseif all(x->abs(x[1] - x_west) < 1e-4, edge_points) 
                insert_boundary_edge!(iedge, edges_west, p_SW, mesh)
            else
                @error "Edge $iedge could not be classified as NSEW"
            end
        end
        return [ edges_north, edges_east, edges_south, edges_west ]
    else
        return [ convert(Vector{I}, boundary_edges) ]
    end
end

# Return each edge for a face
# Note, this returns a vector of vectors because we want to mutate the elements of the edge vectors
function edges(face::NTuple{4, I}) where {I <: Unsigned} 
    cell_type = face[1]
    if cell_type ∈  UnstructuredMesh_2D_linear_cell_types 
        edges = [ [face[2], face[3]],
                  [face[3], face[4]],
                  [face[4], face[2]] ]
    else
        @error "Unsupported cell type"
        edges = [[I(0), I(0)]]
    end
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

function edges(face::NTuple{5, I}) where {I <: Unsigned}
    cell_type = face[1]
    if cell_type ∈  UnstructuredMesh_2D_linear_cell_types 
        edges = [ [face[2], face[3]],
                  [face[3], face[4]],
                  [face[4], face[5]],
                  [face[5], face[2]] ]
    else
        @error "Unsupported cell type"
        edges = [[I(0), I(0)]]
    end
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

function edges(face::NTuple{7, I}) where {I <: Unsigned} 
    cell_type = face[1]
    if cell_type ∈  UnstructuredMesh_2D_quadratic_cell_types
        edges = [ [face[2], face[3], face[5]],
                  [face[3], face[4], face[6]],
                  [face[4], face[2], face[7]] ]
    else
        @error "Unsupported cell type."
        edges = [[I(0), I(0)]]
    end
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

function edges(face::NTuple{9, I}) where {I <: Unsigned}  
    cell_type = face[1]
    if cell_type ∈  UnstructuredMesh_2D_quadratic_cell_types
        edges = [ [face[2], face[3], face[6]],
                  [face[3], face[4], face[7]],
                  [face[4], face[5], face[8]],
                  [face[5], face[2], face[9]] ]
    else
        @error "Unsupported cell type."
        edges = [[I(0), I(0)]]
    end
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

function edges(faces::Vector{<:Union{NTuple{4, I}, NTuple{5, I}}}) where {I <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(collect(Set{Vector{I}}(edges_unfiltered)))
    return [ Tuple(e) for e in edges_filtered ]::Vector{NTuple{2, I}}
end

function edges(faces::Vector{<:Union{NTuple{7, I}, NTuple{9, I}}}) where {I <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(collect(Set{Vector{I}}(edges_unfiltered)))
    return [ Tuple(e) for e in edges_filtered ]::Vector{NTuple{3, I}}
end

function edges(faces::Vector{<:Tuple{Vararg{I, N} where N}}) where {I <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(collect(Set{Vector{I}}(edges_unfiltered)))
    return [ Tuple(e) for e in edges_filtered ]::Vector{<:Union{NTuple{2, I}, NTuple{3, I}}}
end

# Create the edges for each face
function edges(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    return edges(mesh.faces) 
end

function edge_face_connectivity(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                        I <: Unsigned}
    # Each edge should only border 2 faces if it is an interior edge, and 1 face if it is
    # a boundary edge.
    # Loop through each face in the face_edge_connectivity vector and mark each edge with 
    # the faces that it borders.
    if length(mesh.edges) === 0
        @error "Mesh does not have edges!"
        edge_face = [MVector{2, I}(zeros(I, 2)) for i = 1:2]
    elseif length(mesh.face_edge_connectivity) === 0
        edge_face = [MVector{2, I}(zeros(I, 2)) for i in eachindex(mesh.edges)]
        face_edge_conn = face_edge_connectivity(mesh)
        for (iface, edges) in enumerate(face_edge_conn)
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
    else # has face_edge connectivity
        edge_face = [MVector{2, I}(zeros(I, 2)) for i in eachindex(mesh.edges)]
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
    end
    return [Tuple(sort(two_faces)) for two_faces in edge_face]
end

function edges_materialized(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    return edge_materialized.(mesh, mesh.edges)
end

function edge_materialized(mesh::UnstructuredMesh_2D{T, I},
                           edge::NTuple{2, I}) where {T <: AbstractFloat, I <: Unsigned}
    return LineSegment_2D(get_edge_points(mesh, edge))
end

function edge_materialized(mesh::UnstructuredMesh_2D{T, I},
                           edge::NTuple{3, I}) where {T <: AbstractFloat, I <: Unsigned}
    return QuadraticSegment_2D(get_edge_points(mesh, edge))
end

function face_edge_connectivity(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    # A vector of MVectors of zeros for each face
    # Each MVector is the length of the number of edges
    face_edge = [MVector{Int64(num_edges(face)), I}(zeros(I, num_edges(face))) for face in mesh.faces]
    if length(mesh.edges) === 0
        @error "Mesh does not have edges!"
    else
        # for each face in the mesh, generate the edges.
        # Search for the index of the edge in the mesh.edges vector
        # Insert the index of the edge into the face_edge connectivity vector 
        for i in eachindex(mesh.faces)
            for (j, edge) in enumerate(edges(mesh.faces[i]))
                face_edge[i][j] = searchsortedfirst(mesh.edges, Tuple(edge))
            end
        end
    end
    return [Tuple(sort(conn)) for conn in face_edge]::Vector{<:Tuple{Vararg{I, N} where N}}
end

function faces_materialized(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                    I <: Unsigned}
    return face_materialized.(mesh, mesh.faces)::Vector{<:Face_2D{T}}
end

function face_materialized(mesh::UnstructuredMesh_2D{T, I}, 
                           face::NTuple{4, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Triangle_2D(get_face_points(mesh, face))
end

function face_materialized(mesh::UnstructuredMesh_2D{T, I}, 
                           face::NTuple{5, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Quadrilateral_2D(get_face_points(mesh, face))
end

function face_materialized(mesh::UnstructuredMesh_2D{T, I}, 
                           face::NTuple{7, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Triangle6_2D(get_face_points(mesh, face))
end

function face_materialized(mesh::UnstructuredMesh_2D{T, I}, 
                           face::NTuple{9, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Quadrilateral8_2D(get_face_points(mesh, face))
end

function find_face(p::Point_2D{T}, mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                           I <: Unsigned}
    if 0 < length(mesh.faces_materialized)
        return I(find_face_explicit(p, mesh.faces_materialized))
    else
        return I(find_face_implicit(p, mesh, mesh.faces))
    end
end

function find_face_explicit(p::Point_2D{T}, 
                            faces::Vector{<:Union{Triangle_2D{T}, Quadrilateral_2D{T}}}
                           ) where {T <: AbstractFloat}
    for i = 1:length(faces)
        if p ∈  faces[i]
            return i
        end
    end
    return 0
end

function find_face_explicit(p::Point_2D{T}, 
                            faces::Vector{<:Union{Triangle6_2D{T}, Quadrilateral8_2D{T}}}
                           ) where {T <: AbstractFloat}
    for i = 1:length(faces)
        if p ∈  faces[i]
            return i
        end
    end
    return 0
end

function find_face_explicit(p::Point_2D{T}, 
                            faces::Vector{Face_2D{T}}
                           ) where {T <: AbstractFloat}
    for i = 1:length(faces)
        if p ∈  faces[i]
            return i
        end
    end
    return 0
end

function find_face_implicit(p::Point_2D{T}, 
                            mesh::UnstructuredMesh_2D{T, I},
                            faces::Vector{<:Union{NTuple{4, I}, NTuple{5, I}}}
                            ) where {T <: AbstractFloat, I <: Unsigned}
    for i = 1:length(faces)
        face = faces[i]
        bool = false
        if face isa NTuple{4, I} # Triangle
            bool = p ∈ Triangle_2D(get_face_points(mesh, face))
        else # Quadrilateral
            bool = p ∈ Quadrilateral_2D(get_face_points(mesh, face))
        end
        if bool
            return i
        end
    end
    return 0
end

function find_face_implicit(p::Point_2D{T}, 
                            mesh::UnstructuredMesh_2D{T, I},
                            faces::Vector{<:Union{NTuple{7, I}, NTuple{9, I}}}
                            ) where {T <: AbstractFloat, I <: Unsigned}
    for i = 1:length(faces)
        face = faces[i]
        bool = false
        if face isa NTuple{7, I} # Triangle6
            bool = p ∈ Triangle6_2D(get_face_points(mesh, face))
        else # Quadrilateral8
            bool = p ∈ Quadrilateral8_2D(get_face_points(mesh, face))
        end
        if bool
            return i
        end
    end
    return 0
end

function find_face_implicit(p::Point_2D{T}, 
                            mesh::UnstructuredMesh_2D{T, I},
                            faces::Vector{<:Tuple{Vararg{I, N} where N}}
                            ) where {T <: AbstractFloat, I <: Unsigned}
    for i = 1:length(faces)
        face = faces[i]
        bool = false
        if face isa NTuple{4, I} # Triangle
            bool = p ∈ Triangle_2D(get_face_points(mesh, face))
        elseif face isa NTuple{5, I} # Quadrilateral
            bool = p ∈ Quadrilateral_2D(get_face_points(mesh, face))
        elseif face isa NTuple{7, I} # Triangle6
            bool = p ∈ Triangle6_2D(get_face_points(mesh, face))
        else # Quadrilateral8
            bool = p ∈ Quadrilateral8_2D(get_face_points(mesh, face))
        end
        if bool
            return i
        end
    end
    return 0
end

function get_edge_points(mesh::UnstructuredMesh_2D{T, I}, 
                         edge::NTuple{2, I}) where {T <: AbstractFloat, I <: Unsigned}
    return (mesh.points[edge[1]], mesh.points[edge[2]])
end

function get_edge_points(mesh::UnstructuredMesh_2D{T, I}, 
                         edge::NTuple{3, I}) where {T <: AbstractFloat, I <: Unsigned}
    return (mesh.points[edge[1]], 
            mesh.points[edge[2]], 
            mesh.points[edge[3]]
           )
end

# This gets called enough that ugly code for optimization makes sense
function get_face_points(mesh::UnstructuredMesh_2D{T, I}, 
                         face::NTuple{4, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:4])])::NTuple{3, Point_2D{T}}
end

function get_face_points(mesh::UnstructuredMesh_2D{T, I}, 
                         face::NTuple{5, I}) where {T <: AbstractFloat, I <: Unsigned} 
    return Tuple(mesh.points[collect(face[2:5])])::NTuple{4, Point_2D{T}}
end

function get_face_points(mesh::UnstructuredMesh_2D{T, I}, 
                         face::NTuple{7, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:7])])::NTuple{6, Point_2D{T}}
end

function get_face_points(mesh::UnstructuredMesh_2D{T, I}, 
                         face::NTuple{9, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:9])])::NTuple{8, Point_2D{T}}
end

function get_intersection_algorithm(mesh::UnstructuredMesh_2D)
    if length(mesh.edges_materialized) !== 0
        return "Edges - Materialized"
    elseif length(mesh.edges) !== 0
        return "Edges - Implicit"
    elseif length(mesh.faces_materialized) !== 0
        return "Faces - Materialized"
    else
        return "Faces - Implicit"
    end
end

function insert_boundary_edge!(edge_index::I, edge_indices::Vector{I}, p_ref::Point_2D{T},
        mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}

    # Compute the minimum distance from the edge to be inserted to the reference point
    edge_points = get_edge_points(mesh, mesh.edges[edge_index])
    insertion_distance = minimum([ distance(p_ref, p_edge) for p_edge in edge_points ])
    # Loop through the edge indices until an edge with greater distance from the reference point
    # is found, then insert
    nindices = length(edge_indices)
    for i = 1:nindices
        iedge = edge_indices[i]
        iedge_points = get_edge_points(mesh, mesh.edges[iedge])
        iedge_distance = minimum([ distance(p_ref, p_edge) for p_edge in iedge_points ])
        if insertion_distance < iedge_distance
            insert!(edge_indices, i, edge_index)
            return
        end
    end
    insert!(edge_indices, nindices+1, edge_index)    
end

function intersect(l::LineSegment_2D{T}, mesh::UnstructuredMesh_2D{T}
                   ) where {T <: AbstractFloat}
    if length(mesh.edges) !== 0 || length(mesh.edges_materialized) !== 0
        return intersect_edges(l, mesh)
    else
        return intersect_faces(l, mesh)
    end
end

function intersect_edges(l::LineSegment_2D{T}, 
                         mesh::UnstructuredMesh_2D{T, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    if length(mesh.edges_materialized) !== 0
        intersection_points = intersect_edges_explicit(l, mesh.edges_materialized)
        return sort_intersection_points(l, intersection_points)
    else
        intersection_points = intersect_edges_implicit(l, mesh, mesh.edges)
        return sort_intersection_points(l, intersection_points)
    end
end

function intersect_edges_explicit(l::LineSegment_2D{T}, 
                                  edges::Vector{LineSegment_2D{T}}) where {T <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for edge in edges
        npoints, points = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints 
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_edges_explicit(l::LineSegment_2D{T}, 
                                  edges::Vector{QuadraticSegment_2D{T}}) where {T <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for edge in edges
        npoints, points = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints 
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_edges_implicit(l::LineSegment_2D{T}, 
                                  mesh::UnstructuredMesh_2D{T, I},
                                  edges::Vector{NTuple{2, I}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for edge in edges              
        npoints, points = l ∩ LineSegment_2D(get_edge_points(mesh, edge))
        if 0 < npoints 
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_edges_implicit(l::LineSegment_2D{T}, 
                                  mesh::UnstructuredMesh_2D{T, I},
                                  edges::Vector{NTuple{3, I}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for edge in edges              
        npoints, points = l ∩ QuadraticSegment_2D(get_edge_points(mesh, edge))
        if 0 < npoints 
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_faces(l::LineSegment_2D{T}, 
                         mesh::UnstructuredMesh_2D{T, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    if length(mesh.faces_materialized) !== 0
        intersection_points = intersect_faces_explicit(l, mesh.faces_materialized)
        return sort_intersection_points(l, intersection_points)
    else
        # Check if any of the face types are unsupported
        unsupported = sum(x->x[1] ∉  UnstructuredMesh_2D_cell_types, mesh.faces)
        if 0 < unsupported
            @warn "Mesh contains an unsupported face type"
        end
        intersection_points = intersect_faces_implicit(l, mesh, mesh.faces)
        return sort_intersection_points(l, intersection_points)
    end
end

function intersect_faces_explicit(l::LineSegment_2D{T}, 
                                  faces::Vector{<:Union{Triangle_2D{T}, Quadrilateral_2D{T}}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints 
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_faces_explicit(l::LineSegment_2D{T}, 
                                  faces::Vector{<:Union{Triangle6_2D{T}, Quadrilateral8_2D{T}}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints 
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_faces_explicit(l::LineSegment_2D{T}, 
                                  faces::Vector{<:Face_2D{T}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints 
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end


function intersect_faces_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  faces::Vector{<:Union{NTuple{4, I}, NTuple{5, I}}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for face in faces              
        type_id = face[1]
        if type_id == 5 # Triangle
            npoints, points = l ∩ Triangle_2D(get_face_points(mesh, 
                                                              face::NTuple{4, I})::NTuple{3, Point_2D{T}})
            # If the intersections yields 1 or more points, push those points to the array of points
            if 0 < npoints 
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 9 # Quadrilateral
            npoints, points = l ∩ Quadrilateral_2D(get_face_points(mesh, 
                                                                   face::NTuple{5, I})::NTuple{4, Point_2D{T}})
            if 0 < npoints 
                append!(intersection_points, collect(points[1:npoints]))
            end
        end
    end
    return intersection_points
end

function intersect_faces_implicit(l::LineSegment_2D{T}, 
                                  mesh::UnstructuredMesh_2D{T, I},
                                  faces::Vector{<:Union{NTuple{7, I}, NTuple{9, I}}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for face in faces              
        type_id = face[1]
        if type_id == 22 # Triangle6
            npoints, points = l ∩ Triangle6_2D(get_face_points(mesh, 
                                                               face::NTuple{7, I})::NTuple{6, Point_2D{T}})
            if 0 < npoints 
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 23 # Quadrilateral8
            npoints, points = l ∩ Quadrilateral8_2D(get_face_points(mesh,
                                                                    face::NTuple{9, I})::NTuple{8, Point_2D{T}})
            if 0 < npoints 
                append!(intersection_points, collect(points[1:npoints]))
            end
        end
    end
    return intersection_points
end

function intersect_faces_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  faces::Vector{<:Tuple{Vararg{I, N} where N}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for face in faces              
        type_id = face[1]
        if type_id == 5 # Triangle
            npoints, points = l ∩ Triangle_2D(get_face_points(mesh, 
                                                              face::NTuple{4, I})::NTuple{3, Point_2D{T}})
            # If the intersections yields 1 or more points, push those points to the array of points
            if 0 < npoints 
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 9 # Quadrilateral
            npoints, points = l ∩ Quadrilateral_2D(get_face_points(mesh, 
                                                                   face::NTuple{5, I})::NTuple{4, Point_2D{T}})
            if 0 < npoints 
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 22 # Triangle6
            npoints, points = l ∩ Triangle6_2D(get_face_points(mesh, 
                                                               face::NTuple{7, I})::NTuple{6, Point_2D{T}})
            if 0 < npoints 
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 23 # Quadrilateral8
            npoints, points = l ∩ Quadrilateral8_2D(get_face_points(mesh,
                                                                    face::NTuple{9, I})::NTuple{8, Point_2D{T}})
            if 0 < npoints 
                append!(intersection_points, collect(points[1:npoints]))
            end
        end
    end
    return intersection_points
end

function num_edges(face::Tuple{Vararg{I}}) where {I <: Unsigned}
    cell_type = face[1]
    if cell_type == 5 || cell_type == 22
        return I(3)
    elseif cell_type == 9 || cell_type == 23
        return I(4)
    else
        @error "Unsupported cell type."
        return I(0)
    end
end

function Base.show(io::IO, mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    println(io, "UnstructuredMesh_2D{$T}{$I}")
    name = mesh.name
    println(io, "  ├─ Name      : $name")
    size_MB = Base.summarysize(mesh)/1E6
    println(io, "  ├─ Size (MB) : $size_MB")
    npoints = length(mesh.points)
    println(io, "  ├─ Points    : $npoints")
    if 0 < length(mesh.edges_materialized)
        nedges = length(mesh.edges_materialized)
        nlin   = sum(x->x isa LineSegment_2D,  mesh.edges_materialized)       
        nquad  = sum(x->x isa QuadraticSegment_2D,  mesh.edges_materialized)       
    elseif 0 < length(mesh.edges)
        nedges = length(mesh.edges)
        nlin   = sum(x->length(x) == 2,  mesh.edges)
        nquad  = sum(x->length(x) == 3,  mesh.edges)
    else
        nedges = 0
        nlin = 0
        nquad = 0
    end
    ematerialized = length(mesh.edges_materialized) !== 0
    println(io, "  ├─ Edges     : $nedges")
    println(io, "  │  ├─ Linear         : $nlin")
    println(io, "  │  ├─ Quadratic      : $nquad")
    println(io, "  │  └─ Materialized?  : $ematerialized")
    nfaces = length(mesh.faces)
    println(io, "  ├─ Faces     : $nfaces")
    if 0 < nfaces
        ntri   = sum(x->x[1] == 5,  mesh.faces)
        nquad  = sum(x->x[1] == 9,  mesh.faces)
        ntri6  = sum(x->x[1] == 22, mesh.faces)
        nquad8 = sum(x->x[1] == 23, mesh.faces)
    else
        ntri   = 0 
        nquad  = 0 
        ntri6  = 0 
        nquad8 = 0 
    end
    fmaterialized = length(mesh.faces_materialized) !== 0
    println(io, "  │  ├─ Triangle       : $ntri")
    println(io, "  │  ├─ Quadrilateral  : $nquad")
    println(io, "  │  ├─ Triangle6      : $ntri6")
    println(io, "  │  ├─ Quadrilateral8 : $nquad8")
    println(io, "  │  └─ Materialized?  : $fmaterialized")
    nface_sets = length(keys(mesh.face_sets))
    ef_con = 0 < length(mesh.edge_face_connectivity)
    fe_con = 0 < length(mesh.face_edge_connectivity)
    println(io, "  ├─ Connectivity")
    println(io, "  │  ├─ Edge/Face : $ef_con")
    println(io, "  │  └─ Face/Edge : $fe_con")
    if 0 < length(mesh.boundary_edges)
        nbsides = length(mesh.boundary_edges)
        nbedges = 0
        for side in mesh.boundary_edges
            nbedges += length(side)
        end
        println(io, "  ├─ Boundary edges")
        println(io, "  │  ├─ Edges : $nbedges")
        println(io, "  │  └─ Sides : $nbsides")
    else
        nbsides = 0 
        nbedges = 0
        println(io, "  ├─ Boundary edges")
        println(io, "  │  ├─ Edges : $nbedges")
        println(io, "  │  └─ Sides : $nbsides")
    end
    println(io, "  └─ Face sets : $nface_sets")
end

function sort_intersection_points(l::LineSegment_2D{T},
                                  points::Vector{Point_2D{T}}) where {T <: AbstractFloat}
    if 0 < length(points)
        # Sort the points based upon their distance to the first point in the line
        distances = distance.(l.points[1], points)
        sorted_pairs = sort(collect(zip(distances, points)); by=first)
        points_sorted::Vector{Point_2D{T}} = getindex.(sorted_pairs, 2)
        # Remove duplicate points
        points_reduced::Vector{Point_2D{T}} = [points_sorted[1]]
        npoints::Int64 = length(points_sorted)
        for i = 2:npoints
            if minimum_segment_length < distance(last(points_reduced), points_sorted[i])
                push!(points_reduced, points_sorted[i])
            end
        end
        return points_reduced::Vector{Point_2D{T}}
    else
        return points
    end
end

function submesh(mesh::UnstructuredMesh_2D{T, I}, 
                 face_ids::Set{I};
                 name::String = "DefaultMeshName") where {T<:AbstractFloat, I <: Unsigned}
    # Setup faces and get all vertex ids
    faces = Vector{Vector{I}}(undef, length(face_ids))
    vertex_ids = Set{I}()
    for (i, face_id) in enumerate(face_ids)
        face = collect(mesh.faces[face_id])
        faces[i] = face
        union!(vertex_ids, Set{I}(face[2:length(face)]))
    end
    # Need to remap vertex ids in faces to new ids
    vertex_ids_sorted = sort(collect(vertex_ids))
    vertex_map = Dict{I, I}()
    for (i,v) in enumerate(vertex_ids_sorted)
        vertex_map[v] = i
    end
    points = Vector{Point_2D{T}}(undef, length(vertex_ids_sorted))
    for (i, v) in enumerate(vertex_ids_sorted)
        points[i] = mesh.points[v]
    end
    # remap vertex ids in faces
    for face in faces
        for (i, v) in enumerate(face[2:length(face)])
            face[i + 1] = vertex_map[v] 
        end
    end
    # At this point we have points, faces, & name.
    # Just need to get the face sets
    face_sets = Dict{String, Set{I}}()
    for face_set_name in keys(mesh.face_sets)
        set_intersection = intersect(mesh.face_sets[face_set_name], face_ids)
        if length(set_intersection) !== 0
            face_sets[face_set_name] = set_intersection
        end
    end
    # Need to remap face ids in face sets
    face_map = Dict{I, I}()
    for (i,f) in enumerate(face_ids)
        face_map[f] = i
    end
    for face_set_name in keys(face_sets)                                       
        new_set = Set{I}()
        for fid in face_sets[face_set_name]
            union!(new_set, face_map[fid])
        end
        face_sets[face_set_name] = new_set
    end
    return UnstructuredMesh_2D{T, I}(name = name,
                                     points = points,
                                     faces = [Tuple(f) for f in faces],
                                     face_sets = face_sets
                                    )
end

function submesh(mesh::UnstructuredMesh_2D{T, I}, 
                 set_name::String) where {T <: AbstractFloat, I <: Unsigned}
    @debug "Creating submesh for '$set_name'"
    face_ids = mesh.face_sets[set_name]
    return submesh(mesh, face_ids, name = set_name) 
end

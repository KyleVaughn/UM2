export import_mesh, export_mesh
"""
    import_mesh(path::String)
    import_mesh(path::String, ::Type{T}=Float64) where {T<:AbstractFloat}

Import a mesh from file. The float type of the mesh may be specified with a second argument.
File type is inferred from the extension.
"""
function import_mesh(path::String, ::Type{T}) where {T<:AbstractFloat}
    @info "Reading "*path
    if endswith(uppercase(path), ".INP")
        return read_abaqus(path, T)
    else
        error("Could not determine mesh file type from extension")
    end
end

import_mesh(path::String) = import_mesh(path, Float64)

function export_mesh(mesh::PolytopeVertexMesh, path::String)
    @info "Writing "*path
    if endswith(uppercase(path), ".XDMF")
        return write_xdmf(mesh, path)
    else
        error("Could not determine mesh file type from extension")
    end
end

function _create_mesh_from_elements(is3D::Bool, 
                                    name::String, 
                                    points::Vector{Point{3,T}}, 
                                    element_vecs::Vector{Vector{UInt64}},
                                    element_sets::Dict{String, BitSet}) where {T}
    # Determine element lengths
    element_lengths = Int64[]
    for element in element_vecs
        l = length(element)
        if l ∉ element_lengths
            push!(element_lengths, l)
        end
    end
    sort!(element_lengths)
    U = _select_mesh_UInt_type(max(length(points), length(element_vecs)))
    if !is3D # is2D
        K = 2
        # Verify all points are approximately the same z-coordinate
        z = points[1][3]
        for i ∈ 2:length(points)
            if 1e-4 < abs(z - points[i][3])
                error("Points of 2D mesh do not lie in the same plane")
            end
        end
        points2D = convert.(Point{2,T}, points) # convert to 2D
        if all(x->x < 6, element_lengths) # Linear mesh
            P = 1
        else
            P = 2
        end
        polytopes = [ Polytope{K,P,length(elem),U}(elem) for elem in element_vecs ]
        # Verify convexity
        return PolytopeVertexMesh(name, points2D, polytopes, element_sets)
    else
        K = 3
        if all(x->x < 10, element_lengths) # Linear mesh
            P = 1
        else # Quadratic Mesh
            P = 2
        end
        polytopes = [ Polytope{K,P,length(elem),U}(elem) for elem in element_vecs ]
        # Verify convexity
        return PolytopeVertexMesh(name, points, polytopes, element_sets)
    end
    error("Invalid mesh type")
end

function _select_mesh_UInt_type(N::Int64)
    if N ≤ typemax(UInt16) 
        U = UInt16
    elseif N ≤ typemax(UInt32) 
        U = UInt32
    elseif N ≤ typemax(UInt64) 
        U = UInt64
    else 
        error("That's a big mesh! Number of edges exceeds typemax(UInt64)")
    end
    return U
end

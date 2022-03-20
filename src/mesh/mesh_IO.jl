"""
    import_mesh(path::String)
    import_mesh(path::String, ::Type{T}=Float64) where {T<:AbstractFloat}

Import a mesh from file. The float type of the mesh may be specified with a second argument.
File type is inferred from the extension.
"""
function import_mesh(path::String, ::Type{T}=Float64) where {T<:AbstractFloat}
    @info "Reading "*path
    if endswith(path, ".inp")
        return read_abaqus(path, T)
    else
        error("Could not determine mesh file type from extension")
    end
end

function _create_mesh_from_elements(is3D::Bool, 
                                    name::String, 
                                    points::Vector{Point3D{T}}, 
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
        # Verify all points are approximately the same z-coordinate
        z = points[1].z
        for i ∈ 2:length(points)
            if 1e-4 < abs(z - points[i].z)
                error("Points of 2D mesh do not lie in the same plane")
            end
        end
        points2D = convert.(Point2D{T}, points) # convert to 2D
        if all(x->x < 6, element_lengths) # Linear mesh
            if element_lengths == [3]
                faces = [ SVector{3, U}(f) for f in element_vecs]
                return TriangleMesh{T, U}(name = name,
                                          points = points2D,
                                          faces = faces,
                                          face_sets = element_sets)
            elseif element_lengths == [4]
                faces = [ SVector{4, U}(f) for f in element_vecs]
                return QuadrilateralMesh{T, U}(name = name,
                                               points = points2D,
                                               faces = faces, 
                                               face_sets = element_sets)
            elseif element_lengths == [3, 4]
                faces = [ SVector{length(f), U}(f) for f in element_vecs]
                return PolygonMesh{T, U}(name = name,
                                               points = points2D,
                                               faces = faces,
                                               face_sets = element_sets)
            end
            # Verify convexity
        else # Quadratic Mesh
            if element_lengths == [6]
                faces = [ SVector{6, U}(f) for f in element_vecs]
                return QuadraticTriangleMesh{T, U}(name = name,
                                                   points = points2D,
                                                   faces = faces, 
                                                   face_sets = element_sets)

            elseif element_lengths == [8]
                faces = [ SVector{8, U}(f) for f in element_vecs]
                return QuadraticQuadrilateralMesh{T, U}(name = name,
                                                        points = points2D,
                                                        faces = faces, 
                                                        face_sets = element_sets)

            elseif element_lengths == [6, 8]
                faces = [ SVector{length(f), U}(f) for f in element_vecs]
                return QuadraticPolygonMesh{T, U}(name = name,
                                                  points = points2D,
                                                  faces = faces, 
                                                  face_sets = element_sets)
            end
        end
    else # is3D
        if all(x->x < 10, element_lengths) # Linear mesh
            if element_lengths == [4]
                cells = [ SVector{3, U}(f) for f in element_vecs]
                return TetrahedonMesh{T, U}(name = name,
                                            points = points,
                                            cells = cells,
                                            cell_sets = element_sets)
            elseif element_lengths == [8]
                cells = [ SVector{4, U}(f) for f in element_vecs]
                return HexahedonMesh{T, U}(name = name,
                                           points = points,
                                           cells = cells, 
                                           cell_sets = element_sets)
            elseif element_lengths == [4, 8]
                cells = [ SVector{length(f), U}(f) for f in element_vecs]
                return PolyhedronMesh{T, U}(name = name,
                                            points = points,
                                            cells = cells,
                                            cell_sets = element_sets)
            end
        else # Quadratic Mesh
            if element_lengths == [10]
                cells = [ SVector{3, U}(f) for f in element_vecs]
                return QuadraticTetrahedonMesh{T, U}(name = name,
                                                     points = points,
                                                     cells = cells, 
                                                     cell_sets = element_sets)

            elseif element_lengths == [20]
                cells = [ SVector{3, U}(f) for f in element_vecs]
                return QuadraticHexahedonMesh{T, U}(name = name,
                                                        points = points,
                                                        cells = cells, 
                                                        cell_sets = element_sets)

            elseif element_lengths == [10, 20]
                cells = [ SVector{3, U}(f) for f in element_vecs]
                return QuadraticPolyhedronMesh{T, U}(name = name,
                                                     points = points,
                                                     cells = cells, 
                                                     cell_sets = element_sets)
            end
        end
    end
    error("Invalid mesh type")
end















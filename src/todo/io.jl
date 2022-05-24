#function _create_mesh_from_elements(is3D::Bool, 
#                                    name::String, 
#                                    points::Vector{Point{3,T}}, 
#                                    element_vecs::Vector{Vector{UInt64}},
#                                    element_sets::Dict{String, BitSet}) where {T}
#    # Determine element lengths
#    element_lengths = Int64[]
#    for element in element_vecs
#        l = length(element)
#        if l ∉ element_lengths
#            push!(element_lengths, l)
#        end
#    end
#    sort!(element_lengths)
#    U = _select_UInt_type(length(points))
#    if !is3D # is2D
#        K = 2
#        # Verify all points are approximately the same z-coordinate
#        if any(i->ϵ_Point < abs(points[1][3] - points[i][3]), 2:length(points))
#            error("Points of 2D mesh do not lie in the same plane")
#        end
#        points2D = convert.(Point{2,T}, points) # convert to 2D
#        if all(x->x < 6, element_lengths) # Linear mesh
#            P = 1
#        else
#            P = 2
#        end
#        polytopes = [ Polytope{K,P,length(elem),U}(elem) for elem in element_vecs ]
#        # Verify convexity
#        return PolytopeVertexMesh(name, points2D, polytopes, element_sets)
#    else
#        K = 3
#        if all(x->x < 10, element_lengths) # Linear mesh
#            P = 1
#        else # Quadratic Mesh
#            P = 2
#        end
#        polytopes = [ Polytope{K,P,length(elem),U}(elem) for elem in element_vecs ]
#        # Verify convexity
#        return PolytopeVertexMesh(name, points, polytopes, element_sets)
#    end
#    error("Invalid mesh type")
#end

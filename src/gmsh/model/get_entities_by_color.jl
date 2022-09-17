export get_entities_by_color

"""
    get_entities_by_color(dim::Int64=-1)

Return a dictionary of RGBA color keys and Vector{NTuple{2,Int32}} 
values, representing the dim tags of the entities with that color.
If `dim` == -1, return all entities, otherwise return only entities of dimension `dim`.

Note, entities in gmsh have color (0, 0, 255, 0) by default, so these entities are excluded.
"""
function get_entities_by_color(dim::Int64 = -1)
    gmsh_default_color = RGBA(0, 0, 255, 0)
    # Assumes no entity has two colors, so using a Vector
    # makes more sense than a Set.
    color_dict = Dict{RGBA, Vector{NTuple{2, Int32}}}()
    for ent in gmsh.model.get_entities(dim)
        edim, tag = ent
        color = RGBA(gmsh.model.get_color(edim, tag))
        if color != gmsh_default_color
            if color ∉ keys(color_dict)
                color_dict[color] = NTuple{2, Int32}[]
            end
            push!(color_dict[color], ent)
        end
    end
    return color_dict
end

export get_entities_by_color

"""
    get_entities_by_color(dim::Int64=-1)

Return a dictionary of NTuple{4, Int32} RGBA color keys and Vector{NTuple{2,Int32}} 
values, representing the dim tags of the entities with that color.
If `dim` == -1, return all entities, otherwise return only entities of dimension `dim`.

Note, entities in gmsh have color (0, 0, 255, 0) by default, so these entities are excluded.
"""
function get_entities_by_color(dim::Int64 = -1)
    color_dict = Dict{NTuple{4, Int32}, Vector{NTuple{2, Int32}}}()
    for ent in gmsh.model.get_entities(dim)
        edim, tag = ent
        color = gmsh.model.get_color(edim, tag)
        if color != (0, 0, 255, 0) # Default color when unassigned
            if color ∉ keys(color_dict)
                color_dict[color] = NTuple{2, Int32}[]
            end
            push!(color_dict[color], ent)
        end
    end
    return color_dict
end

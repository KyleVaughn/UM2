"""
    entities_by_color(dim::Int64=2)

Return a dictionary of entities of dimension `dim`, sorted by color.
"""
function get_entities_by_color(dim::Int64=2)
    color_dict = Dict{NTuple{4, Int32}, Vector{Int32}}()
    for ent in gmsh.model.get_entities(dim)
        tag = ent[2]
        color = gmsh.model.get_color(dim, tag)
        if color != (0, 0, 255, 0) # Default color when unassigned
            if color ∉ keys(color_dict)
                color_dict[color] = Int32[]
            end
            push!(color_dict[color], tag)
        end
    end
    return color_dict
end

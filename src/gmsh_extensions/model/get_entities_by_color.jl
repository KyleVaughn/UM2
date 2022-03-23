"""
    entities_by_color(dim::Int64=-1)

Return a dictionary of entities of dimension `dim`, sorted by color.
"""
function get_entities_by_color(dim::Int64=-1)
    color_dict = Dict{NTuple{4, Int32}, Vector{Tuple{Int32, Int32}}}()
    for ent in gmsh.model.get_entities(dim)
        edim, tag = ent
        color = gmsh.model.get_color(edim, tag)
        if color != (0, 0, 255, 0) # Default color when unassigned
            if color ∉ keys(color_dict)
                color_dict[color] = Tuple{Int32, Int32}[]
            end
            push!(color_dict[color], ent)
        end
    end
    return color_dict
end

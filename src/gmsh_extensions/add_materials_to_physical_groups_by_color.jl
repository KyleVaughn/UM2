@eval gmsh.model begin
    """
        gmsh.model.add_materials_to_physical_groups_by_color(
            mat_to_color::Dict{String, NTuple{4, Int64}}, 
            color_to_ent::Dict{NTuple{4, Int32}, Vector{Int32}},
            dim::Int64=2)

    Assign materials to entities of dimension `dim`, mapping material name -> 
    color -> entities. Note, each material name must contain "Material", but is
    not case-sensitive.

    See `gmsh.model.get_entities_by_color`.
    """
    function add_materials_to_physical_groups_by_color(
                mat_to_color::Dict{String, NTuple{4, Int32}}, 
                color_to_ent::Dict{NTuple{4, Int32}, Vector{Int32}},
                dim::Int64=2)

        for (name, color) in sort!(collect(mat_to_color))
            if !(occursin("MATERIAL", uppercase(name)))
                 error("Material names must contain the word 'Material'")
            end
            tags = color_to_ent[color]
            p = gmsh.model.add_physical_group(dim, tags)
            gmsh.model.set_physical_name(dim, p, name)
        end
        return sort!(collect(keys(mat_to_color)))
    end
end

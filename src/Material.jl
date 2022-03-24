mutable struct Material
    name::String
    color::NTuple{4, Int32}
    mesh_size::Float64
end

function Material(name, color::NTuple{4, Int64}, mesh_size)
    return Material(name, convert(NTuple{4, Int32}, color), mesh_size)
end

Material(name) = Material(name, (0, 0, 0, 255), 1.0)

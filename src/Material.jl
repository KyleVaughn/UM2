mutable struct Material
    name::String
    color::RGBA{N0f8}
    mesh_size::Float64
end

function Material(;name::String = "", color = RGBA{N0f8}(0, 0, 1, 1), mesh_size = 1)
    return Material(name, parse(RGBA{N0f8}, color), Float64(mesh_size))
end

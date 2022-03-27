mutable struct Material
    name::String
    color::RGBA
    mesh_size::Float64
end

function Material(;name::String = "", color = RGBA(0,0,1,1), mesh_size = 1)
    return Material(name, parse(RGBA, color), Float32(mesh_size))
end

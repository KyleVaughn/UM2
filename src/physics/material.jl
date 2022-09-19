export Material

struct Material
    name::String
    color::RGBA
#    xsec::CrossSection
#    lc::Float64
end

function Material(;name::String = "",
                  color::Union{RGBA, String} = RGBA(0, 0, 0, 255))
    if color isa String
        return Material(name, RGBA(color))
    else
        return Material(name, color)
    end
end

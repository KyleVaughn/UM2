export Material

struct Material
    name::String
    color::RGBA
#    xsec::CrossSection
#    lc::Float64
end
#
#function Material(; id::UInt8 = GLOBAL_MATERIAL_COUNTER,
#                  name::String = "",
#                  color = RGBA{N0f8}(0, 0, 0, 1),
#                  xsec::CrossSection = null_cross_section(Float64),
#                  lc::Float64 = 0.0)
#    global GLOBAL_MATERIAL_COUNTER += 1
#    return Material(id, name, parse(RGBA{N0f8}, color), xsec, lc)
#end

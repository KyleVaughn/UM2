export CrossSection
export null_cross_section

# We don't anticipate going beyond a few groups in this code, hence Σs is an SMatrix.        
# Ideally, G should be a multiple of 2 for SIMD purposes. 
struct CrossSection{G, T <: AbstractFloat, L} # L = G^2
    Σt::Vec{G, T}           # total cross section
    χ::Vec{G, T}            # probability density
    νΣf::Vec{G, T}          # fission cross section
    Σs::SMatrix{G, G, T, L} # scattering matrix
end

function null_cross_section(T)
    v = @SVector zeros(T, 0)
    mat = @SMatrix zeros(T, 0, 0)
    return CrossSection{0, T, 0}(v, v, v, mat)
end

struct CrossSection{G,T,L}
    Σt::Vec{G,T}
    Σs::SMatrix{G,G,T,L}
    χ::Vec{G,T}
    νΣf::Vec{G,T}
    fissile::Bool
end

struct MPACTLattice
    bb::AABox2D{F} # local
    module_ids::Matrix{UInt32} # global RT module ids
    id::UInt32 # global id
end

const MPACTLattices = MPACTLattice[]

function MPACTLattice(bb::AABox2D{F}, module_ids::Matrix{UInt32}=Matrix{UInt32}(undef,0,0))
    nlats = length(MPACTLattices)
    lat = MPACTLattice(bb, module_ids, UInt32(nlats + 1))
    push!(MPACTLattices, lat)
    return MPACTLattices[nlats + 1]
end

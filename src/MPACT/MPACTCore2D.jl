struct MPACTCore2D
    bb::AABox2D{F}
    lattice_ids::Matrix{UInt32} # global lattice ids
end

const MPACTCore = MPACTLattice[]

function MPACTCore2D(bb::AABox2D{F})
    nlats = length(MPACTLattices)
    lat = MPACTLattice(bb, module_ids, UInt32(nlats + 1))
    push!(MPACTLattices, lat)
    return MPACTLattices[nlats + 1]
end

# Renumber lattices. Reorder in memory, along with everything else, to match Z-order or hilbert

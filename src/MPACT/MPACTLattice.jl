struct MPACTLattice{T}
    id::UInt32 # global id
    bb::AABox2D{T} # local
    coarse_cell_ids::Matrix{UInt32} # global coarse cell ids
end

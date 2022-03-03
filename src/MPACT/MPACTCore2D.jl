struct MPACTCore2D{T}
    bb::AABox2D{T} # local
    lattice_ids::Matrix{UInt32} # global lattice ids
end

struct MPACTCoarseCell{T}
    bb::AABox2D{T} # local
    finemesh_id::UInt32 # global id of finemesh contained in coarse cell
    id::UInt32 # global id
end

const MPACTCoarseCells = MPACTCoarseCell[] 

function MPACTCoarseCell(bb::AABox2D, finemesh_id::UInt32=0x00000000)
    ncells = length(MPACTCoarseCells)
    cell = MPACTCoarseCell(bb, finemesh_id, UInt32(ncells + 1))
    push!(MPACTCoarseCells, cell)
    return MPACTCoarseCells[ncells + 1]
end

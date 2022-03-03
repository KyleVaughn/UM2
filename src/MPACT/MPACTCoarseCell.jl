struct MPACTCoarseCell
    bb::AABox2D{F} # local
    finemesh_id::UInt32 # global id of finemesh contained in coarse cell
    id::UInt32 # global id
end

function MPACTCoarseCell(bb::AABox2D)
    ncells = length(MPACTCoarseCells)
    cell = MPACTCoarseCell(bb, 0x00000000, UInt32(ncells + 1))
    push!(MPACTCoarseCells, cell)
    return MPACTCoarseCells[ncells + 1]
end

const MPACTCoarseCells = MPACTCoarseCell[] 

struct MPACTCoarseCell{T}
    id::UInt32 # global id
    bb::AABox2D{T} # local
    finemesh_id::UInt32 # global id of finemesh contained in coarse cell
end

MPACTCoarseCell(bb::AABox2D{T}) where {T} = MPACTCoarseCell(bb, 0x00000000)

const MPACTCoarseCellManager = MPACTCoarseCell[] 

getCoarseCellManager() = MPACTCoarseCellManager32

addMPACTCoarseCell!(

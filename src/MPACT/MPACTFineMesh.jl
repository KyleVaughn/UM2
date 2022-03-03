struct MPACTFineMesh{M<:UnstructureMesh2D}
    mesh::M
    id::UInt32 # global id
end

const MPACTFineMeshes = MPACTFineMesh[] 

function MPACTFineMesh(mesh::UnstructuredMesh2D, id::UInt32=0x00000000)
    nmeshes = length(MPACTFineMeshes)
    finemesh = MPACTCoarseCell(mesh, UInt32(nmeshes + 1))
    push!(MPACTFineMeshes, finemesh)
    return MPACTFineMeshes[nmeshes + 1]
end

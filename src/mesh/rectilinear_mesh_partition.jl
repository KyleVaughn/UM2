struct RectilinearMeshPartition{X, Y, T, M <: AbstractMesh}
    grid::RectilinearGrid{X,Y,0,T}
    meshes::Matrix{M} # size = (X, Y)
end

# Construct from rg + leaf_meshes, then sort the meshes into the correct array index
# based upon centroid of a face. for mesh in leaf meshes,
#
# Just deep copy to init.
#
# Need function that categorizes a point as being in a cell in the grid

export RegularPartition2

struct RegularPartition2{T, P}
    name::String
    grid::RegularGrid2{T}
    children::Matrix{P}
end

# Construct from rg + leaf_meshes, then sort the meshes into the correct array index
# based upon centroid of a face. for mesh in leaf meshes,
#
# Just deep copy to init.
#
# Need function that categorizes a point as being in a cell in the grid

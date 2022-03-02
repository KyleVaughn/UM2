# MPACT stores pin geometry in arrays with 
      #        x 1   2   3   4   5
      #      y ---------------------
      #      1 | 11| 12| 13| 14| 15|
      #        ---------------------
      #      2 | 6 | 7 | 8 | 9 | 10|
      #        ---------------------
      #      3 | 1 | 2 | 3 | 4 | 5 |
      #        *--------------------  * is where (0.0,0.0,0.0) is
      # so we need to flip the indicies of the array



# A tree representing a hierarchical mesh partition. 
# Used in the HierarchicalMeshPartition type
#
# Each node has the name of the partition and an ID.
# If the node is an internal node, it has id = 0.
# If the node is a leaf node, it has a non-zero id, corresponding to the index by 
# which the mesh may be found in the HierarchicalMeshPartition type's leaf_meshes.
struct MPACTCore2D{M, T}
    geomtree::MPACTCoreGeomTree2D{T}
    pinmeshes::Vector{M}
end

# Spit out the MPACT Core 2D into xdmf so that you can start from there next time

function interactive_MPACTCore2D_generator()
    # Assume CAD model has already been imported

end

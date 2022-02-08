# A data structure to hold a hierarchical partition of a mesh.
# Since the mesh is partitioned, we only need to store the leaves of the partition
# tree to reconstruct the mesh.
#   - partition_tree is a tree with String type data at all nodes except the leaves,
#       denoting the name of the partition. At the leaves, the tree has Int64 data,
#       denoting the index of leaf_meshes in which the leaf mesh may be found.
#   - leaf_meshes is vector of meshes
struct MeshPartition{M <: UnstructuredMesh}
    partition_tree::Tree
    leaf_meshes::Vector{M}
end

#partition mesh
# by = type, face_set

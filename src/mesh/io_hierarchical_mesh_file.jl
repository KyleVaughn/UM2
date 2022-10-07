export HierarchicalMeshFile

# HIERARCHICAL MESH FILE
# -----------------------------------------------------------------------------    
#    
# An intermediate representation of a hierarchical mesh that can be used to:
#   - read from a file
#   - write to a file
#   - convert to a mesh
# 
# Each node has an ID and the name of the partition. 
# If the node is a leaf, the ID is the index of the mesh in the leaf mesh list.
mutable struct HierarchicalMeshFile{T <: AbstractFloat, I <: Integer}
    filepath::String
    format::Int64
    partition_tree::Tree{Tuple{I, String}}    
    leaf_meshes::Vector{MeshFile{T, I}}
end

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

function partition_mesh(mesh::UnstructuredMesh2D; by::String="GRID_L")
    @info "Partitioning mesh"
    by = uppercase(by)
    # Extract set names, partition names, and max level
    set_names, partition_names, max_level = _process_partition_mesh_input(mesh, by)

    # Create a tree to store the partition hierarchy.
    root = _create_partition_tree(mesh, by, partition_names, max_level)

end

#partition mesh
# by = type, face_set
# Partition a mesh into an HRPM based upon the names of its face sets.
# Must contain face sets of the form "GRID_LN_X_Y" where N,X,Y are integers
# N is the level of the node and X,Y are indices of the mesh's location in a rectangular
# grid
# function partition_rectangularly(mesh::UnstructuredMesh_2D)
#     @info "Converting $(typeof(mesh)) into HierarchicalRectangularlyPartitionedMesh"
#     # Extract set names, grid names, and max level
#     set_names, grid_names, max_level = process_partition_rectangularly_input(mesh)
# 
#     # Create a tree to store grid relationships.
#     root = create_HRPM_tree(mesh, grid_names, max_level)
# 
#     # Construct the leaf meshes
#     leaf_meshes = create_HRPM_leaf_meshes(mesh, grid_names, max_level)
# 
#     # Construct the mesh hierarchy
#     HRPM = create_HRPM(root, leaf_meshes)
#     return HRPM
# end

# function attach_HRPM_children!(HRPM::HierarchicalRectangularlyPartitionedMesh,
#                                tree::Tree,
#                                leaf_meshes::Vector{<:UnstructuredMesh_2D})
#     for child in tree.children
#         name = child[].data
#         child_mesh = HierarchicalRectangularlyPartitionedMesh(name = name,
#                                                               parent = Ref(HRPM) )
#         for leaf_mesh in leaf_meshes
#             if name == leaf_mesh.name
#                 child_mesh.mesh[] = leaf_mesh
#             end
#         end
#         attach_HRPM_children!(child_mesh, child[], leaf_meshes)
#     end
#     return nothing
# end
# 
# # Construct the HRPM
# function create_HRPM(tree::Tree, leaf_meshes::Vector{UnstructuredMesh_2D})
#     # Construct the HRPM from the top down
#     root = HierarchicalRectangularlyPartitionedMesh( name = tree.data )
#     attach_HRPM_children!(root, tree, leaf_meshes)
#     # Add the rectangles
#     boundingbox(root)
#     return root
# end
# 
# function create_HRPM_leaf_meshes(mesh::UnstructuredMesh_2D, grid_names::Vector{String}, max_level::Int64)
#     # Generate the leaf meshes (The smallest spatially)
#     leaf_meshes = UnstructuredMesh_2D[]
#     leaf_names = String[]
#     for name in grid_names
#         level = parse(Int64, name[7])
#         if level == max_level
#             push!(leaf_names, name)
#         end
#     end
#     for name in leaf_names
#         push!(leaf_meshes, submesh(name, mesh))
#     end
#     # remove grid levels
#     for leaf_mesh in leaf_meshes
#         for name in keys(leaf_mesh.face_sets)
#             if occursin("GRID_", uppercase(name))
#                 delete!(leaf_mesh.face_sets, name)
#             end
#         end
#     end
#     return leaf_meshes
# end
# 
# Create a tree to store grid relationships.
function _create_partition_tree(mesh::UnstructuredMesh2D, by::String, 
                                partition_names::Vector{String}, max_level::Int64)
    root = Tree(mesh.name)
    current_nodes = []
    next_nodes = []
    old_partition_names = copy(partition_names)
    new_partition_names = copy(partition_names)
    # Do first level
    for partition_name in old_partition_names
        partition_level = parse(Int64, partition_name[length(by) + 1])
        if partition_level === 1
            # Add to appropriate node (root)
            push!(next_nodes, Tree(partition_name, root))
            filter!(x->x ≠ partition_name, new_partition_names)
        end
    end
    # Do all other levels:
    for level in 2:max_level
        old_partition_names = copy(new_partition_names)
        current_nodes = next_nodes
        next_nodes = []
        for partition_name in old_partition_names
            partition_level = parse(Int64, partition_name[length(by) + 1])
            if partition_level == level
                # find the parent for this partition
                partition_faces = mesh.face_sets[partition_name]
                for node in current_nodes
                    node_faces = mesh.face_sets[node.data]
                    if partition_faces ⊆ node_faces
                        push!(next_nodes, Tree(partition_name, node))
                        filter!(x->x ≠ partition_name, new_partition_names)
                        break
                    end
                end
            end
        end
    end
    return root
end

# Extract set names, partition names, and max partition level
function _process_partition_mesh_input(mesh::UnstructuredMesh2D, by::String)
    set_names = collect(keys(mesh.face_sets))
    partition_names = copy(set_names)
    for set_name in set_names
        if !occursin(by, uppercase(set_name))
            filter!(x->x ≠ set_name, partition_names)
        end
    end
    if length(partition_names) === 0
        @error "No partition face sets in mesh"
    end

    # Get the number of partition levels
    max_level = 1
    try
        for partition_name in partition_names
            # No 10+ level hierarchy
            level = parse(Int64, partition_name[length(by) + 1])
            if max_level < level
                max_level = level
            end
        end
    catch
        max_level = 1
    end

    return set_names, partition_names, max_level
end

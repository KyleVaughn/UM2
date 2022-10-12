export MPACTSpatialPartition

# Used to store the spatial partitioning information and an index mapping
# to the pin meshes.
# The core is a rectilinear partiton of lattices
# The lattices are a REGULAR partition of modules.
#   Each module must have the same dimensions, hence the regular grid.
# The modules are a rectilinear partition of pins.
# RecilinearPartition2{RegularPartition2{RectilinearGrid2}}
struct MPACTSpatialPartition
    #      y
    #      ^
    # j = 3|
    # j = 2|
    # j = 1|
    #       _____________> x
    #       i=1  i=2  i=3
    core::RectPart2{    # Core == Rectilinear grid of lattices
           RegPart2{    # Lattice == Regular grid of modules. NOTE: REGULAR GRID
            RectPart2{  # Module == Rectilinear grid of pins
             UM_I}}}    # Pin == Unstructured mesh. But we only store the index.

    # No input checking is required since:
    #   -The rectilinear partition ensures that the lattices align with
    #     each other.
    #   -The regular partition ensures that the modules align with each other and
    #     that the modules are the same size.
    #   -The rectilinear grid ensures that the pins align with each other.
end

function MPACTSpatialPartition(module_grid::RectilinearGrid2)
    module_dims = size(module_grid) .- 1
    module_children = zeros(UM_I, module_dims[1], module_dims[2])
    rt_module = RectilinearPartition2(UM_I(1), "Module_00001",
                                      module_grid, module_children)

    lattice_grid = RegularGrid2(bounding_box(rt_module))
    lattice_children = Matrix{RectilinearPartition2{UM_I}}(undef, 1, 1)
    lattice_children[1, 1] = rt_module
    lattice = RegularPartition2(UM_I(1), "Lattice_00001", 
                                lattice_grid, lattice_children)

    core_grid = RectilinearGrid(bounding_box(lattice))
    core_children = Matrix{RegPart2{RectPart2{UM_I}}}(undef, 1, 1)
    core_children[1, 1] = lattice
    core = RectilinearPartition2(UM_I(1), "Core", 
                                 core_grid, core_children)

    return MPACTSpatialPartition(core)
end
#
#funtion MPACTSpatialPartition(module_grid::Matrix{RectilinearGrid2{T}}) where {T}
#    lattice_grid = Matrix{Matrix{RectilinearGrid2{T}}}(undef, 1, 1)
#    lattice_grid[1, 1] = module_grid
#    return MPACTSpatialPartition(lattice_grid)
#end

function MPACTSpatialPartition(hm::HierarchicalMesh)
    # Build the spatial partitions from the bottom up.
    # Group the pins into modules
    pin_bbs = map(x->bounding_box(x), hm.leaf_meshes)
    module_nodes = nodes_at_level(hm.partition_tree, 2)
    module_children_ids = map(mod_node->getindex.(
                                          data.(
                                            children(mod_node)
                                           ), # data
                                         1), # getindex 
                              module_nodes)
    module_rect_partitions = map(child_ids->RectPart2(pin_bbs[child_ids]),
                                 module_children_ids)
    # Correct id and name to each module rectilinear partition
    for (i, mod) in enumerate(module_rect_partitions)
        mod_data = module_nodes[i].data
        module_rect_partitions[i] = RectPart2(mod_data[1], mod_data[2], mod.grid, mod.children)
    end
    # The modules are now ready to be grouped into lattices
    modules = module_rect_partitions

    # Group the modules into lattices
    module_bbs = map(x->bounding_box(x), module_rect_partitions)
    lattice_nodes = nodes_at_level(hm.partition_tree, 1)
    lattice_children_ids = map(lattice_node->getindex.(
                                          data.(
                                            children(lattice_node)
                                           ), # data
                                         1), # getindex 
                              lattice_nodes)
    # No routine for converting a vector of BB into a regular grid, so we
    # have to do it manually by first creating a rectilinear grid.
    lattice_rect_partitions = map(child_ids->RectPart2(module_bbs[child_ids]),
                                 lattice_children_ids)
    # Correct id and name to each lattice rectilinear partition
    for (i, lat) in enumerate(lattice_rect_partitions)
        lat_data = lattice_nodes[i].data
        lattice_rect_partitions[i] = RectPart2(lat_data[1], lat_data[2], lat.grid, lat.children)
    end
    # Create the regular partitions
    lattices = map(i->begin
                          lat = lattice_rect_partitions[i]
                          RegPart2(lat.id, lat.name, RegularGrid(lat.grid), modules[lat.children]) 
                      end,
                      1:length(lattice_rect_partitions))

    # Group the lattices into the core
    lattice_bbs = map(x->bounding_box(x), lattices)
    core_rect_partition = RectPart2(lattice_bbs)
    # Correct id and name 
    core_node = root(hm.partition_tree)
    core_data = core_node.data
    core = RectPart2(core_data[1], core_data[2], core_rect_partition.grid, lattices[core_rect_partition.children])

    return MPACTSpatialPartition(core)
end

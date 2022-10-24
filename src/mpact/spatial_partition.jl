export MPACTSpatialPartition

struct MPACTSpatialPartition
    core::RectilinearPartition2{UM_I}
    lattices::Vector{RegularPartition2{UM_I}}
    modules::Vector{RectilinearPartition2{UM_I}}

    function MPACTSpatialPartition(core::RectilinearPartition2{UM_I},
                                   lattices::Vector{RegularPartition2{UM_I}},
                                   modules::Vector{RectilinearPartition2{UM_I}})
        # Check that the core is the same size as the sum of the lattices
        core_bb = bounding_box(core)
        lattices_bb = mapreduce(bounding_box, union, lattices)
        if core_bb != lattices_bb
            throw(ArgumentError("The core and lattices do not have the same size"))
        end
        # Check that the lattices are aligned in the core children matrix
        core_children = children(core)
        core_size = size(core_children)
        for j in 1:core_size[2] - 1, i in 1:core_size[1] - 1
            # Aligned in x
            if all(x_coords(bounding_box(lattices[core_children[i, j]])) .≉ 
                   x_coords(bounding_box(lattices[core_children[i, j + 1]])))
                throw(ArgumentError("The lattices are not aligned in the core children matrix"))
            end
            # Aligned in y
            if all(y_coords(bounding_box(lattices[core_children[i, j]])) .≉ 
                   y_coords(bounding_box(lattices[core_children[i + 1, j]])))
                throw(ArgumentError("The lattices are not aligned in the core children matrix"))
            end
        end
        # For each lattice, perform the same as above with the modules
        for lattice in lattices
            lattice_bb = bounding_box(lattice)
            lattice_children = children(lattice)
            modules_bb = mapreduce(bounding_box, union, modules[lattice_children])
            if lattice_bb != modules_bb
                throw(ArgumentError("The lattice and modules do not have the same size"))
            end
            lattice_size = size(lattice_children)
            for j in 1:lattice_size[2] - 1, i in 1:lattice_size[1] - 1
                # Aligned in x
                if all(x_coords(bounding_box(modules[lattice_children[i, j]])) .≉ 
                       x_coords(bounding_box(modules[lattice_children[i, j + 1]])))
                    throw(ArgumentError("The modules are not aligned in the lattice children matrix"))
                end
                # Aligned in y
                if all(y_coords(bounding_box(modules[lattice_children[i, j]])) .≉ 
                       y_coords(bounding_box(modules[lattice_children[i + 1, j]])))
                    throw(ArgumentError("The modules are not aligned in the lattice children matrix"))
                end
            end
        end
        # We assume that the pins are aligned within the modules
        return new(core, lattices, modules)
    end
end

function MPACTSpatialPartition(module_grid::RectilinearGrid2)
    module_dims = size(module_grid)
    module_children = zeros(UM_I, module_dims[1], module_dims[2])
    rt_module = RectilinearPartition2(UM_I(1), "Module_00001",
                                      module_grid, module_children)

    lattice_grid = RegularGrid2(bounding_box(rt_module))
    lattice_children = ones(UM_I, 1, 1)
    lattice = RegularPartition2(UM_I(1), "Lattice_00001", 
                                lattice_grid, lattice_children)

    core_grid = RectilinearGrid(bounding_box(lattice))
    core_children = ones(UM_I, 1, 1)
    core = RectilinearPartition2(UM_I(1), "Core", 
                                 core_grid, core_children)

    return MPACTSpatialPartition(core, [lattice], [rt_module])
end

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
    lattices = map(lat->RegPart2(lat.id, lat.name, RegularGrid(lat.grid), lat.children), 
                   lattice_rect_partitions)

    # Group the lattices into the core
    lattice_bbs = map(x->bounding_box(x), lattices)
    core_rect_partition = RectPart2(lattice_bbs)
    # Correct id and name 
    core_node = root(hm.partition_tree)
    core_data = core_node.data
    core = RectPart2(core_data[1], core_data[2], 
                     core_rect_partition.grid, core_rect_partition.children)

    return MPACTSpatialPartition(core, lattices, modules)
end

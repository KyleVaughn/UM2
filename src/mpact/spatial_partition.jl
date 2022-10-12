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

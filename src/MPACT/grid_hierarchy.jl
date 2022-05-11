export MPACTGridHierarchy

struct MPACTGridHierarchy
    lattice_grid::RectilinearGrid
    module_grid::RectilinearGrid
    coarse_grid::RectilinearGrid
    function MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)
        if !(lattice_grid ⊆ module_grid ⊆ coarse_grid)
            error("Invalid MPACTGridHierarchy. The following must be true:"*
                  " lattice_grid ⊆ module_grid ⊆ coarse_grid")
        end
        
        Δx = module_grid.x[2] - module_grid.x[1]
        nx = length(module_grid.x)
        if !(all(i->module_grid.x[i+1] - module_grid.x[i] ≈ Δx, 1:nx-1))
            error("All ray tracing module must be equal in size")
        end
        Δy = module_grid.y[2] - module_grid.y[1]
        ny = length(module_grid.y)
        if !(all(i->module_grid.y[i+1] - module_grid.y[i] ≈ Δy, 1:ny-1))
            error("All ray tracing module must be equal in size")
        end
        return new(lattice_grid, module_grid, coarse_grid)
    end
end

function MPACTGridHierarchy(coarse_grid::RectilinearGrid)
    module_grid = RectilinearGrid(Vec(xmin(coarse_grid), xmax(coarse_grid)),
                                  Vec(ymin(coarse_grid), ymax(coarse_grid)))
    lattice_grid = module_grid
    return MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)
end

function MPACTGridHierarchy(module_grid::RectilinearGrid,
                            coarse_grid::RectilinearGrid)
    lattice_grid = RectilinearGrid(Vec(xmin(module_grid), xmax(module_grid)),
                                   Vec(ymin(module_grid), ymax(module_grid)))
    return MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)
end

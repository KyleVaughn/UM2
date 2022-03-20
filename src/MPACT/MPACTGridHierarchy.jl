struct MPACTGridHierarchy
    lattice_grid::RectilinearGrid2D
    module_grid::RectilinearGrid2D
    coarse_grid::RectilinearGrid2D
    function MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)
        if !(lattice_grid ⊆ module_grid ⊆ coarse_grid)
            error("Invalid MPACTGridHierarchy. The following must be true:"*
                  " lattice_grid ⊆ module_grid ⊆ coarse_grid")
        end
        if !(lattice_grid.bb == module_grid.bb == coarse_grid.bb)
            error("Grids do not partition the same bounding box!")
        end
        X = length(module_grid.xdiv)
        Δx_previous = module_grid.xdiv[1] - module_grid.bb.xmin
        for i ∈ 1:X-1
            Δx = module_grid.xdiv[i+1] - module_grid.xdiv[i] 
            if !(Δx_previous ≈ Δx)
                error("All ray tracing module must be equal in size")
            end
            Δx_previous = Δx
        end
        Δx = module_grid.bb.xmax - module_grid.xdiv[X] 
        if !(Δx_previous ≈ Δx)
            error("All ray tracing module must be equal in size")
        end
        Y = length(module_grid.ydiv)
        Δy_previous = module_grid.ydiv[1] - module_grid.bb.ymin
        for i ∈ 1:Y-1
            Δy = module_grid.ydiv[i+1] - module_grid.ydiv[i] 
            if !(Δy_previous ≈ Δy)
                error("All ray tracing module must be equal in size")
            end
            Δy_previous = Δy
        end
        Δy = module_grid.bb.ymax - module_grid.ydiv[Y] 
        if !(Δy_previous ≈ Δy)
            error("All ray tracing module must be equal in size")
        end
        return new(lattice_grid, module_grid, coarse_grid)
    end
end

#function MPACTGridHierarchy(grid::RectilinearGrid2D{T}) where {T}
#
#end

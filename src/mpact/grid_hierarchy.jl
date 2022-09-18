export MPACTGridHierarchy

# Lattice grid = Matrix{Lattice}
# Lattice = Module grid = Matrix{Module}
# Module =  A rectilinear grid of coarse cells
mutable struct MPACTGridHierarchy
    lattice_grid::Matrix{Matrix{RectilinearGrid2d}}

    function MPACTGridHierarchy(lattice_grid::Matrix{Matrix{RectilinearGrid2d}})
        # Check that all modules have the same size
        Δx = delta_x(lattice_grid[1][1])
        Δy = delta_y(lattice_grid[1][1])
        for ilat in eachindex(lattice_grid)
            for imod in eachindex(lattice_grid[ilat])
                if Δx != delta_x(lattice_grid[ilat][imod])
                    error("All modules must have the same Δx")
                end
                if Δy != delta_y(lattice_grid[ilat][imod])
                    error("All modules must have the same Δy")
                end
            end
        end
        # Check that the modules in each lattice are aligned
        for ilat in eachindex(lattice_grid)
            imax, jmax = size(lattice_grid[ilat])
            for jmod in 1:jmax
                for imod in 1:imax - 1
                    ymin1 = y_min(lattice_grid[ilat][imod, jmod])
                    ymin2 = y_min(lattice_grid[ilat][imod + 1, jmod])
                    ymax1 = y_max(lattice_grid[ilat][imod, jmod])
                    ymax2 = y_max(lattice_grid[ilat][imod + 1, jmod])
                    if (ymin1 - ymin2) > 1e-8
                        error("Modules in the same lattice row must have the same y_min")
                    end
                    if (ymax1 - ymax2) > 1e-8
                        error("Modules in the same lattice row must have the same y_max")
                    end
                end
            end
            for imod in 1:imax
                for jmod in 1:jmax - 1
                    xmin1 = x_min(lattice_grid[ilat][imod, jmod])
                    xmin2 = x_min(lattice_grid[ilat][imod, jmod + 1])
                    xmax1 = x_max(lattice_grid[ilat][imod, jmod])
                    xmax2 = x_max(lattice_grid[ilat][imod, jmod + 1])
                    if (xmin1 - xmin2) > 1e-8
                        error("Modules in the same lattice column must have the same x_min")
                    end
                    if (xmax1 - xmax2) > 1e-8
                        error("Modules in the same lattice column must have the same x_max")
                    end
                end
            end
        end
        # Check that the lattices are aligned
        imax, jmax = size(lattice_grid)
        for jlat in 1:jmax
            for ilat in 1:imax - 1
                ymin1 = mapreduce(y_min, min, lattice_grid[ilat, jlat])
                ymin2 = mapreduce(y_min, min, lattice_grid[ilat + 1, jlat])
                ymax1 = mapreduce(y_max, max, lattice_grid[ilat, jlat])
                ymax2 = mapreduce(y_max, max, lattice_grid[ilat + 1, jlat])
                if (ymin1 - ymin2) > 1e-8
                    error("Lattices in the same column must have the same y_min")
                end
                if (ymax1 - ymax2) > 1e-8
                    error("Lattices in the same column must have the same y_max")
                end
            end
        end
        for ilat in 1:imax
            for jlat in 1:jmax - 1
                xmin1 = mapreduce(x_min, min, lattice_grid[ilat, jlat])
                xmin2 = mapreduce(x_min, min, lattice_grid[ilat, jlat + 1])
                xmax1 = mapreduce(x_max, max, lattice_grid[ilat, jlat])
                xmax2 = mapreduce(x_max, max, lattice_grid[ilat, jlat + 1])
                if (xmin1 - xmin2) > 1e-8
                    error("Lattices in the same row must have the same x_min")
                end
                if (xmax1 - xmax2) > 1e-8
                    error("Lattices in the same row must have the same x_max")
                end
            end
        end
        return new(lattice_grid)
    end
end

function MPACTGridHierarchy(coarse_grid::RectilinearGrid2d)
    module_grid = Matrix{RectilinearGrid2d}(undef, 1, 1)
    module_grid[1, 1] = coarse_grid
    lattice_grid = Matrix{Matrix{RectilinearGrid2d}}(undef, 1, 1)
    lattice_grid[1, 1] = module_grid
    return MPACTGridHierarchy(lattice_grid)
end

function MPACTGridHierarchy(module_grid::Matrix{RectilinearGrid2d})
    lattice_grid = Matrix{Matrix{RectilinearGrid2d}}(undef, 1, 1)
    lattice_grid[1, 1] = module_grid
    return MPACTGridHierarchy(lattice_grid)
end

struct MPACTCore2D{T}
    bb::AABox2D{T}
    lattice_ids::Matrix{UInt32} # global lattice ids
end

function validate_core(core::MPACTCore2D)
    # core = ⋃ lattices
    if core.bb ≉ union(getfield.(MPACTLattices[core.lattice_ids], :bb)) 
        error("Lattices do not partition the core")
    end
    for ilat ∈ core.lattice_ids
        lat = MPACTLattices[ilat]
        # latticeᵢ = ⋃ ray tracing modules
        if lat.bb ≉ union(getfield.(MPACTRayTracingModules[lat.module_ids], :bb)) 
            error("Modules do not partition lattice $ilat")
        end
        for imod ∈ lat.module_ids
            rt_mod = MPACTRayTracingModules[imod] 
            # moduleᵢ = ⋃ coarse cells
        end
    end
end

# Renumber lattices. Reorder in memory, along with everything else, to match Z-order or hilbert

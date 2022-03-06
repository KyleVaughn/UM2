struct MPACTCore2D{T}
    bb::AABox2D{T}
    lattice_ids::Matrix{UInt32} # global lattice ids
end

#function validate_core_partition(core::MPACTCore2D)
#    # core = ⋃ lattices
#    if core.bb ≉ mapreduce(x->x.bb, union, MPACTLattices[core.lattice_ids]) 
#        error("Lattices do not partition the core")
#    end
#    for ilat ∈ core.lattice_ids
#        lat = MPACTLattices[ilat]
#        # latticeᵢ = ⋃ ray tracing modules
#        if lat.bb ≉ mapreduce(x->x.bb, union, MPACTRayTracingModules[lat.module_ids]) 
#            error("Modules do not partition lattice $ilat")
#        end
#        for imod ∈ lat.module_ids
#            rt_mod = MPACTRayTracingModules[imod] 
#            # moduleᵢ = ⋃ coarse cells
#            if rt_mod.bb ≉ mapreduce(x->x.bb, union, MPACTCoarseCells[rt_mod.cell_ids])
#                error("Coarse cells do not partition ray tracing module $imod")
#            end
#            for icell ∈ rt_mod.cell_ids
#                # Check finemesh
#                cell = MPACTCoarseCells[icell]
#                if cell.bb ≉ boundingbox(MPACTPinMeshes
#            
#            end
#        end
#    end
#end
#
## Renumber lattices. Reorder in memory, along with everything else, to match Z-order or hilbert

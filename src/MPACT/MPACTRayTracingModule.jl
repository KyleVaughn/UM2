struct MPACTRayTracingModule{T}
    bb::AABox2D{T} # local
    cell_ids::Matrix{UInt32} # global coarse cell ids
    id::UInt32 # global id
end

const MPACTRayTracingModules = MPACTRayTracingModule[]

function MPACTRayTracingModule(bb::AABox2D, 
                               cell_ids::Matrix{UInt32}=Matrix{UInt32}(undef,0,0))
    nmods = length(MPACTRayTracingModules)
    mod = MPACTRayTracingModule(bb, cell_ids, UInt32(nmods + 1)) 
    push!(MPACTRayTracingModules, mod)
    return MPACTRayTracingModules[nmods + 1]
end

export VolumeMesh

struct VolumeMesh{Dim,T,U} <: AbstractMesh
    points::Vector{Point{Dim,T}}
    offsets::Vector{U}
    connectivity::Vector{U}
    name::String
    groups::Dict{String,BitSet}
end

function volume_mesh2_alias_string(offset_diff::I) where {I<:Integer}
    if offset_diff == 3 
        return "Triangle"
    elseif offset_diff == 4 
        return "Quadrilateral"
    elseif offset_diff == 6 
        return "QuadraticTriangle"
    elseif offset_diff == 8 
        return "QuadraticQuadrilateral"
    else
        error("Unsupported type.")
        return nothing
    end
end

function Base.show(io::IO, mesh::VolumeMesh{2,T,U}) where {T,U}
    println(io, "VolumeMesh{2, ",T,", ",U,"}")
    println(io, "  ├─ Name      : ", mesh.name)
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        println(io, "  ├─ Size (KB) : ", size_MB*1000)
    else
        println(io, "  ├─ Size (MB) : ", size_MB)
    end
    println(io, "  ├─ Points    : ", length(mesh.points))
    println(io, "  ├─ Cells     : ", length(mesh.offsets) - 1)
    offset_diffs = map(i->mesh.offsets[i+1]-mesh.offsets[i], 1:length(mesh.offsets)-1)
    unique_types = unique(offset_diffs) 
    nunique_types = length(unique_types)
    for i = 1:nunique_types
        type = unique_types[i]
        ncells = count(x->x === type,  offset_diffs)
        if i === nunique_types
            println(io, "  │  └─ ", rpad(volume_mesh2_alias_string(type), 22), ": ", ncells)
        else
            println(io, "  │  ├─ ", rpad(volume_mesh2_alias_string(type), 22), ": ", ncells)
        end
    end
    ngroups = length(mesh.groups)
    println(io, "  └─ Groups    : ", ngroups)
    if 0 < ngroups ≤ 5
        group_keys = sort!(collect(keys(mesh.groups)))
        for i = 1:ngroups
            if i === ngroups
                println(io, "     └─ ", group_keys[i])
            else
                println(io, "     ├─ ", group_keys[i])
            end
        end
    end
end

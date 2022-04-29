export PolytopeVertexMesh

struct PolytopeVertexMesh{Dim,T,P<:Polytope}
    name::String
    vertices::Vector{Point{Dim,T}}
    polytopes::Vector{P}
    groups::Dict{String,BitSet}
end

# constructors
function PolytopeVertexMesh(name::String, 
                            vertices::Vector{Point{Dim,T}}, 
                            polytopes::Vector{P}
                           ) where {Dim,T,P}
    return PolytopeVertexMesh(name, vertices, polytopes, Dict{String,BitSet}())
end

function PolytopeVertexMesh(vertices::Vector{Point{Dim,T}}, 
                            polytopes::Vector{P}
                           ) where {Dim,T,P}
    return PolytopeVertexMesh("", vertices, polytopes, Dict{String,BitSet}())
end

function Base.show(io::IO, mesh::PolytopeVertexMesh)
    println(io, typeof(mesh))
    println(io, "  ├─ Name      : ", mesh.name)
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        println(io, "  ├─ Size (KB) : ", size_MB*1000)
    else
        println(io, "  ├─ Size (MB) : ", size_MB)
    end
    println(io, "  ├─ Vertices  : ", length(mesh.vertices))
    println(io, "  ├─ Polytopes : ", length(mesh.polytopes))
    poly_types = unique!(map(x->typeof(x), mesh.polytopes))
    npoly_types = length(poly_types)
    for i = 1:npoly_types
        poly_type = poly_types[i]
        npoly = count(x->x isa poly_type,  mesh.polytopes)
        if i === npoly_types
            println(io, "  │  └─ ", rpad(alias_string(poly_type), 22), ": ", npoly)
        else                
            println(io, "  │  ├─ ", rpad(alias_string(poly_type), 22), ": ", npoly) 
        end                 
    end
    ngroups = length(mesh.groups)
    println(io, "  └─ Groups    : ", ngroups) 
    if 0 < ngroups ≤ 5
        group_keys = sort(collect(keys(mesh.groups)))
        for i = 1:ngroups
            if i === ngroups
                println(io, "     └─ ", group_keys[i])
            else
                println(io, "     ├─ ", group_keys[i]) 
            end
        end
    end
end

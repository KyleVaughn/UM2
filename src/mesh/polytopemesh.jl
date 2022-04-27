export PolytopeVertexMesh

struct PolytopeVertexMesh{Dim,T,P}
    name::String
    vertices::Vector{Point{Dim,T}}
    polytopes::Vector{P}
    polytope_sets::Dict{String,BitSet}
end

function PolytopeVertexMesh{Dim,T,P}(;
    name::String = "",
    vertices::Vector{Point{Dim,T}} = Point{Dim,T}[],
    polytopes::Vector{P} = P[],
    polytope_sets::Dict{String,BitSet} = Dict{String,BitSet}()
    ) where {Dim,T,P}
    return PolytopeVertexMesh(name, vertices, polytopes, polytope_sets)
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
#    println(io, "  ├─ Faces     : $(length(mesh.faces))")
#    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{3},  mesh.faces))")
#    println(io, "  │  └─ Quadrilateral  : $(count(x->x isa SVector{4},  mesh.faces))")
#    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end

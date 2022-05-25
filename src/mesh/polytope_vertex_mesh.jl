export PolytopeVertexMesh
export name, vertices, polytopes, groups, vtk_type

struct PolytopeVertexMesh{Dim,T,P<:Polytope} <: AbstractMesh
    vertices::Vector{Point{Dim,T}}
    polytopes::Vector{P}
    name::String
    groups::Dict{String,BitSet}
end

name(mesh::PolytopeVertexMesh) = mesh.name
points(mesh::PolytopeVertexMesh) = mesh.vertices
vertices(mesh::PolytopeVertexMesh) = mesh.vertices
polytopes(mesh::PolytopeVertexMesh) = mesh.polytopes
groups(mesh::PolytopeVertexMesh) = mesh.groups

# constructors
function PolytopeVertexMesh(vertices::Vector{Point{Dim,T}}, 
                            polytopes::Vector{P},
                            name::String
                           ) where {Dim,T,P}
    return PolytopeVertexMesh(vertices, polytopes, name, Dict{String,BitSet}())
end

function PolytopeVertexMesh(vertices::Vector{Point{Dim,T}}, 
                            polytopes::Vector{P}
                           ) where {Dim,T,P}
    return PolytopeVertexMesh(vertices, polytopes, "", Dict{String,BitSet}())
end

function PolytopeVertexMesh(mesh::VolumeMesh{Dim,T,U}) where {Dim,T,U}
    return PolytopeVertexMesh(
             mesh.points, 
             map(i->_materialize_face_connectivity(i, mesh), eachindex(mesh.types)),
             mesh.name, 
             mesh.groups
            )
end

vtk_type(::Type{<:Triangle})               = VTK_TRIANGLE
vtk_type(::Type{<:Quadrilateral})          = VTK_QUAD
vtk_type(::Type{<:QuadraticTriangle})      = VTK_QUADRATIC_TRIANGLE
vtk_type(::Type{<:QuadraticQuadrilateral}) = VTK_QUADRATIC_QUAD
vtk_type(::Type{<:Tetrahedron})            = VTK_TETRA
vtk_type(::Type{<:Hexahedron})             = VTK_HEXAHEDRON
vtk_type(::Type{<:QuadraticTetrahedron})   = VTK_QUADRATIC_TETRA
vtk_type(::Type{<:QuadraticHexahedron})    = VTK_QUADRATIC_HEXAHEDRON

function Base.show(io::IO, mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P}
    print(io, "PolytopeVertexMesh{",Dim,", ",T,", ")
    if isconcretetype(P)
        println(io, alias_string(P),"{",vertex_type(P),"}}")
    else
        println(io, P,"}")
    end
    println(io, "  ├─ Name      : ", mesh.name)
    size_B = Base.summarysize(mesh)
    if size_B < 1e6
        println(io, "  ├─ Size (KB) : ", string(@sprintf("%.3f", size_B/1000)))
    else
        println(io, "  ├─ Size (MB) : ", string(@sprintf("%.3f",size_B/1e6)))
    end
    println(io, "  ├─ Vertices  : ", length(mesh.vertices))
    println(io, "  ├─ Polytopes : ", length(mesh.polytopes))
    poly_types = unique!(map(typeof, mesh.polytopes))
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

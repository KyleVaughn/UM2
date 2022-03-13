abstract type UnstructuredMesh{Dim, Ord, T, U} end
const UnstructuredMesh2D = UnstructuredMesh{2}
const UnstructuredMesh3D = UnstructuredMesh{3}
const LinearUnstructuredMesh = UnstructuredMesh{Dim, 1} where {Dim}
const LinearUnstructuredMesh2D = UnstructuredMesh{2, 1}
const LinearUnstructuredMesh3D = UnstructuredMesh{3, 1}
const QuadraticUnstructuredMesh = UnstructuredMesh{Dim, 2} where {Dim}
const QuadraticUnstructuredMesh2D = UnstructuredMesh{2, 2}
const QuadraticUnstructuredMesh3D = UnstructuredMesh{3, 2}
Base.broadcastable(mesh::UnstructuredMesh) = Ref(mesh)

function Base.show(io::IO, mesh::UnstructuredMesh2D)
    mesh_type = typeof(mesh)
    println(io, mesh_type)
    println(io, "  ├─ Name      : $(mesh.name)")
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        size_KB = size_MB*1000
        println(io, "  ├─ Size (KB) : $size_KB")
    else
        println(io, "  ├─ Size (MB) : $size_MB")
    end
    println(io, "  ├─ Points    : $(length(mesh.points))")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end

function _select_mesh_UInt_type(N::Int64)
    if N ≤ typemax(UInt16) 
        U = UInt16
    elseif N ≤ typemax(UInt32) 
        U = UInt32
    elseif N ≤ typemax(UInt64) 
        U = UInt64
    else 
        error("That's a big mesh! Number of edges exceeds typemax(UInt64)")
        U = UInt64
    end
    return U
end

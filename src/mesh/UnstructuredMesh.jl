abstract type UnstructuredMesh{Dim,Ord,T,U} end
const UnstructuredMesh2D = UnstructuredMesh{2}
const LinearUnstructuredMesh = UnstructuredMesh{Dim,1} where {Dim}
const LinearUnstructuredMesh2D = UnstructuredMesh{2,1}
const QuadraticUnstructuredMesh = UnstructuredMesh{Dim,2} where {Dim}
const QuadraticUnstructuredMesh2D = UnstructuredMesh{2,2}
Base.broadcastable(mesh::UnstructuredMesh) = Ref(mesh)

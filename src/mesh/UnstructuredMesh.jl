abstract type UnstructuredMesh{N,T,U} end
const UnstructuredMesh_2D = UnstructuredMesh{2}
const UnstructuredMesh_3D = UnstructuredMesh{3}
Base.broadcastable(mesh::UnstructuredMesh) = Ref(mesh)

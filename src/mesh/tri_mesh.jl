export TriMesh

# TRIMESH    
# -----------------------------------------------------------------------------    
#    
# A 2D triangle mesh.
#

struct TriMesh{T, I}

    # Name of the mesh.
    name::String

    # Face-vertex connectivity.
    fv_conn::Vector{I}

    # Vertex-face connectivity offsets.
    vf_offsets::Vector{I}

    # Vertex-face connectivity.
    vf_conn::Vector{I}

    # Vertex positions.
    vertices::Vector{Point{2, T}}

    # Material ID for each face.
    material_ids::Vector{Int8}

end

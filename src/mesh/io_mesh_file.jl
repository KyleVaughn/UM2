export MeshFile

export ABAQUS_FORMAT, XDMF_FORMAT

# MESH FILE
# -----------------------------------------------------------------------------    
#    
# An intermediate representation of a mesh that can be used to:
#   - read from a file
#   - write to a file
#   - convert to a mesh
# 

const ABAQUS_FORMAT = 0 
const XDMF_FORMAT = 1 

mutable struct MeshFile
    filepath::String
    format::Int64
    name::String
    nodes::Vector{Point2{UM_F}}
    element_types::Vector{Int8}
    element_offsets::Vector{UM_I}
    elements::Vector{UM_I}
    elsets::Dict{String, Set{UM_I}}
end

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

mutable struct MeshFile{T <: AbstractFloat, I <: Integer}
    filepath::String
    format::Int64
    name::String
    nodes::Vector{Point2{T}}
    element_types::Vector{Int8}
    element_offsets::Vector{I}
    elements::Vector{I}
    elsets::Dict{String, Set{I}}
end

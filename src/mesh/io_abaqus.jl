# IO routines for the Abaqus .inp file format
const ABAQUS_CPS3  = VTK_TRIANGLE
const ABAQUS_CPS4  = VTK_QUAD
const ABAQUS_CPS6  = VTK_QUADRATIC_TRIANGLE
const ABAQUS_CPS8  = VTK_QUADRATIC_QUAD
const ABAQUS_C3D4  = VTK_TETRA
const ABAQUS_C3D8  = VTK_HEXAHEDRON
const ABAQUS_C3D10 = VTK_QUADRATIC_TETRA
const ABAQUS_C3D20 = VTK_QUADRATIC_HEXAHEDRON

function _abaqus_element_string_to_int(x::String)
    if x === "CPS3"
        return ABAQUS_CPS3
    elseif x === "CPS4"
        return ABAQUS_CPS4
    elseif x === "CPS6"
        return ABAQUS_CPS6
    elseif x === "CPS8"
        return ABAQUS_CPS8
    elseif x === "C3D4"
        return ABAQUS_C3D4
    elseif x === "C3D8"
        return ABAQUS_C3D8
    elseif x === "C3D10"
        return ABAQUS_C3D10
    elseif x === "C3D20"
        return ABAQUS_C3D20
    else
        error("Invalid Abaqus element type.")
        return nothing
    end
end

function read_abaqus(path::String, ::Type{T}) where {T<:AbstractFloat}
    file = open(path, "r")
    try
        nodes = Point{3,T}[]
        element_types = UInt64[] 
        offsets = UInt64[]
        elements = UInt64[] 
        name = ""
        elsets = Dict{String, BitSet}()
        for line in eachline(file) 
            if length(line) > 0
                if startswith(line, "**") # Comment
                    continue
                elseif "*Heading" === line
                    # Of the form " name.inp"
                    name = readline(file)[2:end-4]
                elseif "*NODE" === line
                    _read_abaqus_nodes!(file, nodes)
                elseif startswith(line, "*ELEMENT")
                    m = match(r"type=(.*?), ELSET", line)
                    if isnothing(m)
                        error("Incorrectly formatted Abaqus file?")
                    else
                        element_type = UInt64(_abaqus_element_string_to_int(String(m.captures[1])))
                    end
                    _read_abaqus_elements!(file, element_type, element_types, offsets, elements)
                elseif startswith(line, "*ELSET")
                    m = match(r"(?<=ELSET=).*", line)
                    if isnothing(m)
                        error("Incorrectly formatted Abaqus file?")
                    else
                        set_name = m.match 
                    end
                    elsets[set_name] = _read_abaqus_elset(file)
                end
            end
        end
        abaqus_2d = (ABAQUS_CPS3, ABAQUS_CPS4, ABAQUS_CPS6, ABAQUS_CPS8)
        abaqus_3d = (ABAQUS_C3D4, ABAQUS_C3D8, ABAQUS_C3D10, ABAQUS_C3D20)
        is2d = any(x->x ∈ abaqus_2d, element_types)
        is3d = any(x->x ∈ abaqus_3d, element_types)
        if is2d && is3d
            error("File contains both surface (CPS) and volume (C3D) elements."*
                  "Limit element types to be CPS or C3D, so mesh dimension may be determined.") 
        end
        U = _select_uint_type(max(length(elements), 
                                  length(nodes)))
        push!(offsets, length(elements)+1)
        if is2d
            return VolumeMesh{2,T,U}(nodes, element_types, offsets, elements, name, elsets)
        else
            return VolumeMesh{3,T,U}(nodes, element_types, offsets, elements, name, elsets)
        end
    finally
        close(file)
    end
    return nothing
end

function _read_abaqus_nodes!(file::IOStream, nodes::Vector{Point{3,T}}) where {T<:AbstractFloat}
    # Count the number of nodes
    mark(file)
    nnodes = 0
    line = readline(file) 
    while '*' !== line[1]
        nnodes += 1
        line = readline(file)
    end
    reset(file)
    # Allocate and populate a vector of nodes
    new_nodes = Vector{Point{3,T}}(undef, nnodes)
    for i in 1:nnodes
        line = readline(file)
        xyz = view(split(line, ','), 2:4)
        new_nodes[i] = Point{3,T}(ntuple(i->parse(Float64, xyz[i]), Val(3)))
    end
    append!(nodes, new_nodes)
    return nothing
end

function _read_abaqus_elements!(file::IOStream, 
                                element_type::UInt64,
                                element_types::Vector{UInt64},
                                offsets::Vector{UInt64},
                                elements::Vector{UInt64}) 
    mark(file)
    line = readline(file)
    npts = points_in_vtk_type(element_type)
    nelements = 0
    while !('*' === line[1] || eof(file))
        nelements += 1
        line = readline(file)
    end
    reset(file)
    append!(element_types, fill(element_type, nelements))
    new_offsets = Vector{UInt64}(undef, nelements)
    new_elements = Vector{UInt64}(undef, npts*nelements)
    ectr = 1
    len_elements = length(elements)
    for i in 1:nelements 
        line = readline(file)
        vids = view(split(line, ','), 2:npts+1)
        @. new_elements[ectr:ectr + npts - 1] = parse(UInt64, vids)
        new_offsets[i] = ectr + len_elements
        ectr += npts 
    end
    append!(offsets, new_offsets)
    append!(elements, new_elements)
    return elements
end

function _read_abaqus_elset(file::IOStream)
    splitline = strip.(split(readuntil(file, "*")), [','])
    seek(file, position(file)-1)
    return BitSet(parse.(Int64, splitline))
end

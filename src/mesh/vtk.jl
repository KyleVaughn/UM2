# Routines for reading and writing VTK files

function read_vtk_2d(filepath::String)
    @info "Reading $filepath"
    file = open(filepath, "r")
    name = "DefaultMeshName"
    cell_types = Vector{Int64}[]
    cells = Vector{Int64}[]
    points = nothing
    while !eof(file)
        line_split = split(readline(file))
        if length(line_split) > 0
            if line_split[1] == "#"
                if line_split[2] == "vtk"
                    name = readline(file)
                end
            elseif line_split[1] == "DATASET"
                if line_split[2] != "UNSTRUCTURED_GRID"
                    @error "DATASET type is $(line_split[2]). Only UNSTRUCTURED_GRID is supported."
                end
            elseif line_split[1] == "POINTS"
                points = _read_vtk_points_2d(file, line_split[2], line_split[3])
            elseif line_split[1] == "CELLS"
                cells = _read_vtk_cells(file, line_split[2])
            elseif line_split[1] == "CELL_TYPES"
                cell_types = _read_vtk_cell_types(file, line_split[2])
            end
        end
    end
    close(file)

    # Remove all cells that are not 2d
    # UnstructuredMesh (UM) uses the same cell types as VTK.
    delete_indices = findall(x->x ∉  UnstructuredMesh_2D_cell_types, cell_types)
    deleteat!(cell_types, delete_indices)
    deleteat!(cells, delete_indices)

    faces = Vector{Vector{Int64}}(undef, length(cells))
    for i in eachindex(cell_types)
        faces[i] = vcat(cell_types[i], cells[i])
    end
    # We can save a lot on memory and boost performance by choosing the smallest usable
    # integer for our mesh. The relationship between faces, edges, and vertices is
    # Triangle
    #   F ≈ 2V
    #   E ≈ 3F/2
    # Quadrilateral
    #   F ≈ V
    #   E ≈ 2F
    # Triangle6
    #   2F ≈ V
    #   E ≈ 3F/2
    # Quadrilateral8
    #   4F ≈ V 
    #   E ≈ 2F
    # We see that V ≤ F ≤ 2E, so we use 2.2F as the max number of faces, edges, or vertices
    # plus a fudge factor for small meshes, where the relationships become less accurate.
    # If 2.2*length(faces) < typemax(UInt), convert to UInt
    float_type = typeof(points[1].x[1])
    if ceil(2.2*length(faces)) < typemax(UInt16)
        I = UInt16
        faces_16 = convert(Vector{Vector{UInt16}}, faces)
        return UnstructuredMesh_2D{float_type, I}(name = name,
                                                  points = points,
                                                  faces = [ Tuple(f) for f in faces_16]
                                                 )
    elseif ceil(2.2*length(faces)) < typemax(UInt32)
        I = UInt32
        faces_32 = convert(Vector{Vector{UInt32}}, faces)
        return UnstructuredMesh_2D{float_type, I}(name = name,
                                                  points = points,
                                                  faces = [ Tuple(f) for f in faces_32]
                                                 )
    else
        I = UInt64
        return UnstructuredMesh_2D{float_type, I}(name = name,
                                                  points = points,
                                                  faces = [ Tuple(f) for f in faces]
                                                 )
    end
end

function _read_vtk_points_2d(
        file::IOStream, 
        npoints_string::SubString{String}, 
        datatype_string::SubString{String}
    )
    npoints = parse(Int64, npoints_string)
    if datatype_string == "float"
        datatype = Float32
    elseif datatype_string == "double"
        datatype = Float64
    else
        @error "Unable to identify POINTS data type."
    end
    points = Vector{Point_2D{datatype}}(undef, npoints)
    for i in 1:npoints 
        xyz = parse.(datatype, split(readline(file)))
        points[i] = Point_2D(xyz[1], xyz[2])
    end
    return points
end

function _read_vtk_cells(
        file::IOStream, 
        ncells_string::SubString{String}, 
    )
    ncells = parse(Int64, ncells_string)
    cells = Vector{Vector{Int64}}(undef, ncells)
    for i in 1:ncells
        # Strip the number of points and account for base 1 indexing
        pointIDs = parse.(Int64, split(readline(file))) .+ 1
        cells[i] = pointIDs[2:length(pointIDs)]
    end
    return cells
end

function _read_vtk_cell_types(
        file::IOStream, 
        ncells_string::SubString{String}, 
    )
    ncells = parse(Int64, ncells_string)
    cell_types = Vector{Int64}(undef, ncells)
    for i in 1:ncells
        cellID = parse(Int64, readline(file))
        cell_types[i] = cellID
    end
    return cell_types
end

function write_vtk_2d(filename::String, mesh::UnstructuredMesh_2D)
    @info "Writing $filename"
    @warn "  - Face sets not currently supported for vtk files"
    # Check valid filename
    if !occursin(".vtk", filename)
        @error "Invalid filename. '.vtk' does not occur in $filename"
    end
    file = open(filename, "w")
    println(file, "# vtk DataFile Version 2.0")
    println(file, mesh.name)
    println(file, "ASCII")
    println(file, "DATASET UNSTRUCTURED_GRID")

    # Points
    pointtype = typeof(mesh.points[1].x[1])
    if pointtype === Float64
        type_points = "double"
    elseif pointtype === Float32
        type_points = "float"
    else
        @error "Unrecognized point type."
    end
    npoints = length(mesh.points)
    println(file, "POINTS $npoints $type_points")
    for i in 1:npoints
        x, y = mesh.points[i].x
        println(file, "$x $y 0.0")
    end
    println(file, "")

    # Cells
    ncells = 0
    ncell_parts = 0
    for cell in mesh.faces
        ncells += 1
        ncell_parts += length(cell)
    end
    println(file, "CELLS $ncells $ncell_parts")
    for cell in mesh.faces
        nverts = length(cell) - 1
        write(file, "$nverts ")
        for i in 1:nverts
            vert = cell[i + 1] - 1 # 0 based index
            if i < nverts
                write(file, "$vert ")
            else
                println(file, "$vert")
            end
        end
    end
    println(file, "")

    # Cell types
    # UnstructuredMesh uses the same cell types as VTK
    println(file, "CELL_TYPES $ncells")
    for cell in mesh.faces
        cell_type = cell[1]
        println(file, "$cell_type")
    end

    close(file)
end

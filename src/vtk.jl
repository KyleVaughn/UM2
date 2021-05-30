function read_vtk(filepath::String)
    cell_type_whitelist = [5,  # Triangle 
                         9,  # Quadrilateral
                         22, # Triangle6 
                         23] # Quad8
    cell_type_2D = [5]
    cell_type_3D = [10]

    name = "DefaultMeshName"
    file = open(filepath, "r")
    while !eof(file)
        line_split = split(readline(file))
        if length(line_split) > 0
            if line_split[1] == "#"
                if line_split[2] == "vtk"
                    name = readline(file)
                end
            elseif line_split[1] == "DATASET"
                if line_split[2] != "UNSTRUCTURED_GRID"
                    error("DATASET type is $line_split[2]. Only UNSTRUCTURED_GRID is supported.")
                end
            elseif line_split[1] == "POINTS"
                global points = read_vtk_points(file, line_split[2], line_split[3])
            elseif line_split[1] == "CELLS"
                global cells = read_vtk_cells(file, line_split[2])
            elseif line_split[1] == "CELL_TYPES"
                global cell_types = read_vtk_cell_types(file, line_split[2])
            end
        end
    end
    close(file)

    # Remove 0D, 1D cells
    delete_indices = findall(x->x ∉  cell_type_whitelist, cell_types)
    deleteat!(cell_types, delete_indices)
    deleteat!(cells, delete_indices)

    # Find the dimension based upon cell types
    if all(x->x ∈  cell_type_2D, cell_types)
        dim = 2
    elseif all(x->x ∈  cell_type_3D, cell_types)
        dim = 3
    else
        error("VTK file contains mixed dimension elements. The developer needs to address this.")
        dim = 0
    end

    cells_combined = Vector{Int64}[]
    for i in 1:length(cell_types)
        push!(cells_combined, vcat(cell_types[i], cells[i]))
    end

    # Construct edges
    edges = construct_edges(cells_combined)

#    if dim == 2
        return UnstructuredMesh(
                                points = points,
                                edges = edges,
                                faces = cells_combined,
                                dim = dim,
                                name = name
                                )
#    else
#        return UnstructuredMesh(
#                                points = points,
#                                edges = edges,
#                                faces = faces,
#                                cells = cells,
#                                dim = dim
#                                )
#    end
end

function read_vtk_points(
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
        error("Unable to identify POINTS data type.")
    end
    points = Point{datatype}[]
    for i in 1:npoints 
        xyz = parse.(datatype, split(readline(file)))
        p = Point(xyz[1], xyz[2], xyz[3])
        push!(points, p)
    end
    return points
end

function read_vtk_cells(
        file::IOStream, 
        ncells_string::SubString{String}, 
    )
    ncells = parse(Int64, ncells_string)
    cells = Vector{Int64}[]
    for i in 1:ncells
        # Strip the number of points and account for base 1 indexing
        pointIDs = parse.(Int64, split(readline(file))) .+ 1
        push!(cells, pointIDs[2:length(pointIDs)])
    end
    return cells
end

function read_vtk_cell_types(
        file::IOStream, 
        ncells_string::SubString{String}, 
    )
    ncells = parse(Int64, ncells_string)
    cell_types = Int64[]
    for i in 1:ncells
        cellID = parse(Int64, readline(file))
        push!(cell_types, cellID)
    end
    return cell_types
end

function write_vtk(filename::String, mesh::UnstructuredMesh)
    file = open(filename, "w")
    println(file, "# vtk DataFile Version 2.0")
    println(file, mesh.name)
    println(file, "ASCII")
    println(file, "DATASET UNSTRUCTURED_GRID")

    # Points
    pointtype = typeof(mesh.points[1].coord[1])
    if pointtype == Float64
        type_points = "double"
    elseif pointtype == Float32
        type_points = "float"
    else
        error("Unrecognized point type.")
    end
    npoints = length(mesh.points)
    println(file, "POINTS $npoints $type_points")
    for i in 1:npoints
        x, y, z = mesh.points[i].coord
        println(file, "$x $y $z")
    end
    println(file, "")

    # Cells
    if mesh.dim == 2
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
    else
        error("Implement 3d")
    end
    println(file, "")

    # Cell types
    println(file, "CELL_TYPES $ncells")
    if mesh.dim == 2
        for cell in mesh.faces
            cell_type = cell[1]
            println(file, "$cell_type")
        end
    else
        error("Implement 3d")
    end

    close(file)
end

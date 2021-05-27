function read_vtk(filepath::String)

    cell_type_whitelist = [5,  # Triangle 
                         9,  # Quadrilateral
                         22, # Triangle6 
                         23] # Quad8


    file = open(filepath, "r")
    while !eof(file)
        line_split = split(readline(file))
        if length(line_split) > 0
            if line_split[1] == "DATASET"
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

    # Remove 0D, 1D cells
    delete_indices = findall(x->x ∉  cell_type_whitelist, cell_types)
    deleteat!(cell_types, delete_indices)
    deleteat!(cells, delete_indices)



    return points, cells, cell_types
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








filepath = "../../sphere_surface_mod.vtk"
read_vtk(filepath)


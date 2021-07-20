function read_abaqus_2d(filepath::String)
    file = open(filepath, "r")
    while !eof(file)
        line_split = split(readline(file))
        if length(line_split) > 0
            println(line_split)
            if occursin("**", line_split[1]) # Comment
                continue
            elseif occursin("*NODE", line_split[1])
                points = _read_nodes(file)
            end
        end
    end
    close(file)

#    return UnstructuredMesh_2D(points,
#                               faces,
#                              )
end

function _read_nodes(file::IOStream)

end

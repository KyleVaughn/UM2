f = open("./test/mesh_files/cube.stl", "r") 

# Get vertices and name first, then do faces once we can map each point to a unique ID.
name = "DefaultName"
vertices = Point{Float32}[]
for line in readlines(f) 
    line_split = split(line)
    if (line_split[1] == "solid") && length(line_split) > 1
        name = line_split[2] 
    elseif line_split[1] == "vertex"
        x, y, z = parse.(Float32, line_split[2:4])
        push!(vertices, Point(x, y, z))
    end
end
vertices = collect(Set(vertices))

# Get faces
seekstart(f)
faces = Vector{Int64}[]
face = Int64[] 
for line in readlines(f) 
    line_split = split(line)
    if line_split[1] == "endfacet"
        push!(faces, face)
        face = Int64[]
    elseif line_split[1] == "vertex"
        x, y, z = parse.(Float32, line_split[2:4])
        point = Point(x, y, z)
        index = findfirst(v->v==point, vertices)
        push!(face, index)
    end 
end
close(f)

um = UnstructuredMesh(Tuple(vertices), faces = faces)

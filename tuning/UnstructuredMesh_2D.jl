using MOCNeutronTransport
using BenchmarkTools
using Plots
pyplot()

import MOCNeutronTransport.area

struct UM_Tuple{P, F, T}
    points::NTuple{P, Point_2D{T}}
    faces::NTuple{F, Tuple{Vararg{Int64}}}
end
struct UM_Vector{P, T}
    points::NTuple{P, Point_2D{T}}
    faces::Vector{Vector{Int64}}
end

function area(mesh::UM_Vector)
    mesh_area = 0.0
    for face in mesh.faces
        if face[1] == 9
            mesh_area += area(
                        Quadrilateral_2D(
                                map(x -> mesh.points[x], Tuple(face[2:5]))
                            )
                        ) 
        end
    end
    return mesh_area
end

function area(mesh::UM_Tuple)
    mesh_area = 0.0
    for face in mesh.faces
        if face[1] == 9
            mesh_area += area(
                        Quadrilateral_2D(
                                map(x -> mesh.points[x], face[2:5])
                            )
                        ) 
        end
    end
    return mesh_area
end

tuple_times = Float64[]
vector_times = Float64[]
Nset = Int64.([1E1, 1E2, 1E3, 1E4, 1E5, 1E6])
Nctr = 1
for N = Nset
    # Generate the points
    points = Tuple(reduce(vcat, [[Point_2D(1.0*i, 0.0), Point_2D(1.0*i, 1.0)] for i = 0:N]))
    # Generate the faces for quadrilaterals
    quad_faces_vec = [[9, i, i+2, i+3, i+1] for i = 1:2:2N]
    quad_faces_tuple = Tuple([ Tuple(v) for v in quad_faces_vec ])
    
    faces_tuple = quad_faces_tuple
    faces_vec = quad_faces_vec
    
    tuple_mesh = UM_Tuple(points, faces_tuple)
    vector_mesh = UM_Vector(points, faces_vec)
    
    # Test speed of converting to quadrilaterals and getting the area
    println("N = 1E$Nctr")
    time = @belapsed area($vector_mesh)
    push!(vector_times, time/N)
    println("  Vector: $time")
    time = @belapsed area($tuple_mesh)
    push!(tuple_times, time/N)
    println("  Tuple: $time")
    global Nctr += 1
end
plot(Nset,
     vector_times,
     label = "Vector",
     title = "Number of mesh cells vs area time per cell",
     xaxis=:log,
     yaxis=:log
    )
plot!(Nset, tuple_times, label = "Tuple")
xlabel!("Mesh cells")
ylabel!("Area computation time per cell")

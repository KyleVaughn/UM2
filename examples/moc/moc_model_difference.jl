using UM2
using UM2_Vis

# Create a figure
f = Figure(resolution = (1024, 1024))
ax = Axis(f[1,1], aspect = 1)
display(f)

# Import the mesh
mat_names, mats, elsets, mesh = import_mesh("1a_quad2.inp")
color_map = Dict("Clad" => "pgrey",
                 "Fuel" => "pred",
                 "Water" => "pblue")

# Setup modular rays
bb = AABox(Point(0.0, 0.0), Point(1.26, 1.26)) 
γ = π/8
s = 0.4
rays = get_modular_rays(γ, s, bb)

# Determine the line segments and their material
for r in rays
    rvec = intersect_faces_all(r, mesh)
    lines = [LineSegment(r(rvec[i]), r(rvec[i+1])) for i in 1:2:length(rvec)-1]
    mesh_faces = [find_face(l(0.5), mesh) for l in lines]
    mesh_materials = [mats[f] for f in mesh_faces]
    for i in 1:length(lines)
        color = color_map[mat_names[mesh_materials[i]]]
        println("draw [", color, "] (", lines[i][1][1], ", " -- ", lines[i][2])
    end
end


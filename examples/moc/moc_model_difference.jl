# Create a Tikz figure representing the MOC geometry
using UM2

γ = π/8
s = 0.4

# Import the mesh
mat_names, mats, elsets, mesh = import_mesh("1a_quad2.inp")
color_map = Dict("Clad" => "pgrey",
                 "Fuel" => "pred",
                 "Water" => "pblue")

# Setup modular rays
bb = bounding_box(mesh)
rays = get_modular_rays(γ, s, bb)
w = width(bb)    
h = height(bb)    
γ1 = γ
if PI_2 < γ    
    γ1 = π - γ    
end    
nx = ceil(Int64, w * sin(γ1) / s)    
ny = ceil(Int64, h * cos(γ1) / s)    
# Effective angle to ensure cyclic rays    
γₑ = atan((h * nx) / (w * ny))
γ_str = string(round(180 * γₑ / π, digits = 3))
sin_γₑ, cos_γₑ = sincos(γₑ)    
inv_sin_γₑ = 1 / sin_γₑ    
inv_cos_γₑ = 1 / cos_γₑ    
tan_γₑ = sin_γₑ * inv_cos_γₑ    
inv_tan_γₑ = cos_γₑ * inv_sin_γₑ    
s_eff = w * sin_γₑ / nx

for r in rays
    rvec = Float64[]
    sorted_intersect_faces_all!(rvec, r, mesh)
    rvec_final = [rvec[1]]
    N = 1
    for rval in rvec
        if (rval - rvec_final[N]) > 1e-6
            push!(rvec_final, rval)
            N += 1
        end
    end
    rvec = rvec_final
#    println("rvec: ", rvec)
#    readline()
    points = r.(rvec)
#    println("points: ", points)
#    readline()
    midpoints = (points[1:end-1] .+ points[2:end]) ./ 2
#    println("midpoints: ", midpoints)
#    readline()
    mesh_faces = [find_face(midpoint, mesh) for midpoint in midpoints]
#    println("mesh_faces: ", mesh_faces)
#    readline()
    for i in 1:length(mesh_faces)
        if mesh_faces[i] == 0 
            println("Could not find face for for ", midpoints[i])
        end
    end
    mesh_materials = [mats[f] for f in mesh_faces]
#    println("mesh_materials: ", mesh_materials)
#    readline()
    ray_origin = r(0.0)
    ray_origin_x_str = string(round(ray_origin[1], digits = 3))
    ray_origin_y_str = string(round(ray_origin[2], digits = 3))
    for i in 1:length(mesh_faces)
        color = color_map[mat_names[mesh_materials[i]]]
        println("\\draw [pedge, fill=", color, ", rotate around={",γ_str,":(",
                    ray_origin_x_str,", ",ray_origin_y_str,")}]")
        xy1 = ray_origin + Point(rvec[i  ], -s_eff/2)
        xy2 = ray_origin + Point(rvec[i+1],  s_eff/2)
        x1_str = string(round(xy1[1], digits = 3))
        y1_str = string(round(xy1[2], digits = 3))
        x2_str = string(round(xy2[1], digits = 3))
        y2_str = string(round(xy2[2], digits = 3))
        println("  (" * x1_str * ", " * y1_str * ") rectangle (" 
                      * x2_str * ", " * y2_str * ");")
    end
end


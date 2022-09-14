
###########################################################
# ONLY VALID FOR RINGS OF RADIUS LESS THAN HALF THE PITCH #
###########################################################

# Parameters
filename = "default_quad_mesh.inp"
order = 2 # mesh order
r_mat = [0.4096, 0.475, 0.575] # radius of the rings for each material
rdivs = [3, 1, 1] # radial divisions for each material
n_azi = 8
pitch = 1.26
materials = ["Fuel", "Clad", "Water", "Water"]


# Generate the mesh
####################################################################################
if (n_azi & (n_azi - 1)) != 0 
    error("Number of azimuthal divisions must be a power of 2")
end
if any(r_mat .> pitch/2)
    error("Radius of rings must be less than half the pitch")
end
include("default_mesh_utils.jl")

# Get the radius and area of each ring after it has been divided
r_div, a_div = get_divided_radii_and_areas(r_mat, rdivs,  pitch)

# Get the equivalent radii for quad or quad8
if order == 1
    r_equiv = get_equiv_quad_radii(r_div, a_div, n_azi, pitch)
    points = get_quad_mesh_points(r_equiv, n_azi, pitch)
    faces = get_quad_mesh_faces(sum(rdivs), n_azi)
    write_quad_mesh(filename, pitch, points, faces, rdivs, materials)
elseif order == 2
    r_equiv = get_equiv_quad8_radii(r_div, a_div, n_azi, pitch)
    points = get_quad8_mesh_points(r_equiv, n_azi, pitch)
    faces = get_quad8_mesh_faces(sum(rdivs), n_azi)
    write_quad8_mesh(filename, pitch, points, faces, rdivs, materials)
else
    error("Order of quadrilateral must be 1 or 2")
end

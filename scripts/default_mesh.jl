
#####################################################################################
# ONLY VALID FOR 8 AZIMUTHAL DIVISIONS AND RINGS OF RADIUS LESS THAN HALF THE PITCH #
#####################################################################################

# Parameters
filename = "default_quad_mesh.inp"
order = 2
radii = [0.4096, 0.475, 0.575] 
radial_divisions = [3, 1, 1]
const azimuthal_divisions = 8
pitch = 1.26
materials = ["Fuel", "Clad", "Water", "Water"]


# Generate the mesh
####################################################################################
include("default_mesh_utils.jl")

# Get the radius and area of each ring after it has been divided
radial_areas = get_radial_areas(radii, pitch)
div_radii, div_areas = get_divided_radii_and_areas(radii, radial_divisions, radial_areas, pitch)

# Get the equivalent radii for quad or quad8
if order == 1
    equiv_radii = get_equiv_quad_radii(div_radii, div_areas, pitch)
    points = get_quad_mesh_points(equiv_radii, pitch)
    faces = get_quad_mesh_faces(sum(radial_divisions))
    write_quad_mesh(filename, points, faces, radial_divisions, materials)
elseif order == 2
    equiv_radii = get_equiv_quad8_radii(div_radii, div_areas, pitch)
    points = get_quad8_mesh_points(equiv_radii, pitch)
    faces = get_quad8_mesh_faces(sum(radial_divisions))
else
    error("Order of quadrilateral must be 1 or 2")
end

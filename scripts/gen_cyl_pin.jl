include("gen_cyl_pin_utils.jl")

# Parameters
filename = "default_quad_mesh.inp"
order = 1
radii = [0.4096, 0.475, 0.575] 
radial_divisions = [3, 1, 1]
azimuthal_divisions = 8
pitch = 1.26
materials = ["Fuel", "Clad", "Water", "Water"]

# Get the radius and area of each ring after it has been divided
radial_areas = get_radial_areas(radii, pitch)
div_radii, div_areas = get_divided_radii_and_areas(radii, radial_divisions, radial_areas, pitch)

# Get the equivalent radii for quad or quad8
if order == 1
    equiv_radii = get_equiv_quad_radii(div_radii, div_areas, pitch, azimuthal_divisions)
    points = get_quad_mesh_points(equiv_radii, pitch, azimuthal_divisions)
    faces = get_quad_mesh_faces(sum(radial_divisions), azimuthal_divisions)
    write_quad_mesh(filename, points, faces, radial_divisions, azimuthal_divisions, materials)
#elseif order == 2
#    equiv_radii = get_equiv_quad8_radii(div_radii, div_areas, pitch, azimuthal_divisions)
else
    error("Order of quadrilateral must be 1 or 2")
end

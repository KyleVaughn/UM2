# Meant to produce the MPACT fine mesh discretization for the 1a_nogap
# case.  This is a manual version of the script, which is meant to
# preserve areas of the mesh that are important for the physics.
filename = "1a_quad2.inp"
r_mat = [0.4096, 0.475, 0.582] # radius of the rings for each material
rdivs = [5, 1, 1] # radial divisions for each material
n_azi = 16
pitch = 1.26
materials = ["Fuel", "Clad", "Water", "Water"]
elsets = ["Lattice_00001", "Module_00001", "Cell_00001"]

include("manual_mesh.jl")

r_div, a_div = get_divided_radii_and_areas(r_mat, rdivs,  pitch)
r_equiv = get_equiv_quad8_radii(r_div, a_div, n_azi, pitch)
points = get_quad8_mesh_points(r_equiv, n_azi, pitch)
faces = get_quad8_mesh_faces(sum(rdivs), n_azi)

# These should be tweaked ≈ 1e-3 to perfectly match the original mesh
r0 = 0.682
r1 = 0.772
θ = 2 * π / n_azi
γ = θ / 2
ω = θ / 4

# Move points to the correct locations
points[ 64] = (r0 * cos( 2θ), r0 * sin( 2θ))
points[160] = (r0 * cos( 6θ), r0 * sin( 6θ))
points[256] = (r0 * cos(10θ), r0 * sin(10θ))
points[352] = (r0 * cos(14θ), r0 * sin(14θ))
xa = acos(pitch / (2 * r1))
ya = asin(pitch / (2 * r1))
points[ 49] = (r1 * cos(xa), r1 * sin(xa))
points[ 73] = (r1 * cos(ya), r1 * sin(ya))
points[145] = (-r1 * cos(ya), r1 * sin(ya))
points[169] = (-r1 * cos(xa), r1 * sin(xa))
points[241] = (-r1 * cos(xa), -r1 * sin(xa))
points[265] = (-r1 * cos(ya), -r1 * sin(ya))
points[337] = (r1 * cos(ya), -r1 * sin(ya))
points[361] = (r1 * cos(xa), -r1 * sin(xa))

# Insert new points
for i in [3, 5, 11, 13, 19, 21, 27, 29]
    push!(points, (r0 * cos(i * γ), r0 * sin(i * γ)))
end
for i in [2, 6, 10, 14]
    push!(points, (r1 * cos(i * θ), r1 * sin(i * θ)))
end
push!(points, (points[ 63] .+ points[ 64]) ./ 2)
push!(points, (points[159] .+ points[160]) ./ 2)
push!(points, (points[255] .+ points[256]) ./ 2)
push!(points, (points[351] .+ points[352]) ./ 2)
for i in [5, 7, 9, 11, 21, 23, 25, 27, 37, 39, 41, 43, 53, 55, 57, 59]
    sinω, cosω = sincos(i * ω)
    rx = abs(pitch / (2 * cosω))
    ry = abs(pitch / (2 * sinω))
    rb = min(rx, ry)
    push!(points, (rb * cosω, rb * sinω))
end
for i in [7, 9, 23, 25, 39, 41, 55, 57]
    sinω, cosω = sincos(i * ω)
    push!(points, (r1 * cosω, r1 * sinω))
end
r01 = (r0 + r1) / 2
push!(points, (r01 * cos( 2θ), r01 * sin( 2θ)))
push!(points, (r01 * cos( 6θ), r01 * sin( 6θ)))
push!(points, (r01 * cos(10θ), r01 * sin(10θ)))
push!(points, (r01 * cos(14θ), r01 * sin(14θ)))


# Modify faces
f = faces[16]
faces[16] = (f[1], f[2], f[3] - 1, f[4], f[5], 402, 414, f[8])
f = faces[24]
faces[24] = (f[1], f[2] - 1, f[3], f[4], 414, 403, f[7], f[8])
f = faces[48]
faces[48] = (f[1], f[2], f[3] - 1, f[4], f[5], 404, 415, f[8])
f = faces[56]
faces[56] = (f[1], f[2] - 1, f[3], f[4], 415, 405, f[7], f[8])
f = faces[80]
faces[80] = (f[1], f[2], f[3] - 1, f[4], f[5], 406, 416, f[8])
f = faces[88]
faces[88] = (f[1], f[2] - 1, f[3], f[4], 416, 407, f[7], f[8])
f = faces[112]
faces[112] = (f[1], f[2], f[3] - 1, f[4], f[5], 408, 417, f[8])
f = faces[120]
faces[120] = (f[1], f[2] - 1, f[3], f[4], 417, 409, f[7], f[8])

# Insert new faces
push!(faces, (41, 49, 410, 64, 418, 434, 442, 402))
push!(faces, (64, 410, 73, 89, 442, 435, 421, 403))
push!(faces, (137, 145, 411, 160, 422, 436, 443, 404))
push!(faces, (160, 411, 169, 185, 443, 437, 425, 405))
push!(faces, (233, 241, 412, 256, 426, 438, 444, 406))
push!(faces, (256, 412, 265, 281, 444, 439, 429, 407))
push!(faces, (329, 337, 413, 352, 430, 440, 445, 408))
push!(faces, (352, 413, 361, 377, 445, 441, 433, 409))

push!(faces, (49, 419, 65, 410, 446, 447, 448, 434))
push!(points, (points[49] .+ points[419]) ./ 2)
push!(points, (points[419] .+ points[65]) ./ 2)
push!(points, (points[65] .+ points[410]) ./ 2)
push!(faces, (410, 65, 420, 73, 448, 449, 450, 435))
push!(points, (points[65] .+ points[420]) ./ 2)
push!(points, (points[420] .+ points[73]) ./ 2)

push!(faces, (145, 423, 161, 411, 451, 452, 453, 436))
push!(points, (points[145] .+ points[423]) ./ 2)
push!(points, (points[423] .+ points[161]) ./ 2)
push!(points, (points[161] .+ points[411]) ./ 2)
push!(faces, (411, 161, 424, 169, 453, 454, 455, 437))
push!(points, (points[161] .+ points[424]) ./ 2)
push!(points, (points[424] .+ points[169]) ./ 2)

push!(faces, (241, 427, 257, 412, 456, 457, 458, 438))
push!(points, (points[241] .+ points[427]) ./ 2)
push!(points, (points[427] .+ points[257]) ./ 2)
push!(points, (points[257] .+ points[412]) ./ 2)
push!(faces, (412, 257, 428, 265, 458, 459, 460, 439))
push!(points, (points[257] .+ points[428]) ./ 2)
push!(points, (points[428] .+ points[265]) ./ 2)

push!(faces, (337, 431, 353, 413, 461, 462, 463, 440))
push!(points, (points[337] .+ points[431]) ./ 2)
push!(points, (points[431] .+ points[353]) ./ 2)
push!(points, (points[353] .+ points[413]) ./ 2)
push!(faces, (413, 353, 432, 361, 463, 464, 465, 441))
push!(points, (points[353] .+ points[432]) ./ 2)
push!(points, (points[432] .+ points[361]) ./ 2)

write_quad8_mesh(filename, pitch, points, faces, rdivs, materials, elsets)

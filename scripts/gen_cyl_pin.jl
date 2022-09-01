using UM2

order = 1
radii = [0.4096, 0.475, 0.575] 
radial_divisions = [3, 1, 1]
azimuthal_divisions = 8
pitch = 1.26
materials = ["Fuel", "Clad", "Water", "Water"]

# Get the areas of each radial region
areas = Vector{Float64}(undef, length(radii) + 1)
areas[1] = π * radii[1]^2
for ir = 2:length(radii)
    areas[ir] = π * (radii[ir]^2 - radii[ir-1]^2)
end
areas[end] = pitch^2 - π * radii[end]^2

# Get all radii and areas after splitting each radial region with its radial divisions
all_areas = Vector{Float64}(undef, sum(radial_divisions) + 1)
all_radii = Vector{Float64}(undef, sum(radial_divisions))
# Inside the innermost ring 
all_areas[1] = areas[1] / radial_divisions[1]
all_radii[1] = sqrt(all_areas[1] / π)
# All other rings
for ir in 2:radial_divisions[1]
    all_areas[ir] = areas[1] / radial_divisions[1]
    all_radii[ir] = sqrt(all_areas[ir] / π + all_radii[ir-1]^2)
end
ctr = radial_divisions[1] + 1
for ir in 2:length(radii)
    nrad = radial_divisions[ir]
    for _ in 1:nrad
        all_areas[ctr] = areas[ir] / nrad
        all_radii[ctr] = sqrt(all_areas[ctr] / π + all_radii[ctr-1]^2)
        global ctr += 1
    end
end
# Outside the outermost ring
all_areas[end] = pitch^2 - π * all_radii[end]^2

# Get equivalent radii
θ = 2 * π / azimuthal_divisions
equiv_radii = Vector{Float64}(undef, length(all_radii))
# The innermost radius sn a special case, and is essentially a triangle.
# A_t = l² * sin(θ) / 2
equiv_radii[1] = sqrt(2 * all_areas[1] / (sin(θ) * azimuthal_divisions))
# A_q = (l² - l²₀) * sin(θ) / 2
for ir in 2:length(all_radii)
    equiv_radii[ir] = sqrt(2 * all_areas[ir] / (sin(θ) * azimuthal_divisions) + equiv_radii[ir-1]^2)
end
if any(r -> r > pitch/2, equiv_radii)
    error("Equivalent radii are greater than half the pitch")
end

# Compare.
Ar = Vector{Float64}(undef, length(all_radii))
Ap = Vector{Float64}(undef, length(all_radii))
Ar[1] = π * all_radii[1]^2
Ap[1] = equiv_radii[1]^2 * sin(θ) / 2
for ir in 2:length(equiv_radii)
    Ar[ir] = π * (all_radii[ir]^2 - all_radii[ir-1]^2)
    Ap[ir] = (equiv_radii[ir]^2 - equiv_radii[ir-1]^2) * sin(θ) / 2
end
if any(i -> abs(Ar[i] - azimuthal_divisions * Ap[i]) > 1e-6, 1:length(Ar))
    error("Ar and Ap are not equal")
end

# Generate the mesh points 
points = Point{2, Float64}[]
push!(points, Point{2, Float64}(0.0, 0.0))
# Inside the innermost ring
for ia in 1:2:azimuthal_divisions
    r = equiv_radii[1] / 2
    push!(points, Point{2, Float64}(r * cos(ia * θ), r * sin(ia * θ)))
end
# All other rings
for ir in 1:length(all_radii)
    for ia in 0:azimuthal_divisions - 1
        push!(points, Point{2, Float64}(all_radii[ir] * cos(ia * θ), all_radii[ir] * sin(ia * θ)))
    end
end
# Outside the outermost ring
for ia in 0:azimuthal_divisions - 1
    if abs(cos(ia * θ)) < 1e-3
        r = pitch / 2
    else
        r = abs(pitch / (2 * cos(ia * θ)))
    end
    push!(points, Point{2, Float64}(r * cos(ia * θ), r * sin(ia * θ)))
end

faces = NTuple{4, Int64}[]
# Inside the innermost ring
na = azimuthal_divisions
na2 = na ÷ 2
for ia = 1:2:na
    p1 = 1
    p2 = ia + (1 + na2)
    p3 = ia + 1 + (1 + na2)
    p4 = ia ÷ 2 + 2
    p5 = ia + 2 + (1 + na2)
    if p5 == 1 + na + (1 + na2)
        p5 = 1 + (1 + na2)
    end
    push!(faces, (p1, p2, p3, p4))
    push!(faces, (p1, p4, p3, p5))  
end
# Rings
nr = length(all_radii) - 1
for ir = 1:nr
    for ia = 1:na
        p1 = ia     + (ir - 1) * na + (1 + na2) 
        p2 = ia     + (ir    ) * na + (1 + na2) 
        p3 = ia + 1 + (ir    ) * na + (1 + na2) 
        p4 = ia + 1 + (ir - 1) * na + (1 + na2) 
        if p3 == 1 + (ir + 1) * na + (1 + na2)
            p3 -= na
        end
        if p4 == 1 + (ir    ) * na + (1 + na2)
            p4 -= na
        end
        push!(faces, (p1, p2, p3, p4))
    end
end
# Outside the outermost ring
for ia = 1:na
    p1 = ia +     (nr    ) * na + (1 + na2) 
    p2 = ia +     (nr + 1) * na + (1 + na2)
    p3 = ia + 1 + (nr + 1) * na + (1 + na2) 
    p4 = ia + 1 + (nr    ) * na + (1 + na2)
    if p3 == 1 + na * (nr + 2) + (1 + na2)
        p3 -= na
    end
    if p4 == 1 + na * (nr + 1) + (1 + na2)
        p4 -= na
    end
    push!(faces, (p1, p2, p3, p4))
end

# Write the file
io = open("default_mesh_quad.inp", "w");
try
    println(io, "*Heading")
    println(io, " default_mesh_quad.inp")
    println(io, "*NODE")
    for (i, p) in enumerate(points)
        println(io, i, ", ", p[1], ", ", p[2], ", 0.0")
    end
    println(io, "*ELEMENT, type=CPS4, ELSET=ALL")
    for (i, f) in enumerate(faces)
        println(io, i, ", ", f[1], ", ", f[2], ", ", f[3], ", ", f[4])
    end
    fctr = 1
    for (i, mat) in enumerate(materials)
        if i == 1 || mat != materials[i-1]
            println(io, "*ELSET,ELSET=Material:_" * mat)
            ndiv = radial_divisions[i]
        else
            ndiv = 1
        end
        for j in 1:ndiv
            for k in 1:azimuthal_divisions - 1
                print(io, fctr, ", ")
                fctr += 1
            end
            print(io, fctr, ",\n")
            fctr += 1
        end
    end
catch e
    println(e)
finally
    close(io)
end

#f = Figure(resolution = (1024, 1024)); ax = Axis(f[1,1]); display(f)
#scatter!(points)
#for i = 1:na*(nr+2)
#    println(i)
#    segments = Vector{LineSegment{Point{2, Float64}}}(undef, 4)
#    face = faces[i]
#    segments[1] = LineSegment(points[face[1]], points[face[2]])
#    segments[2] = LineSegment(points[face[2]], points[face[3]])
#    segments[3] = LineSegment(points[face[3]], points[face[4]])
#    segments[4] = LineSegment(points[face[4]], points[face[1]])
#    linesegments!(segments)
#    readline()
#end

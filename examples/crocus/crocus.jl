using MOCNeutronTransport
# Reactor-Physics-Benchmarks-Handbook/2015/LWR/CROCUS-LWR-RESR-001
gmsh.initialize()

uo2_fuel_entities = Int32[]
um_fuel_entities = Int32[]
gap_entities = Int32[]
clad_entities = Int32[]
absorber_entities = Int32[]
# The external radius of the water is 65 cm
# How much water is needed to count as infinite reflector?
# What is the minimum distance from fuel to boundary?
# Don't want to simulate what we don't have to. OR could make the mesh towards the problem
# boundary really large.
bb_apothem = 65.0
bb = (0.0, 2*bb_apothem, 0.0, 2*bb_apothem)
# gmsh.model.occ.add_disk(bb_apothem, bb_apothem, 0.0, 0.2, 0.2)

# UO2 pins
# ----------------------------------------------------------------------------------------------
# Fuel diameter = 10.52 mm (pg. 34)
# External cladding diameter = 12.6 mm (pg. 34)
# Cladding thickness = 0.85 mm (pg. 34)
# The document does not mention the gap thickness, so believe we are left to compute that quantity
# Gap thickness = (External cladding diameter - 2(Clad thickness) - Fuel diameter)/2
# Gap thickness = (12.6 - 2(0.85) - 10.52)/2 = 0.19 mm
# Therefore 
#   r_fuel = 10.52/2 = 5.26 mm = 0.526 cm
#   r_gap = 5.26 + 0.19 mm = 0.545 cm
#   r_clad = 12.6/2 = 6.3 mm = 0.63 cm 
r_fuel = 0.526
r_gap = 0.545
r_clad = 0.63
pitch = 1.8370 # pg.34
for i = -11:11
    if i == 0
        continue
    end
    j_max = [11, 11, 11, 9, 9, 9, 6, 6, 6, 3, 3]
    for j = (-j_max[abs(i)] + 1):j_max[abs(i)]
        push!(uo2_fuel_entities, 
              gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      j*pitch - pitch/2 + bb_apothem, 
                                      0, r_fuel, r_fuel))
        push!(gap_entities, 
              gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      j*pitch - pitch/2 + bb_apothem, 
                                      0, r_gap,  r_gap))
        push!(clad_entities, 
              gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      j*pitch - pitch/2 + bb_apothem, 
                                      0, r_clad, r_clad))
    end
end
# U metal pins
# ----------------------------------------------------------------------------------------------
# Fuel diameter = 17.0 mm (pg. 34)
# External cladding diameter = 19.3 mm (pg. 34)
# Cladding thickness = 0.975 mm (pg. 34)
# Gap thickness = (External cladding diameter - 2(Clad thickness) - Fuel diameter)/2
# Gap thickness = (19.3 - 2(0.975) - 17.0)/2 = 0.175 mm
# Therefore 
#   r_fuel = 17.0/2 = 8.5 mm = 0.85 cm
#   r_gap = 8.5 + 0.175 mm = 0.8675 cm
#   r_clad = 19.3/2 = 9.65 mm = 0.965 cm 
# Note that the distribution of pins is not quarter symmetric.
r_fuel = 0.85
r_gap = 0.8675
r_clad = 0.965
pitch = 2.917 # pg.34
for i = -10:10
    if i == 0
        continue
    end
    j_max = [10, 10, 10, 9, 9, 8, 8, 7, 5, 3]
    j_min = [ 8,  8,  7, 7, 5, 5, 3, 1, 1, 1]
    for j = j_min[abs(i)]:j_max[abs(i)]
        push!(uo2_fuel_entities, 
              gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      0, r_fuel, r_fuel))
        push!(gap_entities, 
              gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      0, r_gap,  r_gap))
        push!(clad_entities, 
              gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                                      0, r_clad, r_clad))
    end
end
for i = -10:10
    if i == 0
        continue
    end
    j_max = [10, 10, 10, 9, 9, 9, 8, 7, 6, 4]
    j_min = [ 8,  8,  7, 7, 5, 5, 3, 1, 1, 1]
    for j = (j_min[abs(i)] - 1):(j_max[abs(i)] - 1)
        # Empty tube
        if (i == 6 && j == 5) || (i == -6 && j == 5)
            push!(gap_entities, 
                  gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          -sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          0, r_gap,  r_gap))
            push!(clad_entities, 
                  gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          -sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          0, r_clad, r_clad))
        else
            push!(uo2_fuel_entities, 
                  gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          -sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          0, r_fuel, r_fuel))
            push!(gap_entities, 
                  gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          -sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          0, r_gap,  r_gap))
            push!(clad_entities, 
                  gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          -sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                                          0, r_clad, r_clad))
        end
    end
end
# Absorber rod
# ----------------------------------------------------------------------------------------------
# Absorber radius = 6.0 mm (pg. 32)
# Clad radius = 8.0 mm (pg. 32)
# gmsh.model.occ.add_disk(bb_apothem, bb_apothem, 0.0, 0.2, 0.2)
push!(uo2_fuel_entities, 
      gmsh.model.occ.add_disk(i*pitch - sign(i)*pitch/2 + bb_apothem, 
                              -sign(i)*j*pitch - sign(i)*pitch/2 + bb_apothem, 
                              0, r_fuel, r_fuel))














# Materials
# ----------------------------------------------------------------------------------------------
gmsh.model.occ.synchronize()
p = gmsh.model.add_physical_group(2, uo2_fuel_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_UO2")
p = gmsh.model.add_physical_group(2, gap_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_GAP")
p = gmsh.model.add_physical_group(2, clad_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_CLAD")
gmsh_group_preserving_fragment(gmsh.model.get_entities(2),
                               gmsh.model.get_entities(2);
                               material_hierarchy = ["MATERIAL_UO2", 
                                                     "MATERIAL_GAP",
                                                     "MATERIAL_CLAD"])

gmsh.fltk.run()
gmsh.finalize()

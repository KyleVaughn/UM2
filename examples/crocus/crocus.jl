using MOCNeutronTransport
# Reactor-Physics-Benchmarks-Handbook/2015/LWR/CROCUS-LWR-RESR-001
gmsh.initialize()
uo2_fuel_entities = Int32[]
um_fuel_entities = Int32[]
gap_entities = Int32[]
clad_entities = Int32[]


CANT BE QUARTER SYMMETRIC DUE TO EMPTY CELL. ENSURE CELL IS EMPTY FOR BENCHMARK

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
for i = 1:11
    j_max = [11, 11, 11, 9, 9, 9, 6, 6, 6, 3, 3]
    for j = 1:j_max[i]
        push!(uo2_fuel_entities, 
              gmsh.model.occ.add_disk(i*pitch + pitch/2, j*pitch + pitch/2, 0, r_fuel, r_fuel))
        push!(gap_entities, 
              gmsh.model.occ.add_disk(i*pitch + pitch/2, j*pitch + pitch/2, 0, r_gap,  r_gap))
        push!(clad_entities, 
              gmsh.model.occ.add_disk(i*pitch + pitch/2, j*pitch + pitch/2, 0, r_clad, r_clad))
    end
end
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
# gmsh.finalize()

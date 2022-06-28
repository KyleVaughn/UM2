using MOCNeutronTransport

# Model setup
# ----------------------------------------------------------------------------------------
gmsh.initialize()
gmsh.option.set_number("Geometry.OCCParallel", 1)
gmsh.option.set_number("General.NumThreads", 32)

# Import CAD file
gmsh.merge("/home/kcvaughn/Downloads/Core Assembly - Test Layout 2 - 6-30-20 Export.STEP")

# Shift the model to the origin of the xy-plane
dim_tags = gmsh.model.get_entities(3)
# Rotate the entities in dim_tags in the built-in CAD representation by angle radians 
# around the axis of revolution defined by the point (x, y, z) and the direction (ax, ay, az).
# gmsh.model.occ.rotate(dim_tags, x, y, z, ax, ay, az, angle)
gmsh.model.occ.rotate(dim_tags, 0, 0, 0, 1, 0, 0, π / 2)
# Translate it so that the corner is at the origin
#gmsh.model.occ.translate(dim_tags, 37.3225 - 0.0099 + 10*grid_offset, 
#                                   44.09 - 0.00174209 + 10*grid_offset, 
#                                   -436.5625)
# Model is in mm. Convert to cm by shrinking everything by 1/10
#gmsh.model.occ.dilate(dim_tags, 0, 0, 0, 1//10, 1//10, 0)

rect_tag = gmsh.model.occ.add_rectangle(-500.0, -500.0, 300, 1000.0, 1000.0)

# Synchronize the CAD and Gmsh models
gmsh.model.occ.synchronize()

#Geometry.OCCParallel
# Verify results
# gmsh.fltk.run()
intersections = gmsh.model.occ.intersect([(Int32(2), rect_tag)], dim_tags)
gmsh.model.occ.synchronize()
gmsh.write("FNR_slice.step")
#gmsh.finalize()

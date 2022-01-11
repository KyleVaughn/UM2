

# Minimum segment length for ray tracing
const minimum_segment_length = 1e-4

# QuadraticSegment_2D
# ---------------------------------------------------------------------------------------------------
# Threshold beneath which the segment is treated as linear.
# Equal to 2|d⃗|θ, where d⃗ is the displacement of x⃗₃ from the midpoint of the linear segment (x⃗₁, x⃗₂)
# and θ is the angle between the intersecting line and d⃗
const QuadraticSegment_2D_1_intersection_ϵ = 5e-6

# UnstructuredMesh_2D
# ---------------------------------------------------------------------------------------------------
# Cell types are the same as VTK
const UnstructuredMesh_2D_linear_cell_types = UInt32[5, # Triangle 
                                                     9  # Quadrilateral
                                                    ]
const UnstructuredMesh_2D_quadratic_cell_types = UInt32[22, # Triangle6
                                                        23  # Quadrilateral8
                                                       ]
const UnstructuredMesh_2D_cell_types = vcat(UnstructuredMesh_2D_linear_cell_types,
                                            UnstructuredMesh_2D_quadratic_cell_types)

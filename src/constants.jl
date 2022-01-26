# Minimum segment length for ray tracing
const minimum_segment_length = 1e-4

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

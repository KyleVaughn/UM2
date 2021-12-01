# Constants used in more than 1 file, very important constants, and constants that would be  

# Parametric coordinate fudge factor
const parametric_coordinate_ϵ = 5e-6

# Minimum segment length for ray tracing
const minimum_segment_length = 1e-4


# Point_2D
# ---------------------------------------------------------------------------------------------------
# Minimum distance between two points to be considered different
const Point_2D_differentiation_distance = 5e-6

# LineSegment_2D
# ---------------------------------------------------------------------------------------------------
# θ ∈ [0, π/2] between two lines, each translated to share a point, beneath which the lines
# are declared to be parallel
# -------------> u⃗
# \θ)
#  \
#   \
#    \ 
#     v
#      v⃗
const LineSegment_2D_parallel_θ = π/180000 # ≈ 0.001°

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

# Ray tracing
# ---------------------------------------------------------------------------------------------------
# Visualization of edge-to-edge ray tracing.
const visualize_ray_tracing = false 

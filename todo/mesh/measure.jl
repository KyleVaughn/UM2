export element_measures 

element_measures(mesh::AbstractMesh) = map(measure, materialize_polytopes(mesh))

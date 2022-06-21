# UM<sup>2</sup> - The University of Michigan Unstructured Mesh Code
A tool for automating the pipeline from CAD model to hierarchical unstructured mesh, for use in method of characteristics neutron transport.

This tool:
1. Imports a CAD model of the STEP, IGES, or BREP file format.
2. Overlays a hierarchical rectangular grid onto the model. This grid is optimized for use with coarse mesh finite difference and parallel decomposition methods.
3. Generates an unstructured mesh to discretize the domain into source regions. The meshing process uses material/cross section information to attempt to minimze the total number of mesh cells, whlile maintaining accuracy. Supports linear and quadratic meshes. 
4. Exports the mesh in Abaqus, VTK, or XDMF format.

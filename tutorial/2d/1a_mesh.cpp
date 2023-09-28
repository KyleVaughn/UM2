// Model reference:
//  VERA Core Physics Benchmark Progression Problem Specifications
//  Revision 4, August 29, 2014
//  CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  // Import the geometry from the brep file we created in the previous step.
  // The "extra_info" flag is set to true so that we can import the groups and
  // colors of the geometric entities in the brep file.
  um2::gmsh::open("1a.brep", /*extra_info=*/true);

  // Our goal is to create a model which can be used in an MPACT simulation.
  // Therefore, we need to partition the domain in a way that maps to MPACT's
  // spatial hierarchy. MPACT is a 2D/1D code, so we will effectively be creating
  // a stack of 2D meshes (axial slices). In this case, since the model is 2D, we just
  // have one slice.
  //
  // The MPACT spatial partition consists of:
  //  1. The core
  //    A rectilinear partition of the XY-domain into assemblies. All assemblies
  //    must have the same start and stop heights.
  //  2. Assemblies
  //    Each assembly is a rectilinear partition of the Z-domain into 2D axial slices.
  //    These axial slices are called "lattices".
  //  3. Lattices
  //    A regular partition of the XY-domain into equal-sized axis-aligned
  //    rectangles, also known as "ray tracing modules".
  //  4. Ray tracing modules (RTMs)
  //    A rectilinear partition of the XY-domain into coarse cells.
  //    Every RTM is exactly the same width and height in all lattices.
  //    This property is a necessity for modular ray tracing.
  //  5. Coarse cells
  //    A 2D axis-aligned box (AABB), containing a mesh which completely
  //    fills the box's interior. This mesh is the "fine mesh". It is made
  //    up of fine cells (triangles, quadrilaterals, etc.).
  //
  //  In MPACT, the coarse cells typically contain the geometry for a single
  //  pin, centered in the middle of the coarse cell. However, UM2 allows the coarse
  //  cells to contain a piece of a pin, multiple pins, or any other arbitrary
  //  geometry.

  // We will now build the model from the ground up, starting with the coarse cells.
  um2::mpact::SpatialPartition model;

  // We create a coarse cell that is 1.26 cm x 1.26 cm in size. It is implicitly
  // given the ID of 0.
  double const pitch = 1.26; // Pitch = 1.26 cm (pg. 4)
  model.makeCoarseCell({pitch, pitch});

  // The next level of the hierarchy is the ray tracing module (RTM). The RTM
  // is a 2D array of coarse cells. We can create the RTM by providing
  // a vector of vectors of coarse cell IDs. In this case, we only have one coarse cell.
  model.makeRTM({{0}});

  // The next level of the hierarchy is the lattice. The lattice is a 2D array
  // of RTMs. We can create the lattice by providing a vector of vectors of RTM IDs.
  // In this case, we only have one RTM.
  model.makeLattice({{0}});

  // The next level of the hierarchy is the assembly. The assembly is a 1D array (a
  // vertical stack) of lattices. We can create the assembly by providing a vector of
  // lattice IDs. In 2D problems, each lattice should map one-to-one to an assembly.
  model.makeAssembly({0});

  // The next level of the hierarchy is the core. The core is a 2D array of assemblies.
  // We can create the core by providing a vector of vectors of assembly IDs. In this
  // case, we only have one assembly.
  model.makeCore({{0}});

  // We finally have the full spatial hierarchy. We will now overlay the spatial
  // partition onto the geometry, creating the necessary boxes to contain each
  // unique coarse cell. During this process, any area that does not already
  // have a material is assigned "Material_Moderator" and the color "royalblue".
  // This can be changed by providing optional arguments to the function.
  // UM2 assumes that the bottom left corner of the domain is at (0, 0, 0).
  // Using this assumption the position of all coarse cells, RTMs, lattices,
  // etc. can be calculated.
  um2::gmsh::model::occ::overlaySpatialPartition(model);

  // Uncomment the following line to see the spatial partition overlaid on the geometry.
  // um2::gmsh::fltk::run();

  // Our model is now ready to be meshed. We will set the characteristic mesh size
  // to 0.15 cm for the entire model. This is the target mesh edge length.
  double const lc = 0.15;
  um2::gmsh::model::mesh::setGlobalMeshSize(lc);

  // Alternatively, we can set the characteristic mesh size for groups of entities.
  // The function below sets the mesh size for the 2D entities in the groups
  // "Material_Fuel", "Material_Gap", "Material_Clad", and "Material_Moderator"
  // to 0.1 cm, 0.01 cm, 0.1 cm, and 0.2 cm, respectively.
  // um2::gmsh::model::mesh::setMeshFieldFromGroups(
  //     2, {"Material_Fuel", "Material_Gap", "Material_Clad", "Material_Moderator"},
  //     {0.1, 0.01, 0.1, 0.2});

  // We need to specify the mesh type. The options are:
  // Tri            - Triangular mesh
  // Quad           - Quadrilateral mesh
  // QuadraticTri   - Quadratic triangular mesh
  // QuadraticQuad  - Quadratic quadrilateral mesh
  //
  // It is recommended that you use a quadratic triangle mesh when meshing non-trivial
  // geometries. This will allow for a more accurate representation of the geometry,
  // minimizing the error introduced by a failure to preserve material areas/volumes.
  // For common geometries like this, UM2 has mesh builders that can perfectly preserve
  // material areas/volumes, but this will be covered in a later example.
  // For now, we will use a general mesh generation approach. This will not preserve
  // material areas/volumes exactly.
  um2::MeshType const mesh_type = um2::MeshType::QuadraticTri;

  // We can now generate the mesh. This will create a mesh for the entire model.
  // Note that warnings about a jacobian determinant less than 0 are usually fine for
  // quadratic elements, since we only care that the mesh is geometrically valid
  // (no overlapping faces, etc.), not that each face has an invertible transformation
  // to the reference element.
  um2::gmsh::model::mesh::generateMesh(mesh_type);

  // Uncomment the following line to see the mesh.
  // um2::gmsh::fltk::run();

  // We can now export the mesh to a file.
  um2::gmsh::write("1a.inp");

  // We now have a fine MOC mesh to go inside of each of the coarse cells in our
  // model, but we need to export a model which also contains the MPACT spatial hierarchy
  // information. To do this we will first use the mesh we just created to populate the
  // otherwise empty coarse cells in the spatial partition.
  model.importCoarseCells("1a.inp");

  // We can now export the model to a file. This will create a file called "1a.xdmf".
  // The XDMF file format consists of an XML file (1a.xdmf) and a binary HDF5 file
  // (1a.h5). The XML file contains the spatial hierarchy information and the HDF5 file
  // contains the mesh information. Therefore, both files are required to load the model
  // into MPACT. Furthermore, the XML file contains the path to the HDF5 file, so if the
  // h5 file is renamed, the XML file must be updated to reflect the new name.
  //
  // We choose this format over the Abaqus .inp format or others due to the ability to
  // compress and chunk the HDF5 file. This allows for much faster loading of the model
  // into MPACT and for much smaller file sizes.
  um2::exportMesh("1a.xdmf", model);

  um2::finalize();
  return 0;
}

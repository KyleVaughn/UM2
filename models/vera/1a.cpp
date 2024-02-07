// Model reference:
//  VERA Core Physics Benchmark Progression Problem Specifications
//  Revision 4, August 29, 2014
//  CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  //============================================================================
  // Materials
  //============================================================================
  // Nuclides and number densities from table P1-4 (pg. 23)

  // Fuel
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setDensity(10.42); // g/cm^3, Table P1-2 (pg. 20)
  fuel.setTemperature(565.0); // K, Table P1-1 (pg. 20)
  fuel.setColor(um2::forestgreen);
  fuel.addNuclide("U234", 6.11864e-6); // Number density in atoms/b-cm
  fuel.addNuclide("U235", 7.18132e-4);
  fuel.addNuclide("U236", 3.29861e-6);
  fuel.addNuclide("U238", 2.21546e-2);
  fuel.addNuclide("O16", 4.57642e-2);

  // Gap
  um2::Material gap;
  gap.setName("Gap");
  // "Helium with nominal density" (pg. 22). He at 565 K and 2250 psia, according to NIST
  // has a density of 0.012768 g/cm^3.
  gap.setDensity(0.012768); // g/cm^3,
  gap.setTemperature(565.0); // K, Hot zero power temperature (pg. 20)
  gap.setColor(um2::red);
  gap.addNuclide("He4", 2.68714e-5);

  // Clad
  um2::Material clad;
  clad.setName("Clad");
  clad.setDensity(6.56); // g/cm^3, (pg. 18)
  clad.setTemperature(565.0); // K, Hot zero power temperature (pg. 20)
  clad.setColor(um2::slategray);
  clad.addNuclide("Zr90", 2.18865e-02);
  clad.addNuclide("Zr91", 4.77292e-03);
  clad.addNuclide("Zr92", 7.29551e-03);
  clad.addNuclide("Zr94", 7.39335e-03);
  clad.addNuclide("Zr96", 1.19110e-03);
  clad.addNuclide("Sn112", 4.68066e-06);
  clad.addNuclide("Sn114", 3.18478e-06);
  clad.addNuclide("Sn115", 1.64064e-06);
  clad.addNuclide("Sn116", 7.01616e-05);
  clad.addNuclide("Sn117", 3.70592e-05);
  clad.addNuclide("Sn118", 1.16872e-04);
  clad.addNuclide("Sn119", 4.14504e-05);
  clad.addNuclide("Sn120", 1.57212e-04);
  clad.addNuclide("Sn122", 2.23417e-05);
  clad.addNuclide("Sn124", 2.79392e-05);
  clad.addNuclide("Fe54", 8.68307e-06);
  clad.addNuclide("Fe56", 1.36306e-04);
  clad.addNuclide("Fe57", 3.14789e-06);
  clad.addNuclide("Fe58", 4.18926e-07);
  clad.addNuclide("Cr50", 3.30121e-06);
  clad.addNuclide("Cr52", 6.36606e-05);
  clad.addNuclide("Cr53", 7.21860e-06);
  clad.addNuclide("Cr54", 1.79686e-06);
  clad.addNuclide("Hf174", 3.54138e-09);
  clad.addNuclide("Hf176", 1.16423e-07);
  clad.addNuclide("Hf177", 4.11686e-07);
  clad.addNuclide("Hf178", 6.03806e-07);
  clad.addNuclide("Hf179", 3.01460e-07);
  clad.addNuclide("Hf180", 7.76449e-07);

  // Moderator
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setDensity(0.743); // g/cm^3, Table P1-1 (pg. 20)
  moderator.setTemperature(565.0); // K, Table P1-1 (pg. 20)
  moderator.setColor(um2::blue);
  moderator.addNuclide("O16", 2.48112e-02);
  moderator.addNuclide("H1", 4.96224e-02);
  moderator.addNuclide("B10", 1.07070e-05);
  moderator.addNuclide("B11", 4.30971e-05);

  //============================================================================
  // Geometry
  //============================================================================

  // Parameters for the pin-cell geometry
  double const r_fuel = 0.4096; // Pellet radius = 0.4096 cm (pg. 4)
  double const r_gap = 0.418;   // Inner clad radius = 0.418 cm (pg. 4)
  double const r_clad = 0.475;  // Outer clad radius = 0.475 cm (pg.4)
  double const pitch = 1.26;    // Pitch = 1.26 cm (pg. 4)

  um2::Point2 const center = {pitch / 2, pitch / 2};

  um2::Vector<double> const radii = {r_fuel, r_gap, r_clad};
  um2::Vector<um2::Material> const materials = {fuel, gap, clad};

  um2::gmsh::model::occ::addCylindricalPin2D(center, radii, materials);

  um2::mpact::SpatialPartition model;

  // We create a coarse cell that is 1.26 cm x 1.26 cm in size. It is implicitly
  // given the ID of 0.
  double const pitch = 1.26; // Pitch = 1.26 cm (pg. 4)
//  model.makeCoarseCell({pitch, pitch});
//
//  // The next level of the hierarchy is the ray tracing module (RTM). The RTM
//  // is a 2D array of coarse cells. We can create the RTM by providing
//  // a vector of vectors of coarse cell IDs. In this case, we only have one coarse cell.
//  model.makeRTM({{0}});
//
//  // The next level of the hierarchy is the lattice. The lattice is a 2D array
//  // of RTMs. We can create the lattice by providing a vector of vectors of RTM IDs.
//  // In this case, we only have one RTM.
//  model.makeLattice({{0}});
//
//  // The next level of the hierarchy is the assembly. The assembly is a 1D array (a
//  // vertical stack) of lattices. We can create the assembly by providing a vector of
//  // lattice IDs. In 2D problems, each lattice should map one-to-one to an assembly.
//  model.makeAssembly({0});
//
//  // The next level of the hierarchy is the core. The core is a 2D array of assemblies.
//  // We can create the core by providing a vector of vectors of assembly IDs. In this
//  // case, we only have one assembly.
//  model.makeCore({{0}});
//
//  // We finally have the full spatial hierarchy. We will now overlay the spatial
//  // partition onto the geometry, creating the necessary boxes to contain each
//  // unique coarse cell. During this process, any area that does not already
//  // have a material is assigned "Material_Moderator" and the color "royalblue".
//  // This can be changed by providing optional arguments to the function.
//  // UM2 assumes that the bottom left corner of the domain is at (0, 0, 0).
//  // Using this assumption the position of all coarse cells, RTMs, lattices,
//  // etc. can be calculated.
//  um2::gmsh::model::occ::overlaySpatialPartition(model);
//
//  // Uncomment the following line to see the spatial partition overlaid on the geometry.
//  // um2::gmsh::fltk::run();
//
//  // Our model is now ready to be meshed. We will set the characteristic mesh size
//  // to 0.15 cm for the entire model. This is the target mesh edge length.
//  double const lc = 0.15;
//  um2::gmsh::model::mesh::setGlobalMeshSize(lc);
//
//  // Alternatively, we can set the characteristic mesh size for groups of entities.
//  // The function below sets the mesh size for the 2D entities in the groups
//  // "Material_Fuel", "Material_Gap", "Material_Clad", and "Material_Moderator"
//  // to 0.1 cm, 0.01 cm, 0.1 cm, and 0.2 cm, respectively.
//  // um2::gmsh::model::mesh::setMeshFieldFromGroups(
//  //     2, {"Material_Fuel", "Material_Gap", "Material_Clad", "Material_Moderator"},
//  //     {0.1, 0.01, 0.1, 0.2});
//
//  // We need to specify the mesh type. The options are:
//  // Tri            - Triangular mesh
//  // Quad           - Quadrilateral mesh
//  // QuadraticTri   - Quadratic triangular mesh
//  // QuadraticQuad  - Quadratic quadrilateral mesh
//  //
//  // It is recommended that you use a quadratic triangle mesh when meshing non-trivial
//  // geometries. This will allow for a more accurate representation of the geometry,
//  // minimizing the error introduced by a failure to preserve material areas/volumes.
//  // For common geometries like this, UM2 has mesh builders that can perfectly preserve
//  // material areas/volumes, but this will be covered in a later example.
//  // For now, we will use a general mesh generation approach. This will not preserve
//  // material areas/volumes exactly.
//  um2::MeshType const mesh_type = um2::MeshType::QuadraticTri;
//
//  // We can now generate the mesh. This will create a mesh for the entire model.
//  // Note that warnings about a jacobian determinant less than 0 are usually fine for
//  // quadratic elements, since we only care that the mesh is geometrically valid
//  // (no overlapping faces, etc.), not that each face has an invertible transformation
//  // to the reference element.
//  um2::gmsh::model::mesh::generateMesh(mesh_type);
//
//  // Uncomment the following line to see the mesh.
//  // um2::gmsh::fltk::run();
//
//  // We can now export the mesh to a file.
//  um2::gmsh::write("1a.inp");
//
//  // We now have a fine MOC mesh to go inside of each of the coarse cells in our
//  // model, but we need to export a model which also contains the MPACT spatial hierarchy
//  // information. To do this we will first use the mesh we just created to populate the
//  // otherwise empty coarse cells in the spatial partition.
//  model.importCoarseCells("1a.inp");
//
//  // We can now export the model to a file. This will create a file called "1a.xdmf".
//  // The XDMF file format consists of an XML file (1a.xdmf) and a binary HDF5 file
//  // (1a.h5). The XML file contains the spatial hierarchy information and the HDF5 file
//  // contains the mesh information. Therefore, both files are required to load the model
//  // into MPACT. Furthermore, the XML file contains the path to the HDF5 file, so if the
//  // h5 file is renamed, the XML file must be updated to reflect the new name.
//  //
//  // We choose this format over the Abaqus .inp format or others due to the ability to
//  // compress and chunk the HDF5 file. This allows for much faster loading of the model
//  // into MPACT and for much smaller file sizes.
//  um2::exportMesh("1a.xdmf", model);
//
  um2::finalize();
  return 0;
}

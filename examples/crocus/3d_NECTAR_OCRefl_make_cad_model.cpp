#include <um2.hpp>

auto
main() -> int
{

  // Create a 3 by 3 model that looks like this:
  // +--------+--------+--------+
  // |        |        |        |
  // |   H2O  | Umetal | Umetal |
  // |        |  Pin   |  Pin   |
  // +--------+--------+--------+
  // |        |        |        |
  // |   H2O  | NECTAR | Umetal |
  // |        |  Pin   |  Pin   |
  // +--------+--------+--------+
  // |        |        |        |
  // |   H2O  |   H2O  |   H2O  |
  // |        |        |        |
  // +--------+--------+--------+
  // Since UM2 will fill all empty space with water, we only need to make the
  // Umetal and NECTAR pins

  //============================================================================
  // Model parameters
  //============================================================================

  double constexpr r_fuel = 0.85;
  double constexpr r_gap = 0.8675;
  double constexpr r_innerclad = 0.965;
  double constexpr r_interclad = 1.1;
  double constexpr r_outerclad = 1.25;

  double constexpr pin_pitch = 2.917;
  double constexpr pin_height = 5.00;
  double constexpr half_pitch = 0.5 * pin_pitch;
  double constexpr half_height = 0.5 * pin_height;
  double constexpr dosi_height = 0.01;
  double constexpr holder_thick = 0.05;
  double constexpr cladax_height = 0.15;
  double constexpr holder_height = 1;

  // Materials
  um2::Material const umet("Umet", "forestgreen");
  um2::Material const air("Air", "white");
  um2::Material const clad("Clad", "slategray");
  um2::Material const gold("Gold", "yellow");

  // Umetal pin radii, materials, and coordinates
  std::vector<double> const umetal_radii = {r_fuel, r_gap, r_innerclad};
  std::vector<um2::Material> const umetal_mats = {umet, air, clad};
  // Integer coordinates in the lattice
  um2::gmsh::vectorpair const umetal_coords = {{2, 1}, {1, 2}, {2, 2}};

  // NECTAR pin radii, materials, and coordinates
  // The dosimeter contains 6 concentric rings and 8 azimuthal divisions
  std::vector<double> const nectar_dosi_radii = {0.3531, 0.4994, 0.6116,
                                                 0.7063, 0.7896, r_gap};
  // The x,y coordinates of the NECTAR pin center.
  double constexpr xn = 1.5 * pin_pitch;
  double constexpr yn = xn;

  // The regions directly on top and bottom of the dosimeter are pure clad
  std::vector<double> const dosi_cap_radii = {r_outerclad};
  std::vector<um2::Material> const dosi_cap_mats = {clad};
  double constexpr dosi_top_thick = holder_thick + cladax_height - dosi_height;
  double constexpr dosi_bot_thick = holder_thick + cladax_height;
  um2::Point3d const dosi_top(xn, yn, half_height + dosi_height);
  um2::Point3d const dosi_bot(xn, yn, half_height - dosi_bot_thick); 

  // After the dosimeter, the fuel is normal Umetal again
  um2::Point3d const fuel_top(xn, xn, half_height + dosi_bot_thick);
  um2::Point3d const fuel_bot(xn, xn, half_height - holder_thick - holder_height);
  double constexpr fuel_sec_thick = holder_height - cladax_height;

  // Then the pin transitions to a region of double clad
  std::vector<double> const double_clad_radii = {r_fuel, r_gap, r_innerclad, r_interclad, r_outerclad};
  std::vector<um2::Material> const double_clad_mats = {umet, air, clad, air, clad};
  um2::Point3d const double_clad_top(xn, xn, half_height + holder_height + holder_thick); 
  um2::Point3d const double_clad_bot(xn, xn, half_height - half_height); 
  double constexpr double_clad_thick = half_height - holder_height - holder_thick;

  //============================================================================
  // Create the model
  //============================================================================

  // Initialize UM2
  um2::initialize();

  // Alias for brevity
  namespace factory = um2::gmsh::model::occ;

  // Create the Umetal pins
  for (auto const & coord : umetal_coords) {
    // Compute the pin center based upon coordinate in the lattice
    um2::Point3d const center(coord.first * pin_pitch + half_pitch, 
                              coord.second * pin_pitch + half_pitch, 0.0);
    factory::addCylindricalPin(center, pin_height, umetal_radii, umetal_mats);
  }

  // Create the NECTAR pin
  // Start by adding the concentric cylinders of the dosimeter
  um2::gmsh::vectorpair cyl_dim_tags;
  for (auto const r : nectar_dosi_radii) {
    int const tag = factory::addCylinder(xn, yn, half_height, 0, 0, dosi_height, r);
    cyl_dim_tags.push_back({3, tag});
  }

  // Add boxes which will be used to slice the dosimeter azimuthally
  um2::gmsh::vectorpair box_dim_tags;
  for (int i = 0; i < 8; ++i) {
    int const tag = factory::addBox(xn, yn, half_height, pin_pitch, pin_pitch, dosi_height);
    std::pair<int, int> const box_dimtag = {3, tag};
    factory::rotate({box_dimtag}, xn, yn, half_height, 0, 0, 1, i * um2::pi_4<double>);
    box_dim_tags.push_back(box_dimtag);
  }
  // Use boolean operations to slice the dosimeter axially and radially
  um2::gmsh::vectorpair out_dim_tags;
  std::vector<um2::gmsh::vectorpair> out_dim_tags_map;
  um2::gmsh::model::occ::intersect(cyl_dim_tags, box_dim_tags, out_dim_tags, out_dim_tags_map);
  
  // Add the clad and use a boolean fragment with the dosimeter to create the pin
  int const tag =
    factory::addCylinder(1.5 * pin_pitch, 1.5 * pin_pitch, half_height, 0, 0, dosi_height, r_outerclad);
  out_dim_tags.push_back({3, tag});
  um2::gmsh::vectorpair const all_ents = out_dim_tags;
  out_dim_tags.clear();
  out_dim_tags_map.clear();
  um2::gmsh::model::occ::fragment(all_ents, all_ents, out_dim_tags, out_dim_tags_map);
  um2::gmsh::model::occ::synchronize();

  // Assign materials to the geometric entities
  std::vector<int> out_tags(out_dim_tags.size());
  for (size_t i = 0; i < out_dim_tags.size(); ++i) {
    out_tags[i] = out_dim_tags[i].second;
  }
  
  um2::gmsh::model::addToPhysicalGroup(3, out_tags, -1, "Material_Gold");
  um2::gmsh::model::setColor(out_dim_tags,
    gold.color.r(),
    gold.color.g(),
    gold.color.b(), 255, /*recursive=*/true);
  
  um2::gmsh::model::addToPhysicalGroup(3, {out_tags.back()}, -1, "Material_Clad");
  um2::gmsh::model::setColor({out_dim_tags.back()},
    clad.color.r(),
    clad.color.g(),
    clad.color.b(), 255, /*recursive=*/true);

  // Add the top and bottom of the dosimeter/holder (all clad)
  factory::addCylindricalPin(dosi_top, dosi_top_thick, dosi_cap_radii, dosi_cap_mats);
  factory::addCylindricalPin(dosi_bot, dosi_bot_thick, dosi_cap_radii, dosi_cap_mats);
  // Add the regular fuel pin sections above and below the dosimeter/holder
  factory::addCylindricalPin(fuel_top, fuel_sec_thick, umetal_radii, umetal_mats);
  factory::addCylindricalPin(fuel_bot, fuel_sec_thick, umetal_radii, umetal_mats);
  // Add the double clad section above and below the fuel
  factory::addCylindricalPin(double_clad_top, double_clad_thick, double_clad_radii, double_clad_mats);
  factory::addCylindricalPin(double_clad_bot, double_clad_thick, double_clad_radii, double_clad_mats);

  um2::gmsh::model::occ::synchronize();
  um2::gmsh::fltk::run();
  um2::gmsh::write("3d_NECTAR_OCRefl.brep", true);
  um2::finalize();
  return 0;
}

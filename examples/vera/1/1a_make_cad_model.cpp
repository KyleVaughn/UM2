// VERA Core Physics Benchmark Progression Problem Specifications
// Revision 4, August 29, 2014
// CASL-U-2012-0131-004

#include <um2.hpp>

auto main() -> int 
{
    um2::initialize();

    //// Parameters
    //double r_fuel = 0.4096; // Pellet radius = 0.4096 cm (pg. 4)
    //double r_gap = 0.418;   // Inner clad radius = 0.418 cm (pg. 4)
    //double r_clad = 0.475;  // Outer clad radius = 0.475 cm (pg.4)
    //std::vector<double> radii = {r_fuel, r_gap, r_clad};
    //double pitch = 1.26;    // Pitch = 1.26 cm (pg. 4)
    //double x = pitch / 2;   // x-coordinate of fuel pin center
    //double y = pitch / 2;   // y-coordinate of fuel pin center

    //std::vector<Material> materials = {
    //    Material("Fuel", "forestgreen"),
    //    Material("Gap", "white"),
    //    Material("Clad", "slategray"),
    //    Material("Water", "royalblue"),
    //};

    //gmsh::model::occ::add_2d_cylindrical_pin({x, y}, radii, materials);
//  //  gmsh::fltk::run();
    //gmsh::write("1a.brep", true);
    
    um2::finalize();
    return 0;
}

#include <um2.hpp>

#include <iostream> // std::cout, std::endl, std::cerr
#include <sstream> // std::stringstream
#include <stdexcept> // std::runtime_error

inline void getGlobalMeshParams(int argc, char **argv, um2::MeshType & mesh_type, double & lc)   
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <mesh_type> <lc>\n"
              << "  <mesh_type> is a one of the following: \n"
              << "    - 3 for triangular faces\n" 
              << "    - 4 for quadrilateral faces\n"
              << "    - 6 for quadratic triangular faces\n" 
              << "    - 8 for quadratic quadrilateral faces\n"
              << "  <lc> is the target global characteristic length in cm\n";
    exit(EXIT_FAILURE); 
  }

  std::stringstream ss(argv[1]);
  int mesh_int = 0;
  ss >> mesh_int;
  switch (mesh_int) {
    case 3:
      mesh_type = um2::MeshType::Tri;
      break;
    case 4:
      mesh_type = um2::MeshType::Quad;
      break;
    case 6:
      mesh_type = um2::MeshType::QuadraticTri;
      break;
    case 8:
      mesh_type = um2::MeshType::QuadraticQuad;
      break;
    default:
      std::cerr << "Error: <mesh_type> must be 3, 4, 6, or 8\n";
      exit(EXIT_FAILURE);
  }
  ss.clear();

  ss.str(argv[2]);
  ss >> lc;
  if (lc <= 0) {
    std::cerr << "Error: <lc> must be positive\n";
    exit(EXIT_FAILURE);
  }
  ss.clear();
}

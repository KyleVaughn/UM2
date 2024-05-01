#include <um2.hpp>

#include <iostream>

namespace um2
{

void
initialize()
{
  std::cout << "um2::initialize()" << std::endl;
}

void
finalize()
{
  std::cout << "um2::finalize()" << std::endl;
}

} // namespace um2

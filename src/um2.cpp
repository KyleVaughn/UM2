#include <um2.hpp>

#include <iostream>

namespace um2
{

void
initialize()
{
  std::cout << "um2::initialize()\n";
}

void
finalize()
{
  std::cout << "um2::finalize()\n";
}

} // namespace um2

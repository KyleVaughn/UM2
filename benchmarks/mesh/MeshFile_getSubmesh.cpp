//=============================================================================
// Findings
//=============================================================================
// Serial: 97 ms
// With TBB (parallel STL): 60 ms
// With OpenMP and TBB: 44 ms

#include "../helpers.hpp"
#include "fixtures.hpp"

#include <um2/mesh/io.hpp>

#include <iostream>

// cppcheck-suppress unknownMacro; justification: defined by benchmark
BENCHMARK_DEFINE_F(Lattice76790Fixture, getSubmesh)(benchmark::State & state)
{
  decltype(meshfile) submesh;
  // NOLINTNEXTLINE justification: Need to loop over the state variable
  for (auto s : state) {
    meshfile.getSubmesh("Material_Water", submesh);
  }
}

BENCHMARK_REGISTER_F(Lattice76790Fixture, getSubmesh)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

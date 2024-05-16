#include <um2/physics/cross_section.hpp>

#include <um2/common/logger.hpp>
//#include <um2/math/stats.hpp>
//#include <um2/stdlib/algorithm/max_element.hpp>

namespace um2
{

//==============================================================================
// Member functions
//==============================================================================

void
XSec::validate() const noexcept
{
  if (_num_groups <= 0) {
    LOG_ERROR("Cross section has a non-positive number of groups");
  }

  if (_a.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of absorption values");
  }

  if (_f.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of fission values");
  }

  if (_nuf.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of nu*fission values"); 
  }

  if (_tr.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of transport values");
  }

  if (_s.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of scattering values");
  }

  if (_ss.rows() != _num_groups || _ss.cols() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect scattering matrix size"); 
  }

  // If the cross section is macroscopic, the total XS must be positive. 
  // If the cross section is microscopic, the total XS can be negative due to
  // correction factors.
  if (isMacro()) {
    for (auto const & a : _a) {
      if (a < 0.0) {
        LOG_ERROR("Cross section has a negative absorption value"); 
      }
    }

    for (auto const & f : _f) {
      if (f < 0.0) {
        LOG_ERROR("Cross section has a negative fission value");
      }
    }

    for (auto const & nuf : _nuf) {
      if (nuf < 0.0) {
        LOG_ERROR("Cross section has a negative nu*fission value");
      }
    }

    for (auto const & tr : _tr) {
      if (tr < 0.0) {
        LOG_ERROR("Cross section has a negative transport value");
      }
    }

    for (auto const & s : _s) {
      if (s < 0.0) {
        LOG_ERROR("Cross section has a negative scattering value");
      }
    }
    for (Int i = 0; i < _num_groups * _num_groups; ++i) {
      if (_ss(i) < 0.0) {
        LOG_ERROR("Cross section has a negative scattering matrix value");
      }
    }
  }
}

//auto
//XSec::collapse(XSecReduction const strategy) const noexcept -> XSec
//{
//  XSec result;
//  result.isMacro() = isMacro();
//  ASSERT(!_t.empty());
//  if (strategy == XSecReduction::Max) {
//    result.t().push_back(*um2::max_element(_t.cbegin(), _t.cend()));
//  } else { // strategy == XSecReduction::Mean
//    result.t().push_back(um2::mean(_t.cbegin(), _t.cend()));
//  }
//  result.validate();
//  return result;
//}

} // namespace um2

#include <um2/physics/cross_section.hpp>

namespace um2
{

//==============================================================================
// Member functions
//==============================================================================

void
XSec::validate() const noexcept
{
  if (_t.empty()) {
    LOG_ERROR("Cross section has an empty total XS vector");
  }
  for (auto const & t_i : _t) {
    if (t_i < 0) {
      LOG_WARN("Cross section has a negative total XS in one or more groups");
    }
  }
}

auto
XSec::collapse(XSecReduction const strategy) const noexcept -> XSec
{
  XSec result;
  result.isMacro() = isMacro();
  ASSERT(!_t.empty());
  if (strategy == XSecReduction::Max) {
    result.t().push_back(*um2::max_element(_t.cbegin(), _t.cend()));
  } else { // strategy == XSecReduction::Mean
    result.t().push_back(um2::mean(_t.cbegin(), _t.cend()));
  }
  result.validate();
  return result;
}

} // namespace um2

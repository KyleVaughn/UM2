#include <um2/common/sort.hpp>

namespace um2
{

void
invertPermutation(Vector<Size> const & perm, Vector<Size> & inv_perm) noexcept
{
  ASSERT(perm.size() == inv_perm.size());
  for (Size i = 0; i < perm.size(); ++i) {
    inv_perm[perm[i]] = i;
  }
}

} // namespace um2

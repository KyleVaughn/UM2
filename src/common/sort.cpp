#include <um2/common/sort.hpp>

namespace um2
{

void
invertPermutation(Vector<I> const & perm, Vector<I> & inv_perm) noexcept
{
  ASSERT(perm.size() == inv_perm.size());
  for (I i = 0; i < perm.size(); ++i) {
    inv_perm[perm[i]] = i;
  }
}

} // namespace um2

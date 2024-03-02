#include <um2/geometry/morton_sort_points.hpp>

#include <um2/stdlib/numeric.hpp>

namespace um2
{

PURE HOSTDEV auto
mortonEncode(Point2 const & p) noexcept -> MortonCode
{
  return mortonEncode<MortonCode, Float>(p[0], p[1]);
}

PURE HOSTDEV auto
mortonEncode(Point3 const & p) noexcept -> MortonCode
{
  return mortonEncode<MortonCode, Float>(p[0], p[1], p[2]);
}

HOSTDEV void
mortonDecode(MortonCode const morton, Point2 & p) noexcept
{
  mortonDecode(morton, p[0], p[1]);
}

HOSTDEV void
mortonDecode(MortonCode const morton, Point3 & p) noexcept
{
  mortonDecode(morton, p[0], p[1], p[2]);
}

void
mortonSortPermutation(Point2 const * const begin, Point2 const * const end,
                      Int * const perm_begin, Point2 const & scale) noexcept
{
  auto const n = end - begin;
  um2::iota(perm_begin, perm_begin + n, 0);
  bool const has_scale = !isApprox<2>(scale, Point2::zero());
  if (has_scale) {
    std::sort(perm_begin, perm_begin + n, [&](Int const i, Int const j) {
      return mortonEncode(begin[i] * scale) < mortonEncode(begin[j] * scale);
    });
  } else {
    std::sort(perm_begin, perm_begin + n, [&](Int const i, Int const j) {
      return mortonEncode(begin[i]) < mortonEncode(begin[j]);
    });
  }
}

void
mortonSortPermutation(Point3 const * const begin, Point3 const * const end,
                      Int * const perm_begin, Point3 const & scale) noexcept
{
  auto const n = end - begin;
  um2::iota(perm_begin, perm_begin + n, 0);
  bool const has_scale = !isApprox<3>(scale, Point3::zero());
  if (has_scale) {
    std::sort(perm_begin, perm_begin + n, [&](Int const i, Int const j) {
      return mortonEncode(begin[i] * scale) < mortonEncode(begin[j] * scale);
    });
  } else {
    std::sort(perm_begin, perm_begin + n, [&](Int const i, Int const j) {
      return mortonEncode(begin[i]) < mortonEncode(begin[j]);
    });
  }
}

} // namespace um2

#include <um2/geometry/morton_sort_points.hpp>

#include <um2/stdlib/numeric.hpp>

namespace um2
{

PURE HOSTDEV auto
mortonEncode(Point2 const & p) noexcept -> MortonCode
{
  return mortonEncode<MortonCode, F>(p[0], p[1]);
}

PURE HOSTDEV auto
mortonEncode(Point3 const & p) noexcept -> MortonCode
{
  return mortonEncode<MortonCode, F>(p[0], p[1], p[2]);
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
mortonSortPermutation(
  Point2 const * const begin,
  Point2 const * const end,
  I * const perm_begin,
  Vec2<F> const & scale) noexcept
{
  auto const n = end - begin;
  um2::iota(perm_begin, perm_begin + n, 0);
  bool const has_scale = !isApprox(scale, Vec2<F>::zero());
  if (has_scale) {
    std::sort(perm_begin, perm_begin + n, 
    [&](I const i, I const j) {
      return mortonEncode(begin[i] * scale) < mortonEncode(begin[j] * scale);
    });
  } else {
    std::sort(perm_begin, perm_begin + n, 
    [&](I const i, I const j) {
      return mortonEncode(begin[i]) < mortonEncode(begin[j]);
    });
  }
} 

void
mortonSortPermutation(
  Point3 const * const begin,
  Point3 const * const end,
  I * const perm_begin,
  Vec3<F> const & scale) noexcept
{
  auto const n = end - begin;
  um2::iota(perm_begin, perm_begin + n, 0);
  bool const has_scale = !isApprox(scale, Vec3<F>::zero());
  if (has_scale) {
    std::sort(perm_begin, perm_begin + n, 
    [&](I const i, I const j) {
      return mortonEncode(begin[i] * scale) < mortonEncode(begin[j] * scale);
    });
  } else {
    std::sort(perm_begin, perm_begin + n, 
    [&](I const i, I const j) {
      return mortonEncode(begin[i]) < mortonEncode(begin[j]);
    });
  }
} 

} // namespace um2

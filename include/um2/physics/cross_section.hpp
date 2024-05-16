#pragma once

#include <um2/stdlib/vector.hpp>
#include <um2/math/matrix.hpp>

//======================================================================
// CROSS SECTION
//======================================================================
// A multi-group cross section that can represent macroscopic or
// microscopic cross sections.
//
// Units: [cm^-1] or [barns] depending on whether the cross section is
// macroscopic or microscopic.

namespace um2
{

enum class XSecReduction {
  Max,
  Mean,
};

class XSec
{

  bool _is_macroscopic = false;
  bool _is_fissile = false;
  Int _num_groups = 0;
  Vector<Float> _a; // Absorption cross section
  Vector<Float> _f; // Fission cross section
  Vector<Float> _nuf; // nu * fission cross section
  Vector<Float> _tr; // Transport cross section
  Vector<Float> _s; // Total scattering cross section.
                    // s(g) = sum_{g'=0}^{G-1} \sigma_{s}(g -> g')
                    // This is the sum of column(g) of the scattering matrix.
  Matrix<Float> _ss; // Scattering matrix _ss(i, j) = \sigma_{s}(j -> i) 
                     // A multiplication with a flux vector will give
                     // the scattering source. _ss * phi = q_s

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr XSec() noexcept = default;

  XSec(Int num_groups) noexcept
    : _num_groups(num_groups),
      _a(num_groups, 0.0),
      _f(num_groups, 0.0),
      _nuf(num_groups, 0.0),
      _tr(num_groups, 0.0),
      _s(num_groups, 0.0),
      _ss(num_groups, num_groups, 0.0)
  {
  }

  //======================================================================
  // Accessors
  //======================================================================

  PURE [[nodiscard]] constexpr auto
  isMacro() noexcept -> bool &;

  PURE [[nodiscard]] constexpr auto
  isMacro() const noexcept -> bool;

  PURE [[nodiscard]] constexpr auto
  numGroups() const noexcept -> Int;

  PURE [[nodiscard]] constexpr auto
  isFissile() noexcept -> bool &;

  PURE [[nodiscard]] constexpr auto
  isFissile() const noexcept -> bool;

  PURE [[nodiscard]] constexpr auto
  a() noexcept -> Vector<Float> &;

  PURE [[nodiscard]] constexpr auto
  a() const noexcept -> Vector<Float> const &;

  PURE [[nodiscard]] constexpr auto
  f() noexcept -> Vector<Float> &;

  PURE [[nodiscard]] constexpr auto
  f() const noexcept -> Vector<Float> const &;

  PURE [[nodiscard]] constexpr auto
  nuf() noexcept -> Vector<Float> &;

  PURE [[nodiscard]] constexpr auto
  nuf() const noexcept -> Vector<Float> const &;

  PURE [[nodiscard]] constexpr auto
  tr() noexcept -> Vector<Float> &;

  PURE [[nodiscard]] constexpr auto
  tr() const noexcept -> Vector<Float> const &;

  PURE [[nodiscard]] constexpr auto
  s() noexcept -> Vector<Float> &;

  PURE [[nodiscard]] constexpr auto
  s() const noexcept -> Vector<Float> const &;

  PURE [[nodiscard]] constexpr auto
  ss() noexcept -> Matrix<Float> &;

  PURE [[nodiscard]] constexpr auto
  ss() const noexcept -> Matrix<Float> const &;

  //======================================================================
  // Methods
  //======================================================================

  PURE [[nodiscard]] constexpr auto 
  t(Int g) const noexcept -> Float
  {
    ASSERT(0 <= g);
    ASSERT(g < _num_groups);
    return _a[g] + _s[g];
  }

  void
  validate() const noexcept;

//  // Get the 1-group cross section
//  PURE [[nodiscard]] auto
//  collapse(XSecReduction strategy = XSecReduction::Mean) const noexcept -> XSec;
}; // class XS

//======================================================================
// Accessors
//======================================================================

PURE [[nodiscard]] constexpr auto
XSec::isMacro() noexcept -> bool &
{
  return _is_macroscopic;
}

PURE [[nodiscard]] constexpr auto
XSec::isMacro() const noexcept -> bool
{
  return _is_macroscopic;
}

PURE [[nodiscard]] constexpr auto
XSec::numGroups() const noexcept -> Int
{
  return _num_groups;
}

PURE [[nodiscard]] constexpr auto
XSec::isFissile() noexcept -> bool &
{
  return _is_fissile;
}

PURE [[nodiscard]] constexpr auto
XSec::isFissile() const noexcept -> bool
{
  return _is_fissile;
}

PURE [[nodiscard]] constexpr auto
XSec::a() noexcept -> Vector<Float> &
{
  return _a;
}

PURE [[nodiscard]] constexpr auto
XSec::a() const noexcept -> Vector<Float> const &
{
  return _a;
}

PURE [[nodiscard]] constexpr auto
XSec::f() noexcept -> Vector<Float> &
{
  return _f;
}

PURE [[nodiscard]] constexpr auto
XSec::f() const noexcept -> Vector<Float> const &
{
  return _f;
}

PURE [[nodiscard]] constexpr auto
XSec::nuf() noexcept -> Vector<Float> &
{
  return _nuf;
}

PURE [[nodiscard]] constexpr auto
XSec::nuf() const noexcept -> Vector<Float> const &
{
  return _nuf;
}

PURE [[nodiscard]] constexpr auto
XSec::tr() noexcept -> Vector<Float> &
{
  return _tr;
}

PURE [[nodiscard]] constexpr auto
XSec::tr() const noexcept -> Vector<Float> const &
{
  return _tr;
}

PURE [[nodiscard]] constexpr auto
XSec::s() noexcept -> Vector<Float> &
{
  return _s;
}

PURE [[nodiscard]] constexpr auto
XSec::s() const noexcept -> Vector<Float> const &
{
  return _s;
}

PURE [[nodiscard]] constexpr auto
XSec::ss() noexcept -> Matrix<Float> &
{
  return _ss;
}

PURE [[nodiscard]] constexpr auto
XSec::ss() const noexcept -> Matrix<Float> const &
{
  return _ss;
}

} // namespace um2

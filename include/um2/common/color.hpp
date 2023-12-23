#pragma once

#include <um2/stdlib/string.hpp>
#include <um2/stdlib/utility.hpp>

namespace um2
{

//==============================================================================
// COLOR
//==============================================================================
// A 4 byte RGBA color.
// Little endian: 0xAABBGGRR

class Color
{
  uint8_t _r{};
  uint8_t _g{};
  uint8_t _b{};
  uint8_t _a{255};

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Color() noexcept = default;

  template <std::integral I>
  HOSTDEV constexpr Color(I r, I g, I b, I a = 255) noexcept;

  template <std::floating_point T>
  HOSTDEV constexpr Color(T r, T g, T b, T a = 1) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  r() const noexcept -> uint8_t;

  HOSTDEV [[nodiscard]] constexpr auto
  g() const noexcept -> uint8_t;

  HOSTDEV [[nodiscard]] constexpr auto
  b() const noexcept -> uint8_t;

  HOSTDEV [[nodiscard]] constexpr auto
  a() const noexcept -> uint8_t;

};

//==============================================================================
// Constructors
//==============================================================================

template <std::integral I>
HOSTDEV constexpr Color::Color(I r, I g, I b, I a) noexcept
: _r(static_cast<uint8_t>(r)),
  _g(static_cast<uint8_t>(g)),
  _b(static_cast<uint8_t>(b)),
  _a(static_cast<uint8_t>(a))
{
}

template <std::floating_point T>
HOSTDEV constexpr Color::Color(T r, T g, T b, T a) noexcept
: _r(static_cast<uint8_t>(r * 255)),
  _g(static_cast<uint8_t>(g * 255)),
  _b(static_cast<uint8_t>(b * 255)),
  _a(static_cast<uint8_t>(a * 255))
{
}

//==============================================================================
// Accessors
//==============================================================================

HOSTDEV constexpr auto
Color::r() const noexcept -> uint8_t
{
  return _r;
}

HOSTDEV constexpr auto
Color::g() const noexcept -> uint8_t
{
  return _g;
}

HOSTDEV constexpr auto
Color::b() const noexcept -> uint8_t
{
  return _b;
}

HOSTDEV constexpr auto
Color::a() const noexcept -> uint8_t
{
  return _a;
}

//==============================================================================
// Operators
//==============================================================================

CONST HOSTDEV constexpr auto
operator==(Color const lhs, Color const rhs) noexcept -> bool
{
  // Avoid short circuiting
  bool const same_r = lhs.r() == rhs.r();
  bool const same_g = lhs.g() == rhs.g();
  bool const same_b = lhs.b() == rhs.b();
  bool const same_a = lhs.a() == rhs.a();
  return same_r && same_g && same_b && same_a;
}

CONST HOSTDEV constexpr auto
operator!=(Color const lhs, Color const rhs) noexcept -> bool
{
  return !(lhs == rhs);
}

//=============================================================================
// Common colors
//=============================================================================
// All 147 SVG colors + 1 extra. If you can find the extra color you win a prize!

inline constexpr Color aliceblue{240, 248, 255, 255};
inline constexpr Color antiquewhite{250, 235, 215, 255};
inline constexpr Color aqua{0, 255, 255, 255};
inline constexpr Color aquamarine{127, 255, 212, 255};
inline constexpr Color azure{240, 255, 255, 255};
inline constexpr Color beige{245, 245, 220, 255};
inline constexpr Color bisque{255, 228, 196, 255};
inline constexpr Color black{0, 0, 0, 255};
inline constexpr Color blanchedalmond{255, 235, 205, 255};
inline constexpr Color blue{0, 0, 255, 255};
inline constexpr Color blueviolet{138, 43, 226, 255};
inline constexpr Color brown{165, 42, 42, 255};
inline constexpr Color burlywood{222, 184, 135, 255};
inline constexpr Color cadetblue{95, 158, 160, 255};
inline constexpr Color chartreuse{127, 255, 0, 255};
inline constexpr Color chocolate{210, 105, 30, 255};
inline constexpr Color coral{255, 127, 80, 255};
inline constexpr Color cornflowerblue{100, 149, 237, 255};
inline constexpr Color cornsilk{255, 248, 220, 255};
inline constexpr Color crimson{220, 20, 60, 255};
inline constexpr Color cyan{0, 255, 255, 255};
inline constexpr Color darkblue{0, 0, 139, 255};
inline constexpr Color darkcyan{0, 139, 139, 255};
inline constexpr Color darkgoldenrod{184, 134, 11, 255};
inline constexpr Color darkgray{169, 169, 169, 255};
inline constexpr Color darkgreen{0, 100, 0, 255};
inline constexpr Color darkgrey{169, 169, 169, 255};
inline constexpr Color darkkhaki{189, 183, 107, 255};
inline constexpr Color darkmagenta{139, 0, 139, 255};
inline constexpr Color darkolivegreen{85, 107, 47, 255};
inline constexpr Color darkorange{255, 140, 0, 255};
inline constexpr Color darkorchid{153, 50, 204, 255};
inline constexpr Color darkred{139, 0, 0, 255};
inline constexpr Color darksalmon{233, 150, 122, 255};
inline constexpr Color darkseagreen{143, 188, 143, 255};
inline constexpr Color darkslateblue{72, 61, 139, 255};
inline constexpr Color darkslategray{47, 79, 79, 255};
inline constexpr Color darkslategrey{47, 79, 79, 255};
inline constexpr Color darkturquoise{0, 206, 209, 255};
inline constexpr Color darkviolet{148, 0, 211, 255};
inline constexpr Color deeppink{255, 20, 147, 255};
inline constexpr Color deepskyblue{0, 191, 255, 255};
inline constexpr Color dimgray{105, 105, 105, 255};
inline constexpr Color dimgrey{105, 105, 105, 255};
inline constexpr Color dodgerblue{30, 144, 255, 255};
inline constexpr Color firebrick{178, 34, 34, 255};
inline constexpr Color floralwhite{255, 250, 240, 255};
inline constexpr Color forestgreen{34, 139, 34, 255};
inline constexpr Color fuchsia{255, 0, 255, 255};
inline constexpr Color gainsboro{220, 220, 220, 255};
inline constexpr Color ghostwhite{248, 248, 255, 255};
inline constexpr Color gold{255, 215, 0, 255};
inline constexpr Color goldenrod{218, 165, 32, 255};
inline constexpr Color gray{128, 128, 128, 255};
inline constexpr Color green{0, 128, 0, 255};
inline constexpr Color greenyellow{173, 255, 47, 255};
inline constexpr Color grey{128, 128, 128, 255};
inline constexpr Color honeydew{240, 255, 240, 255};
inline constexpr Color hotpink{255, 105, 180, 255};
inline constexpr Color indianred{205, 92, 92, 255};
inline constexpr Color indigo{75, 0, 130, 255};
inline constexpr Color ivory{255, 255, 240, 255};
inline constexpr Color khaki{240, 230, 140, 255};
inline constexpr Color lavender{230, 230, 250, 255};
inline constexpr Color lavenderblush{255, 240, 245, 255};
inline constexpr Color lawngreen{124, 252, 0, 255};
inline constexpr Color lemonchiffon{255, 250, 205, 255};
inline constexpr Color lightblue{173, 216, 230, 255};
inline constexpr Color lightcoral{240, 128, 128, 255};
inline constexpr Color lightcyan{224, 255, 255, 255};
inline constexpr Color lightgoldenrodyellow{250, 250, 210, 255};
inline constexpr Color lightgray{211, 211, 211, 255};
inline constexpr Color lightgreen{144, 238, 144, 255};
inline constexpr Color lightgrey{211, 211, 211, 255};
inline constexpr Color lightpink{255, 182, 193, 255};
inline constexpr Color lightsalmon{255, 160, 122, 255};
inline constexpr Color lightseagreen{32, 178, 170, 255};
inline constexpr Color lightskyblue{135, 206, 250, 255};
inline constexpr Color lightslateblue{132, 112, 255, 255};
inline constexpr Color lightslategray{119, 136, 153, 255};
inline constexpr Color lightslategrey{119, 136, 153, 255};
inline constexpr Color lightsteelblue{176, 196, 222, 255};
inline constexpr Color lightyellow{255, 255, 224, 255};
inline constexpr Color lime{0, 255, 0, 255};
inline constexpr Color limegreen{50, 205, 50, 255};
inline constexpr Color linen{250, 240, 230, 255};
inline constexpr Color magenta{255, 0, 255, 255};
inline constexpr Color maroon{128, 0, 0, 255};
inline constexpr Color mediumaquamarine{102, 205, 170, 255};
inline constexpr Color mediumblue{0, 0, 205, 255};
inline constexpr Color mediumorchid{186, 85, 211, 255};
inline constexpr Color mediumpurple{147, 112, 219, 255};
inline constexpr Color mediumseagreen{60, 179, 113, 255};
inline constexpr Color mediumslateblue{123, 104, 238, 255};
inline constexpr Color mediumspringgreen{0, 250, 154, 255};
inline constexpr Color mediumturquoise{72, 209, 204, 255};
inline constexpr Color mediumvioletred{199, 21, 133, 255};
inline constexpr Color midnightblue{25, 25, 112, 255};
inline constexpr Color mintcream{245, 255, 250, 255};
inline constexpr Color mistyrose{255, 228, 225, 255};
inline constexpr Color moccasin{255, 228, 181, 255};
inline constexpr Color navajowhite{255, 222, 173, 255};
inline constexpr Color navy{0, 0, 128, 255};
inline constexpr Color oldlace{253, 245, 230, 255};
inline constexpr Color olive{128, 128, 0, 255};
inline constexpr Color olivedrab{107, 142, 35, 255};
inline constexpr Color orange{255, 165, 0, 255};
inline constexpr Color orangered{255, 69, 0, 255};
inline constexpr Color orchid{218, 112, 214, 255};
inline constexpr Color palegoldenrod{238, 232, 170, 255};
inline constexpr Color palegreen{152, 251, 152, 255};
inline constexpr Color paleturquoise{175, 238, 238, 255};
inline constexpr Color palevioletred{219, 112, 147, 255};
inline constexpr Color papayawhip{255, 239, 213, 255};
inline constexpr Color peachpuff{255, 218, 185, 255};
inline constexpr Color peru{205, 133, 63, 255};
inline constexpr Color pink{255, 192, 203, 255};
inline constexpr Color plum{221, 160, 221, 255};
inline constexpr Color powderblue{176, 224, 230, 255};
inline constexpr Color purple{128, 0, 128, 255};
inline constexpr Color red{255, 0, 0, 255};
inline constexpr Color rosybrown{188, 143, 143, 255};
inline constexpr Color royalblue{65, 105, 225, 255};
inline constexpr Color saddlebrown{139, 69, 19, 255};
inline constexpr Color salmon{250, 128, 114, 255};
inline constexpr Color sandybrown{244, 164, 96, 255};
inline constexpr Color seagreen{46, 139, 87, 255};
inline constexpr Color seashell{255, 245, 238, 255};
inline constexpr Color sienna{160, 82, 45, 255};
inline constexpr Color silver{192, 192, 192, 255};
inline constexpr Color skyblue{135, 206, 235, 255};
inline constexpr Color slateblue{106, 90, 205, 255};
inline constexpr Color slategray{112, 128, 144, 255};
inline constexpr Color slategrey{112, 128, 144, 255};
inline constexpr Color snow{255, 250, 250, 255};
inline constexpr Color springgreen{0, 255, 127, 255};
inline constexpr Color steelblue{70, 130, 180, 255};
inline constexpr Color tan{210, 180, 140, 255};
inline constexpr Color teal{0, 128, 128, 255};
inline constexpr Color thistle{216, 191, 216, 255};
inline constexpr Color tomato{255, 99, 71, 255};
inline constexpr Color turquoise{64, 224, 208, 255};
inline constexpr Color violet{238, 130, 238, 255};
inline constexpr Color wheat{245, 222, 179, 255};
inline constexpr Color white{255, 255, 255, 255};
inline constexpr Color whitesmoke{245, 245, 245, 255};
inline constexpr Color yellow{255, 255, 0, 255};
inline constexpr Color yellowgreen{154, 205, 50, 255};

} // namespace um2

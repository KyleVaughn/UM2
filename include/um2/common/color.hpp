#pragma once

#include <um2/stdlib/string.hpp>

#include <concepts>

namespace um2
{

//==============================================================================
// COLOR
//==============================================================================
// A 4 byte RGBA color.
// Little endian: 0xAABBGGRR
// Common colors are defined in the .inl file

class Color
{

  struct RGBA {
    uint8_t r, g, b, a;
  };

  struct Rep {
    union {
      RGBA rgba;
      uint32_t u32; // 0xAABBGGRR
    };
  };

  Rep _rep;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  // Default to black (0, 0, 0, 255)
  HOSTDEV constexpr Color() noexcept;

  template <std::integral I>
  HOSTDEV constexpr Color(I r_in, I g_in, I b_in, I a_in = 255) noexcept;

  template <std::floating_point T>
  HOSTDEV constexpr Color(T r_in, T g_in, T b_in, T a_in = 1) noexcept;

  // We want to allow for implicit conversion for some cases
  // NOLINTBEGIN(google-explicit-constructor) justified
  HOSTDEV constexpr Color(String const & name) noexcept;
  // NOLINTEND(google-explicit-constructor)

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

  HOSTDEV [[nodiscard]] constexpr auto
  u32() const noexcept -> uint32_t
  {
    return _rep.u32;
  }
};

//==============================================================================
// Methods
//==============================================================================

// Need to forward declare this if we want to keep the same order as declared
// above.
PURE HOSTDEV constexpr auto
toColor(String const & name) noexcept -> Color;

//==============================================================================
// Constructors
//==============================================================================

HOSTDEV constexpr Color::Color() noexcept
{
  _rep.u32 = 0xFF000000; // 0xAAGGBBRR
}

template <std::integral I>
HOSTDEV constexpr Color::Color(I r_in, I g_in, I b_in, I a_in) noexcept
{
  _rep.rgba.r = static_cast<uint8_t>(r_in);
  _rep.rgba.g = static_cast<uint8_t>(g_in);
  _rep.rgba.b = static_cast<uint8_t>(b_in);
  _rep.rgba.a = static_cast<uint8_t>(a_in);
}

template <std::floating_point T>
HOSTDEV constexpr Color::Color(T r_in, T g_in, T b_in, T a_in) noexcept
{
  _rep.rgba.r = static_cast<uint8_t>(r_in * 255);
  _rep.rgba.g = static_cast<uint8_t>(g_in * 255);
  _rep.rgba.b = static_cast<uint8_t>(b_in * 255);
  _rep.rgba.a = static_cast<uint8_t>(a_in * 255);
}

HOSTDEV constexpr Color::Color(String const & name) noexcept
    : Color()
{
  *this = toColor(name);
}

//==============================================================================
// Accessors
//==============================================================================

HOSTDEV constexpr auto
Color::r() const noexcept -> uint8_t
{
  return _rep.rgba.r;
}

HOSTDEV constexpr auto
Color::g() const noexcept -> uint8_t
{
  return _rep.rgba.g;
}

HOSTDEV constexpr auto
Color::b() const noexcept -> uint8_t
{
  return _rep.rgba.b;
}

HOSTDEV constexpr auto
Color::a() const noexcept -> uint8_t
{
  return _rep.rgba.a;
}

//==============================================================================
// Operators
//==============================================================================

CONST HOSTDEV constexpr auto
operator==(Color const lhs, Color const rhs) noexcept -> bool
{
  return lhs.u32() == rhs.u32();
}

CONST HOSTDEV constexpr auto
operator!=(Color const lhs, Color const rhs) noexcept -> bool
{
  return !(lhs == rhs);
}

//==============================================================================
// Methods
//==============================================================================

PURE HOSTDEV constexpr auto
toColor(String const & name) noexcept -> Color
{

  struct NamedColor {
    String name;
    Color color;
  };

  // You can find the color swatches in:
  // http://juliagraphics.github.io/Colors.jl/dev/namedcolors/
  //
  // Colors ending in numbers have been removed, otherwise this list is
  // unnecessary long. If you really want another color, just add it to the list.
  uint32_t constexpr num_named_colors = 152;
  NamedColor const named_colors[num_named_colors] = {
      {           "aliceblue", {240, 248, 255, 255}},
      {        "antiquewhite", {250, 235, 215, 255}},
      {                "aqua",   {0, 255, 255, 255}},
      {          "aquamarine", {127, 255, 212, 255}},
      {               "azure", {240, 255, 255, 255}},
      {               "beige", {245, 245, 220, 255}},
      {              "bisque", {255, 228, 196, 255}},
      {               "black",       {0, 0, 0, 255}},
      {      "blanchedalmond", {255, 235, 205, 255}},
      {                "blue",     {0, 0, 255, 255}},
      {          "blueviolet",  {138, 43, 226, 255}},
      {               "brown",   {165, 42, 42, 255}},
      {           "burlywood", {222, 184, 135, 255}},
      {           "cadetblue",  {95, 158, 160, 255}},
      {          "chartreuse",   {127, 255, 0, 255}},
      {           "chocolate",  {210, 105, 30, 255}},
      {               "coral",  {255, 127, 80, 255}},
      {      "cornflowerblue", {100, 149, 237, 255}},
      {            "cornsilk", {255, 248, 220, 255}},
      {             "crimson",   {220, 20, 60, 255}},
      {                "cyan",   {0, 255, 255, 255}},
      {            "darkblue",     {0, 0, 139, 255}},
      {            "darkcyan",   {0, 139, 139, 255}},
      {       "darkgoldenrod",  {184, 134, 11, 255}},
      {            "darkgray", {169, 169, 169, 255}},
      {           "darkgreen",     {0, 100, 0, 255}},
      {            "darkgrey", {169, 169, 169, 255}},
      {           "darkkhaki", {189, 183, 107, 255}},
      {         "darkmagenta",   {139, 0, 139, 255}},
      {      "darkolivegreen",   {85, 107, 47, 255}},
      {          "darkorange",   {255, 140, 0, 255}},
      {          "darkorchid",  {153, 50, 204, 255}},
      {             "darkred",     {139, 0, 0, 255}},
      {          "darksalmon", {233, 150, 122, 255}},
      {        "darkseagreen", {143, 188, 143, 255}},
      {       "darkslateblue",   {72, 61, 139, 255}},
      {       "darkslategray",    {47, 79, 79, 255}},
      {       "darkslategrey",    {47, 79, 79, 255}},
      {       "darkturquoise",   {0, 206, 209, 255}},
      {          "darkviolet",   {148, 0, 211, 255}},
      {            "deeppink",  {255, 20, 147, 255}},
      {         "deepskyblue",   {0, 191, 255, 255}},
      {             "dimgray", {105, 105, 105, 255}},
      {             "dimgrey", {105, 105, 105, 255}},
      {          "dodgerblue",  {30, 144, 255, 255}},
      {           "firebrick",   {178, 34, 34, 255}},
      {         "floralwhite", {255, 250, 240, 255}},
      {         "forestgreen",   {34, 139, 34, 255}},
      {             "fuchsia",   {255, 0, 255, 255}},
      {           "gainsboro", {220, 220, 220, 255}},
      {          "ghostwhite", {248, 248, 255, 255}},
      {                "gold",   {255, 215, 0, 255}},
      {           "goldenrod",  {218, 165, 32, 255}},
      {                "gray", {128, 128, 128, 255}},
      {               "green",     {0, 128, 0, 255}},
      {         "greenyellow",  {173, 255, 47, 255}},
      {                "grey", {128, 128, 128, 255}},
      {            "honeydew", {240, 255, 240, 255}},
      {             "hotpink", {255, 105, 180, 255}},
      {           "indianred",   {205, 92, 92, 255}},
      {              "indigo",    {75, 0, 130, 255}},
      {               "ivory", {255, 255, 240, 255}},
      {               "khaki", {240, 230, 140, 255}},
      {            "lavender", {230, 230, 250, 255}},
      {       "lavenderblush", {255, 240, 245, 255}},
      {           "lawngreen",   {124, 252, 0, 255}},
      {        "lemonchiffon", {255, 250, 205, 255}},
      {           "lightblue", {173, 216, 230, 255}},
      {          "lightcoral", {240, 128, 128, 255}},
      {           "lightcyan", {224, 255, 255, 255}},
      {      "lightgoldenrod", {238, 221, 130, 255}},
      {"lightgoldenrodyellow", {250, 250, 210, 255}},
      {           "lightgray", {211, 211, 211, 255}},
      {          "lightgreen", {144, 238, 144, 255}},
      {           "lightgrey", {211, 211, 211, 255}},
      {           "lightpink", {255, 182, 193, 255}},
      {         "lightsalmon", {255, 160, 122, 255}},
      {       "lightseagreen",  {32, 178, 170, 255}},
      {        "lightskyblue", {135, 206, 250, 255}},
      {      "lightslateblue", {132, 112, 255, 255}},
      {      "lightslategray", {119, 136, 153, 255}},
      {      "lightslategrey", {119, 136, 153, 255}},
      {      "lightsteelblue", {176, 196, 222, 255}},
      {         "lightyellow", {255, 255, 224, 255}},
      {                "lime",     {0, 255, 0, 255}},
      {           "limegreen",   {50, 205, 50, 255}},
      {               "linen", {250, 240, 230, 255}},
      {             "magenta",   {255, 0, 255, 255}},
      {              "maroon",     {128, 0, 0, 255}},
      {    "mediumaquamarine", {102, 205, 170, 255}},
      {          "mediumblue",     {0, 0, 205, 255}},
      {        "mediumorchid",  {186, 85, 211, 255}},
      {        "mediumpurple", {147, 112, 219, 255}},
      {      "mediumseagreen",  {60, 179, 113, 255}},
      {     "mediumslateblue", {123, 104, 238, 255}},
      {   "mediumspringgreen",   {0, 250, 154, 255}},
      {     "mediumturquoise",  {72, 209, 204, 255}},
      {     "mediumvioletred",  {199, 21, 133, 255}},
      {        "midnightblue",   {25, 25, 112, 255}},
      {           "mintcream", {245, 255, 250, 255}},
      {           "mistyrose", {255, 228, 225, 255}},
      {            "moccasin", {255, 228, 181, 255}},
      {         "navajowhite", {255, 222, 173, 255}},
      {                "navy",     {0, 0, 128, 255}},
      {            "navyblue",     {0, 0, 128, 255}},
      {             "oldlace", {253, 245, 230, 255}},
      {               "olive",   {128, 128, 0, 255}},
      {           "olivedrab",  {107, 142, 35, 255}},
      {              "orange",   {255, 165, 0, 255}},
      {           "orangered",    {255, 69, 0, 255}},
      {              "orchid", {218, 112, 214, 255}},
      {       "palegoldenrod", {238, 232, 170, 255}},
      {           "palegreen", {152, 251, 152, 255}},
      {       "paleturquoise", {175, 238, 238, 255}},
      {       "palevioletred", {219, 112, 147, 255}},
      {          "papayawhip", {255, 239, 213, 255}},
      {           "peachpuff", {255, 218, 185, 255}},
      {                "peru",  {205, 133, 63, 255}},
      {                "pink", {255, 192, 203, 255}},
      {                "plum", {221, 160, 221, 255}},
      {          "powderblue", {176, 224, 230, 255}},
      {              "purple",   {128, 0, 128, 255}},
      {       "rebeccapurple",  {102, 51, 153, 255}},
      {                 "red",     {255, 0, 0, 255}},
      {           "rosybrown", {188, 143, 143, 255}},
      {           "royalblue",  {65, 105, 225, 255}},
      {         "saddlebrown",   {139, 69, 19, 255}},
      {              "salmon", {250, 128, 114, 255}},
      {          "sandybrown",  {244, 164, 96, 255}},
      {            "seagreen",   {46, 139, 87, 255}},
      {            "seashell", {255, 245, 238, 255}},
      {              "sienna",   {160, 82, 45, 255}},
      {              "silver", {192, 192, 192, 255}},
      {             "skyblue", {135, 206, 235, 255}},
      {           "slateblue",  {106, 90, 205, 255}},
      {           "slategray", {112, 128, 144, 255}},
      {           "slategrey", {112, 128, 144, 255}},
      {                "snow", {255, 250, 250, 255}},
      {         "springgreen",   {0, 255, 127, 255}},
      {           "steelblue",  {70, 130, 180, 255}},
      {                 "tan", {210, 180, 140, 255}},
      {                "teal",   {0, 128, 128, 255}},
      {             "thistle", {216, 191, 216, 255}},
      {              "tomato",   {255, 99, 71, 255}},
      {           "turquoise",  {64, 224, 208, 255}},
      {              "violet", {238, 130, 238, 255}},
      {           "violetred",  {208, 32, 144, 255}},
      {               "wheat", {245, 222, 179, 255}},
      {               "white", {255, 255, 255, 255}},
      {          "whitesmoke", {245, 245, 245, 255}},
      {              "yellow",   {255, 255, 0, 255}},
      {         "yellowgreen",  {154, 205, 50, 255}}
  };
  // Binary search
  NamedColor const * first = addressof(named_colors[0]);
  NamedColor const * last = first + num_named_colors; // 1 past the end
  auto len = last - first;
  ASSERT_ASSUME(0 <= len);
  ASSERT_ASSUME(len <= num_named_colors);
  while (0 < len) {
    auto half = len / 2;
    if (first[half].name < name) {
      first += len - half;
    }
    len = half;
  }
  return first->color;
}

//=============================================================================
// Common colors
//=============================================================================
// Would love to make this inline constexpr instead of a macro, but CUDA 12.2.1
// has trouble.

#define BLACK   um2::Color(0, 0, 0, 255)
#define WHITE   um2::Color(255, 255, 255, 255)
#define RED     um2::Color(255, 0, 0, 255)
#define GREEN   um2::Color(0, 255, 0, 255)
#define BLUE    um2::Color(0, 0, 255, 255)
#define YELLOW  um2::Color(255, 255, 0, 255)
#define CYAN    um2::Color(0, 255, 255, 255)
#define MAGENTA um2::Color(255, 0, 255, 255)

} // namespace um2

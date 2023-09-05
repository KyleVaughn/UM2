namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

HOSTDEV constexpr Color::Color() noexcept
{
  rep.u32 = 0xFF000000; // 0xAAGGBBRR
}

template <std::integral I>
HOSTDEV constexpr Color::Color(I r_in, I g_in, I b_in, I a_in) noexcept
{
  rep.rgba.r = static_cast<uint8_t>(r_in);
  rep.rgba.g = static_cast<uint8_t>(g_in);
  rep.rgba.b = static_cast<uint8_t>(b_in);
  rep.rgba.a = static_cast<uint8_t>(a_in);
}

template <std::floating_point T>
HOSTDEV constexpr Color::Color(T r_in, T g_in, T b_in, T a_in) noexcept
{
  rep.rgba.r = static_cast<uint8_t>(r_in * 255);
  rep.rgba.g = static_cast<uint8_t>(g_in * 255);
  rep.rgba.b = static_cast<uint8_t>(b_in * 255);
  rep.rgba.a = static_cast<uint8_t>(a_in * 255);
}

HOSTDEV constexpr Color::Color(Colors color) noexcept
{
  rep.u32 = static_cast<uint32_t>(color);
}

HOSTDEV constexpr Color::Color(ShortString const & name) noexcept
    : Color()
{
  *this = toColor(name);
}

template <size_t N>
HOSTDEV constexpr Color::Color(char const (&name)[N]) noexcept
    : Color(ShortString(name))
{
}

//==============================================================================
// Accessors
//==============================================================================

HOSTDEV constexpr auto
Color::r() const noexcept -> uint8_t
{
  return rep.rgba.r;
}

HOSTDEV constexpr auto
Color::g() const noexcept -> uint8_t
{
  return rep.rgba.g;
}

HOSTDEV constexpr auto
Color::b() const noexcept -> uint8_t
{
  return rep.rgba.b;
}

HOSTDEV constexpr auto
Color::a() const noexcept -> uint8_t
{
  return rep.rgba.a;
}

//==============================================================================
// Operators
//==============================================================================

HOSTDEV constexpr auto
Color::operator=(Colors color) noexcept -> Color &
{
  rep.u32 = static_cast<uint32_t>(color);
  return *this;
}

CONST HOSTDEV constexpr auto
operator==(Color const lhs, Color const rhs) noexcept -> bool
{
  return lhs.rep.u32 == rhs.rep.u32;
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
toColor(ShortString const & name) noexcept -> Color
{

  struct NamedColor {
    ShortString name;
    Color color;
  };

  // You can find the color swatches in:
  // http://juliagraphics.github.io/Colors.jl/dev/namedcolors/
  //
  // Colors ending in numbers have been removed, otherwise this list is
  // unnecessary long. If you really want another color, just add it to the list.
  uint32_t constexpr num_named_colors = 152;
  NamedColor const named_colors[num_named_colors] = {
      {           ShortString("aliceblue"), {240, 248, 255, 255}},
      {        ShortString("antiquewhite"), {250, 235, 215, 255}},
      {                ShortString("aqua"),   {0, 255, 255, 255}},
      {          ShortString("aquamarine"), {127, 255, 212, 255}},
      {               ShortString("azure"), {240, 255, 255, 255}},
      {               ShortString("beige"), {245, 245, 220, 255}},
      {              ShortString("bisque"), {255, 228, 196, 255}},
      {               ShortString("black"),       {0, 0, 0, 255}},
      {      ShortString("blanchedalmond"), {255, 235, 205, 255}},
      {                ShortString("blue"),     {0, 0, 255, 255}},
      {          ShortString("blueviolet"),  {138, 43, 226, 255}},
      {               ShortString("brown"),   {165, 42, 42, 255}},
      {           ShortString("burlywood"), {222, 184, 135, 255}},
      {           ShortString("cadetblue"),  {95, 158, 160, 255}},
      {          ShortString("chartreuse"),   {127, 255, 0, 255}},
      {           ShortString("chocolate"),  {210, 105, 30, 255}},
      {               ShortString("coral"),  {255, 127, 80, 255}},
      {      ShortString("cornflowerblue"), {100, 149, 237, 255}},
      {            ShortString("cornsilk"), {255, 248, 220, 255}},
      {             ShortString("crimson"),   {220, 20, 60, 255}},
      {                ShortString("cyan"),   {0, 255, 255, 255}},
      {            ShortString("darkblue"),     {0, 0, 139, 255}},
      {            ShortString("darkcyan"),   {0, 139, 139, 255}},
      {       ShortString("darkgoldenrod"),  {184, 134, 11, 255}},
      {            ShortString("darkgray"), {169, 169, 169, 255}},
      {           ShortString("darkgreen"),     {0, 100, 0, 255}},
      {            ShortString("darkgrey"), {169, 169, 169, 255}},
      {           ShortString("darkkhaki"), {189, 183, 107, 255}},
      {         ShortString("darkmagenta"),   {139, 0, 139, 255}},
      {      ShortString("darkolivegreen"),   {85, 107, 47, 255}},
      {          ShortString("darkorange"),   {255, 140, 0, 255}},
      {          ShortString("darkorchid"),  {153, 50, 204, 255}},
      {             ShortString("darkred"),     {139, 0, 0, 255}},
      {          ShortString("darksalmon"), {233, 150, 122, 255}},
      {        ShortString("darkseagreen"), {143, 188, 143, 255}},
      {       ShortString("darkslateblue"),   {72, 61, 139, 255}},
      {       ShortString("darkslategray"),    {47, 79, 79, 255}},
      {       ShortString("darkslategrey"),    {47, 79, 79, 255}},
      {       ShortString("darkturquoise"),   {0, 206, 209, 255}},
      {          ShortString("darkviolet"),   {148, 0, 211, 255}},
      {            ShortString("deeppink"),  {255, 20, 147, 255}},
      {         ShortString("deepskyblue"),   {0, 191, 255, 255}},
      {             ShortString("dimgray"), {105, 105, 105, 255}},
      {             ShortString("dimgrey"), {105, 105, 105, 255}},
      {          ShortString("dodgerblue"),  {30, 144, 255, 255}},
      {           ShortString("firebrick"),   {178, 34, 34, 255}},
      {         ShortString("floralwhite"), {255, 250, 240, 255}},
      {         ShortString("forestgreen"),   {34, 139, 34, 255}},
      {             ShortString("fuchsia"),   {255, 0, 255, 255}},
      {           ShortString("gainsboro"), {220, 220, 220, 255}},
      {          ShortString("ghostwhite"), {248, 248, 255, 255}},
      {                ShortString("gold"),   {255, 215, 0, 255}},
      {           ShortString("goldenrod"),  {218, 165, 32, 255}},
      {                ShortString("gray"), {128, 128, 128, 255}},
      {               ShortString("green"),     {0, 128, 0, 255}},
      {         ShortString("greenyellow"),  {173, 255, 47, 255}},
      {                ShortString("grey"), {128, 128, 128, 255}},
      {            ShortString("honeydew"), {240, 255, 240, 255}},
      {             ShortString("hotpink"), {255, 105, 180, 255}},
      {           ShortString("indianred"),   {205, 92, 92, 255}},
      {              ShortString("indigo"),    {75, 0, 130, 255}},
      {               ShortString("ivory"), {255, 255, 240, 255}},
      {               ShortString("khaki"), {240, 230, 140, 255}},
      {            ShortString("lavender"), {230, 230, 250, 255}},
      {       ShortString("lavenderblush"), {255, 240, 245, 255}},
      {           ShortString("lawngreen"),   {124, 252, 0, 255}},
      {        ShortString("lemonchiffon"), {255, 250, 205, 255}},
      {           ShortString("lightblue"), {173, 216, 230, 255}},
      {          ShortString("lightcoral"), {240, 128, 128, 255}},
      {           ShortString("lightcyan"), {224, 255, 255, 255}},
      {      ShortString("lightgoldenrod"), {238, 221, 130, 255}},
      {ShortString("lightgoldenrodyellow"), {250, 250, 210, 255}},
      {           ShortString("lightgray"), {211, 211, 211, 255}},
      {          ShortString("lightgreen"), {144, 238, 144, 255}},
      {           ShortString("lightgrey"), {211, 211, 211, 255}},
      {           ShortString("lightpink"), {255, 182, 193, 255}},
      {         ShortString("lightsalmon"), {255, 160, 122, 255}},
      {       ShortString("lightseagreen"),  {32, 178, 170, 255}},
      {        ShortString("lightskyblue"), {135, 206, 250, 255}},
      {      ShortString("lightslateblue"), {132, 112, 255, 255}},
      {      ShortString("lightslategray"), {119, 136, 153, 255}},
      {      ShortString("lightslategrey"), {119, 136, 153, 255}},
      {      ShortString("lightsteelblue"), {176, 196, 222, 255}},
      {         ShortString("lightyellow"), {255, 255, 224, 255}},
      {                ShortString("lime"),     {0, 255, 0, 255}},
      {           ShortString("limegreen"),   {50, 205, 50, 255}},
      {               ShortString("linen"), {250, 240, 230, 255}},
      {             ShortString("magenta"),   {255, 0, 255, 255}},
      {              ShortString("maroon"),     {128, 0, 0, 255}},
      {    ShortString("mediumaquamarine"), {102, 205, 170, 255}},
      {          ShortString("mediumblue"),     {0, 0, 205, 255}},
      {        ShortString("mediumorchid"),  {186, 85, 211, 255}},
      {        ShortString("mediumpurple"), {147, 112, 219, 255}},
      {      ShortString("mediumseagreen"),  {60, 179, 113, 255}},
      {     ShortString("mediumslateblue"), {123, 104, 238, 255}},
      {   ShortString("mediumspringgreen"),   {0, 250, 154, 255}},
      {     ShortString("mediumturquoise"),  {72, 209, 204, 255}},
      {     ShortString("mediumvioletred"),  {199, 21, 133, 255}},
      {        ShortString("midnightblue"),   {25, 25, 112, 255}},
      {           ShortString("mintcream"), {245, 255, 250, 255}},
      {           ShortString("mistyrose"), {255, 228, 225, 255}},
      {            ShortString("moccasin"), {255, 228, 181, 255}},
      {         ShortString("navajowhite"), {255, 222, 173, 255}},
      {                ShortString("navy"),     {0, 0, 128, 255}},
      {            ShortString("navyblue"),     {0, 0, 128, 255}},
      {             ShortString("oldlace"), {253, 245, 230, 255}},
      {               ShortString("olive"),   {128, 128, 0, 255}},
      {           ShortString("olivedrab"),  {107, 142, 35, 255}},
      {              ShortString("orange"),   {255, 165, 0, 255}},
      {           ShortString("orangered"),    {255, 69, 0, 255}},
      {              ShortString("orchid"), {218, 112, 214, 255}},
      {       ShortString("palegoldenrod"), {238, 232, 170, 255}},
      {           ShortString("palegreen"), {152, 251, 152, 255}},
      {       ShortString("paleturquoise"), {175, 238, 238, 255}},
      {       ShortString("palevioletred"), {219, 112, 147, 255}},
      {          ShortString("papayawhip"), {255, 239, 213, 255}},
      {           ShortString("peachpuff"), {255, 218, 185, 255}},
      {                ShortString("peru"),  {205, 133, 63, 255}},
      {                ShortString("pink"), {255, 192, 203, 255}},
      {                ShortString("plum"), {221, 160, 221, 255}},
      {          ShortString("powderblue"), {176, 224, 230, 255}},
      {              ShortString("purple"),   {128, 0, 128, 255}},
      {       ShortString("rebeccapurple"),  {102, 51, 153, 255}},
      {                 ShortString("red"),     {255, 0, 0, 255}},
      {           ShortString("rosybrown"), {188, 143, 143, 255}},
      {           ShortString("royalblue"),  {65, 105, 225, 255}},
      {         ShortString("saddlebrown"),   {139, 69, 19, 255}},
      {              ShortString("salmon"), {250, 128, 114, 255}},
      {          ShortString("sandybrown"),  {244, 164, 96, 255}},
      {            ShortString("seagreen"),   {46, 139, 87, 255}},
      {            ShortString("seashell"), {255, 245, 238, 255}},
      {              ShortString("sienna"),   {160, 82, 45, 255}},
      {              ShortString("silver"), {192, 192, 192, 255}},
      {             ShortString("skyblue"), {135, 206, 235, 255}},
      {           ShortString("slateblue"),  {106, 90, 205, 255}},
      {           ShortString("slategray"), {112, 128, 144, 255}},
      {           ShortString("slategrey"), {112, 128, 144, 255}},
      {                ShortString("snow"), {255, 250, 250, 255}},
      {         ShortString("springgreen"),   {0, 255, 127, 255}},
      {           ShortString("steelblue"),  {70, 130, 180, 255}},
      {                 ShortString("tan"), {210, 180, 140, 255}},
      {                ShortString("teal"),   {0, 128, 128, 255}},
      {             ShortString("thistle"), {216, 191, 216, 255}},
      {              ShortString("tomato"),   {255, 99, 71, 255}},
      {           ShortString("turquoise"),  {64, 224, 208, 255}},
      {              ShortString("violet"), {238, 130, 238, 255}},
      {           ShortString("violetred"),  {208, 32, 144, 255}},
      {               ShortString("wheat"), {245, 222, 179, 255}},
      {               ShortString("white"), {255, 255, 255, 255}},
      {          ShortString("whitesmoke"), {245, 245, 245, 255}},
      {              ShortString("yellow"),   {255, 255, 0, 255}},
      {         ShortString("yellowgreen"),  {154, 205, 50, 255}}
  };
  // Binary search
  NamedColor const * first = addressof(named_colors[0]);
  NamedColor const * last = first + num_named_colors; // 1 past the end
  auto len = last - first;
  while (0 < len) {
    auto half = len / 2;
    if (first[half].name < name) {
      first += len - half;
    }
    len = half;
  }
  return first->color;
}

} // namespace um2

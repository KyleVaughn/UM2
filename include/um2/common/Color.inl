namespace um2
{

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

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

HOSTDEV constexpr Color::Color(String const & name) noexcept
    : Color()
{
  *this = toColor(name);
}

template <size_t N>
HOSTDEV constexpr Color::Color(char const (&name)[N]) noexcept
    : Color(String(name))
{
}

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Operators
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

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
      {           String("aliceblue"), {240, 248, 255, 255}},
      {        String("antiquewhite"), {250, 235, 215, 255}},
      {                String("aqua"),   {0, 255, 255, 255}},
      {          String("aquamarine"), {127, 255, 212, 255}},
      {               String("azure"), {240, 255, 255, 255}},
      {               String("beige"), {245, 245, 220, 255}},
      {              String("bisque"), {255, 228, 196, 255}},
      {               String("black"),       {0, 0, 0, 255}},
      {      String("blanchedalmond"), {255, 235, 205, 255}},
      {                String("blue"),     {0, 0, 255, 255}},
      {          String("blueviolet"),  {138, 43, 226, 255}},
      {               String("brown"),   {165, 42, 42, 255}},
      {           String("burlywood"), {222, 184, 135, 255}},
      {           String("cadetblue"),  {95, 158, 160, 255}},
      {          String("chartreuse"),   {127, 255, 0, 255}},
      {           String("chocolate"),  {210, 105, 30, 255}},
      {               String("coral"),  {255, 127, 80, 255}},
      {      String("cornflowerblue"), {100, 149, 237, 255}},
      {            String("cornsilk"), {255, 248, 220, 255}},
      {             String("crimson"),   {220, 20, 60, 255}},
      {                String("cyan"),   {0, 255, 255, 255}},
      {            String("darkblue"),     {0, 0, 139, 255}},
      {            String("darkcyan"),   {0, 139, 139, 255}},
      {       String("darkgoldenrod"),  {184, 134, 11, 255}},
      {            String("darkgray"), {169, 169, 169, 255}},
      {           String("darkgreen"),     {0, 100, 0, 255}},
      {            String("darkgrey"), {169, 169, 169, 255}},
      {           String("darkkhaki"), {189, 183, 107, 255}},
      {         String("darkmagenta"),   {139, 0, 139, 255}},
      {      String("darkolivegreen"),   {85, 107, 47, 255}},
      {          String("darkorange"),   {255, 140, 0, 255}},
      {          String("darkorchid"),  {153, 50, 204, 255}},
      {             String("darkred"),     {139, 0, 0, 255}},
      {          String("darksalmon"), {233, 150, 122, 255}},
      {        String("darkseagreen"), {143, 188, 143, 255}},
      {       String("darkslateblue"),   {72, 61, 139, 255}},
      {       String("darkslategray"),    {47, 79, 79, 255}},
      {       String("darkslategrey"),    {47, 79, 79, 255}},
      {       String("darkturquoise"),   {0, 206, 209, 255}},
      {          String("darkviolet"),   {148, 0, 211, 255}},
      {            String("deeppink"),  {255, 20, 147, 255}},
      {         String("deepskyblue"),   {0, 191, 255, 255}},
      {             String("dimgray"), {105, 105, 105, 255}},
      {             String("dimgrey"), {105, 105, 105, 255}},
      {          String("dodgerblue"),  {30, 144, 255, 255}},
      {           String("firebrick"),   {178, 34, 34, 255}},
      {         String("floralwhite"), {255, 250, 240, 255}},
      {         String("forestgreen"),   {34, 139, 34, 255}},
      {             String("fuchsia"),   {255, 0, 255, 255}},
      {           String("gainsboro"), {220, 220, 220, 255}},
      {          String("ghostwhite"), {248, 248, 255, 255}},
      {                String("gold"),   {255, 215, 0, 255}},
      {           String("goldenrod"),  {218, 165, 32, 255}},
      {                String("gray"), {128, 128, 128, 255}},
      {               String("green"),     {0, 128, 0, 255}},
      {         String("greenyellow"),  {173, 255, 47, 255}},
      {                String("grey"), {128, 128, 128, 255}},
      {            String("honeydew"), {240, 255, 240, 255}},
      {             String("hotpink"), {255, 105, 180, 255}},
      {           String("indianred"),   {205, 92, 92, 255}},
      {              String("indigo"),    {75, 0, 130, 255}},
      {               String("ivory"), {255, 255, 240, 255}},
      {               String("khaki"), {240, 230, 140, 255}},
      {            String("lavender"), {230, 230, 250, 255}},
      {       String("lavenderblush"), {255, 240, 245, 255}},
      {           String("lawngreen"),   {124, 252, 0, 255}},
      {        String("lemonchiffon"), {255, 250, 205, 255}},
      {           String("lightblue"), {173, 216, 230, 255}},
      {          String("lightcoral"), {240, 128, 128, 255}},
      {           String("lightcyan"), {224, 255, 255, 255}},
      {      String("lightgoldenrod"), {238, 221, 130, 255}},
      {String("lightgoldenrodyellow"), {250, 250, 210, 255}},
      {           String("lightgray"), {211, 211, 211, 255}},
      {          String("lightgreen"), {144, 238, 144, 255}},
      {           String("lightgrey"), {211, 211, 211, 255}},
      {           String("lightpink"), {255, 182, 193, 255}},
      {         String("lightsalmon"), {255, 160, 122, 255}},
      {       String("lightseagreen"),  {32, 178, 170, 255}},
      {        String("lightskyblue"), {135, 206, 250, 255}},
      {      String("lightslateblue"), {132, 112, 255, 255}},
      {      String("lightslategray"), {119, 136, 153, 255}},
      {      String("lightslategrey"), {119, 136, 153, 255}},
      {      String("lightsteelblue"), {176, 196, 222, 255}},
      {         String("lightyellow"), {255, 255, 224, 255}},
      {                String("lime"),     {0, 255, 0, 255}},
      {           String("limegreen"),   {50, 205, 50, 255}},
      {               String("linen"), {250, 240, 230, 255}},
      {             String("magenta"),   {255, 0, 255, 255}},
      {              String("maroon"),     {128, 0, 0, 255}},
      {    String("mediumaquamarine"), {102, 205, 170, 255}},
      {          String("mediumblue"),     {0, 0, 205, 255}},
      {        String("mediumorchid"),  {186, 85, 211, 255}},
      {        String("mediumpurple"), {147, 112, 219, 255}},
      {      String("mediumseagreen"),  {60, 179, 113, 255}},
      {     String("mediumslateblue"), {123, 104, 238, 255}},
      {   String("mediumspringgreen"),   {0, 250, 154, 255}},
      {     String("mediumturquoise"),  {72, 209, 204, 255}},
      {     String("mediumvioletred"),  {199, 21, 133, 255}},
      {        String("midnightblue"),   {25, 25, 112, 255}},
      {           String("mintcream"), {245, 255, 250, 255}},
      {           String("mistyrose"), {255, 228, 225, 255}},
      {            String("moccasin"), {255, 228, 181, 255}},
      {         String("navajowhite"), {255, 222, 173, 255}},
      {                String("navy"),     {0, 0, 128, 255}},
      {            String("navyblue"),     {0, 0, 128, 255}},
      {             String("oldlace"), {253, 245, 230, 255}},
      {               String("olive"),   {128, 128, 0, 255}},
      {           String("olivedrab"),  {107, 142, 35, 255}},
      {              String("orange"),   {255, 165, 0, 255}},
      {           String("orangered"),    {255, 69, 0, 255}},
      {              String("orchid"), {218, 112, 214, 255}},
      {       String("palegoldenrod"), {238, 232, 170, 255}},
      {           String("palegreen"), {152, 251, 152, 255}},
      {       String("paleturquoise"), {175, 238, 238, 255}},
      {       String("palevioletred"), {219, 112, 147, 255}},
      {          String("papayawhip"), {255, 239, 213, 255}},
      {           String("peachpuff"), {255, 218, 185, 255}},
      {                String("peru"),  {205, 133, 63, 255}},
      {                String("pink"), {255, 192, 203, 255}},
      {                String("plum"), {221, 160, 221, 255}},
      {          String("powderblue"), {176, 224, 230, 255}},
      {              String("purple"),   {128, 0, 128, 255}},
      {       String("rebeccapurple"),  {102, 51, 153, 255}},
      {                 String("red"),     {255, 0, 0, 255}},
      {           String("rosybrown"), {188, 143, 143, 255}},
      {           String("royalblue"),  {65, 105, 225, 255}},
      {         String("saddlebrown"),   {139, 69, 19, 255}},
      {              String("salmon"), {250, 128, 114, 255}},
      {          String("sandybrown"),  {244, 164, 96, 255}},
      {            String("seagreen"),   {46, 139, 87, 255}},
      {            String("seashell"), {255, 245, 238, 255}},
      {              String("sienna"),   {160, 82, 45, 255}},
      {              String("silver"), {192, 192, 192, 255}},
      {             String("skyblue"), {135, 206, 235, 255}},
      {           String("slateblue"),  {106, 90, 205, 255}},
      {           String("slategray"), {112, 128, 144, 255}},
      {           String("slategrey"), {112, 128, 144, 255}},
      {                String("snow"), {255, 250, 250, 255}},
      {         String("springgreen"),   {0, 255, 127, 255}},
      {           String("steelblue"),  {70, 130, 180, 255}},
      {                 String("tan"), {210, 180, 140, 255}},
      {                String("teal"),   {0, 128, 128, 255}},
      {             String("thistle"), {216, 191, 216, 255}},
      {              String("tomato"),   {255, 99, 71, 255}},
      {           String("turquoise"),  {64, 224, 208, 255}},
      {              String("violet"), {238, 130, 238, 255}},
      {           String("violetred"),  {208, 32, 144, 255}},
      {               String("wheat"), {245, 222, 179, 255}},
      {               String("white"), {255, 255, 255, 255}},
      {          String("whitesmoke"), {245, 245, 245, 255}},
      {              String("yellow"),   {255, 255, 0, 255}},
      {         String("yellowgreen"),  {154, 205, 50, 255}}
  };
  // Binary search
  NamedColor const * first = addressof(named_colors[0]); 
  NamedColor const * last = first + num_named_colors; // 1 past the end
  auto length = last - first;
  while (0 < length) {
    auto half = length / 2;
    if (first[half].name < name) {
      first += length - half;
    }
    length = half;
  }
  return first->color;
}

} // namespace um2

namespace um2
{

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

HOSTDEV constexpr ShortString::ShortString() noexcept
{
  memset(data(), 0, sizeof(ShortString) - 1);
  _c[31] = static_cast<char>(31);
}

template <uint64_t N>
HOSTDEV constexpr ShortString::ShortString(char const (&s)[N]) noexcept
{
  static_assert(N - 1 <= capacity(), "String too long");
  copy(addressof(s[0]), addressof(s[N - 1]), addressof(_c[0]));
  _c[31] = capacity() - static_cast<char>(N - 1);
}

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------
PURE HOSTDEV constexpr auto
ShortString::size() const noexcept -> Size
{
  return capacity() - _c[31];
}

PURE HOSTDEV constexpr auto
ShortString::capacity() noexcept -> Size
{
  return sizeof(ShortString) - 1;
}

PURE HOSTDEV constexpr auto
ShortString::data() noexcept -> char *
{
  return addressof(_c[0]);
}

PURE HOSTDEV constexpr auto
ShortString::data() const noexcept -> char const *
{
  return addressof(_c[0]);
}

// --------------------------------------------------------------------------
// Operators
// --------------------------------------------------------------------------

HOSTDEV constexpr auto
ShortString::operator==(ShortString const & s) const noexcept -> bool
{
  Size const l_size = size();
  Size const r_size = s.size();
  if (l_size != r_size) {
    return false;
  }
  char const * l_data = data();
  char const * r_data = s.data();
  for (Size i = 0; i < l_size; ++i) {
    if (*l_data != *r_data) {
      return false;
    }
    ++l_data;
    ++r_data;
  }
  return true;
}

HOSTDEV constexpr auto
ShortString::operator!=(ShortString const & s) const noexcept -> bool
{
  return !(*this == s);
}

HOSTDEV constexpr auto
ShortString::operator<(ShortString const & s) const noexcept -> bool
{
  return compare(s) < 0;
}

HOSTDEV constexpr auto
ShortString::operator<=(ShortString const & s) const noexcept -> bool
{
  return compare(s) <= 0;
}

HOSTDEV constexpr auto
ShortString::operator>(ShortString const & s) const noexcept -> bool
{
  return compare(s) > 0;
}

HOSTDEV constexpr auto
ShortString::operator>=(ShortString const & s) const noexcept -> bool
{
  return compare(s) >= 0;
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

HOSTDEV constexpr auto
ShortString::compare(ShortString const & s) const noexcept -> int
{
  Size const l_size = size();
  Size const r_size = s.size();
  Size const min_size = um2::min(l_size, r_size);
  char const * l_data = data();
  char const * r_data = s.data();
  for (Size i = 0; i < min_size; ++i) {
    if (*l_data != *r_data) {
      return static_cast<int>(*l_data) - static_cast<int>(*r_data);
    }
    ++l_data;
    ++r_data;
  }
  return static_cast<int>(l_size) - static_cast<int>(r_size);
}

} // namespace um2

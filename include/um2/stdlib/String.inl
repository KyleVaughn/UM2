namespace um2
{

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

HOSTDEV constexpr String::String() noexcept
    : _r()
{
}

HOSTDEV constexpr String::String(String const & s) noexcept
{
  if (!s.isLong()) {
    // If this is a short string, it is trivially copyable
    _r.r = s._r.r;
  } else {
    _r.l.is_long = s._r.l.is_long;
    _r.l.cap = s._r.l.cap;
    _r.l.size = s._r.l.size;
    _r.l.data = static_cast<char *>(::operator new(s._r.l.cap + 1));
    memcpy(_r.l.data, s._r.l.data, s._r.l.size + 1);
  }
}

HOSTDEV constexpr String::String(String && s) noexcept
{
  // If short string, we can copy trivially
  // If long string, we need to move the data.
  // Since the data is a pointer, we can just copy the pointer.
  // Therefore, either way, we can just copy the whole struct.
  _r.r = s._r.r;
  s._r.l.is_long = 0;
  s._r.l.data = nullptr;
}

template <uint64_t N>
HOSTDEV constexpr String::String(char const (&s)[N]) noexcept
{
  // Short string
  if constexpr (N <= min_cap) {
    _r.s.is_long = 0;
    _r.s.size = N - 1;
    copy(addressof(s[0]), addressof(s[0]) + N, addressof(_r.s.data[0]));
    assert(_r.s.data[N - 1] == '\0');
  } else {
    _r.l.is_long = 1;
    _r.l.cap = N - 1;
    _r.l.size = N - 1;
    _r.l.data = static_cast<char *>(::operator new(N));
    copy(addressof(s[0]), addressof(s[0]) + N, _r.l.data);
    assert(_r.l.data[N - 1] == '\0');
  }
}

HOSTDEV constexpr String::String(char const * s) noexcept
{
  uint64_t n = 0;
  while (s[n] != '\0') {
    ++n;
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
  // Short string
  if (n + 1 <= min_cap) {
    _r.s.is_long = 0;
    _r.s.size = static_cast<uint8_t>(n);
    copy(s, s + (n + 1), addressof(_r.s.data[0]));
    assert(_r.s.data[n] == '\0');
  } else {
    _r.l.is_long = 1;
    _r.l.cap = n;
    _r.l.size = n;
    _r.l.data = static_cast<char *>(::operator new(n + 1));
    copy(s, s + (n + 1), _r.l.data);
    assert(_r.l.data[n] == '\0');
  }
#pragma GCC diagnostic pop
}

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

PURE HOSTDEV constexpr auto
String::isLong() const noexcept -> bool
{
  return this->_r.s.is_long;
}

PURE HOSTDEV constexpr auto
String::size() const noexcept -> uint64_t
{
  return isLong() ? getLongSize() : getShortSize();
}

// Allocated bytes - 1 for null terminator
PURE HOSTDEV constexpr auto
String::capacity() const noexcept -> uint64_t
{
  return isLong() ? getLongCap() : getShortCap();
}

PURE HOSTDEV constexpr auto
String::data() noexcept -> char *
{
  return getPointer();
}

PURE HOSTDEV constexpr auto
String::data() const noexcept -> char const *
{
  return getPointer();
}

// --------------------------------------------------------------------------
// Operators
// --------------------------------------------------------------------------

HOSTDEV constexpr auto
String::operator=(String const & s) noexcept -> String &
{
  if (this != addressof(s)) {
    if (isLong()) {
      ::operator delete(_r.l.data);
    }
    if (!s.isLong()) {
      // If this is a short string, it is trivially copyable
      _r.r = s._r.r;
    } else {
      _r.l.is_long = s._r.l.is_long;
      _r.l.cap = s._r.l.cap;
      _r.l.size = s._r.l.size;
      _r.l.data = static_cast<char *>(::operator new(s._r.l.cap + 1));
      memcpy(_r.l.data, s._r.l.data, s._r.l.size + 1);
    }
  }
  return *this;
}

HOSTDEV constexpr auto
String::operator=(String && s) noexcept -> String &
{
  if (this != addressof(s)) {
    if (isLong()) {
      ::operator delete(_r.l.data);
    }
    // If short string, we can copy trivially
    // If long string, we need to move the data.
    // Since the data is a pointer, we can just copy the pointer.
    // Therefore, either way, we can just copy the whole struct.
    _r.r = s._r.r;
    s._r.l.is_long = 0;
    s._r.l.data = nullptr;
  }
  return *this;
}

// These can be done better. If we use the same short string optimization,
// we should be able to do this more efficiently.
constexpr auto
String::operator=(std::string const & s) noexcept -> String &
{
  String tmp(s.c_str());
  return *this = um2::move(tmp);
}

constexpr auto
String::operator=(std::string && s) noexcept -> String &
{
  String tmp(s.c_str());
  return *this = um2::move(tmp);
}

template <uint64_t N>
constexpr auto
String::operator=(char const (&s)[N]) noexcept -> String &
{
  String tmp(s);
  return *this = um2::move(tmp);
}

PURE HOSTDEV constexpr auto
String::operator==(String const & s) const noexcept -> bool
{
  uint64_t const l_size = size();
  uint64_t const r_size = s.size();
  if (l_size != r_size) {
    return false;
  }
  char const * l_data = data();
  char const * r_data = s.data();
  for (uint64_t i = 0; i < l_size; ++i) {
    if (*l_data != *r_data) {
      return false;
    }
    ++l_data;
    ++r_data;
  }
  return true;
}

PURE HOSTDEV constexpr auto
String::operator!=(String const & s) const noexcept -> bool
{
  return !(*this == s);
}

PURE HOSTDEV constexpr auto
String::operator<(String const & s) const noexcept -> bool
{
  return compare(s) < 0;
}

PURE HOSTDEV constexpr auto
String::operator<=(String const & s) const noexcept -> bool
{
  return compare(s) <= 0;
}

PURE HOSTDEV constexpr auto
String::operator>(String const & s) const noexcept -> bool
{
  return compare(s) > 0;
}

PURE HOSTDEV constexpr auto
String::operator>=(String const & s) const noexcept -> bool
{
  return compare(s) >= 0;
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

PURE HOSTDEV constexpr auto
String::compare(String const & s) const noexcept -> int
{
  uint64_t const l_size = size();
  uint64_t const r_size = s.size();
  uint64_t const min_size = um2::min(l_size, r_size);
  char const * l_data = data();
  char const * r_data = s.data();
  for (uint64_t i = 0; i < min_size; ++i) {
    if (*l_data != *r_data) {
      return static_cast<int>(*l_data) - static_cast<int>(*r_data);
    }
    ++l_data;
    ++r_data;
  }
  return static_cast<int>(l_size) - static_cast<int>(r_size);
}

PURE HOSTDEV constexpr auto
String::c_str() const noexcept -> char const *
{
  return data();
}

PURE HOSTDEV constexpr auto
String::starts_with(String const & s) const noexcept -> bool
{
  if (size() < s.size()) {
    return false;
  }
  char const * l_data = data();
  char const * r_data = s.data();
  for (uint64_t i = 0; i < s.size(); ++i) {
    if (*l_data != *r_data) {
      return false;
    }
    ++l_data;
    ++r_data;
  }
  return true;
}

PURE HOSTDEV constexpr auto
String::ends_with(String const & s) const noexcept -> bool
{
  uint64_t const l_size = size();
  uint64_t const r_size = s.size();
  if (l_size < r_size) {
    return false;
  }
  char const * l_data = data() + l_size - r_size;
  char const * r_data = s.data();
  for (uint64_t i = 0; i < r_size; ++i) {
    if (*l_data != *r_data) {
      return false;
    }
    ++l_data;
    ++r_data;
  }
  return true;
}

template <uint64_t N>
PURE HOSTDEV auto
// NOLINTNEXTLINE(readability-identifier-naming)
String::ends_with(char const (&s)[N]) const noexcept -> bool
{
  return ends_with(String(s));
}

// --------------------------------------------------------------------------
// HIDDEN
// --------------------------------------------------------------------------

PURE HOSTDEV HIDDEN constexpr auto
String::getLongSize() const noexcept -> uint64_t
{
  return this->_r.l.size;
}

PURE HOSTDEV HIDDEN constexpr auto
String::getShortSize() const noexcept -> uint8_t
{
  return this->_r.s.size;
}

PURE HOSTDEV HIDDEN constexpr auto
String::getLongCap() const noexcept -> uint64_t
{
  return this->_r.l.cap;
}

PURE HOSTDEV HIDDEN constexpr auto
String::getShortCap() noexcept -> uint64_t
{
  return sizeof(Short::data) - 1;
}

PURE HOSTDEV HIDDEN constexpr auto
// NOLINTNEXTLINE(readability-make-member-function-const)
String::getLongPointer() noexcept -> char *
{
  return _r.l.data;
}

PURE HOSTDEV HIDDEN constexpr auto
String::getLongPointer() const noexcept -> char const *
{
  return _r.l.data;
}

PURE HOSTDEV HIDDEN constexpr auto
String::getShortPointer() noexcept -> char *
{
  return addressof(_r.s.data[0]);
}

PURE HOSTDEV HIDDEN constexpr auto
String::getShortPointer() const noexcept -> char const *
{
  return addressof(_r.s.data[0]);
}

PURE HOSTDEV HIDDEN constexpr auto
String::getPointer() noexcept -> char *
{
  return isLong() ? getLongPointer() : getShortPointer();
}

PURE HOSTDEV HIDDEN constexpr auto
String::getPointer() const noexcept -> char const *
{
  return isLong() ? getLongPointer() : getShortPointer();
}

// n includes null terminator
CONST HOSTDEV HIDDEN constexpr auto
String::fitsInShort(uint64_t n) noexcept -> bool
{
  return n <= min_cap;
}

} // namespace um2

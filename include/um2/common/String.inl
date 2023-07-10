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
    copy(addressof(s[0]), addressof(s[N - 1]), addressof(_r.s.data[0]));
  } else {
    _r.l.is_long = 1;
    _r.l.cap = N - 1;
    _r.l.size = N - 1;
    _r.l.data = static_cast<char *>(::operator new(N));
    copy(addressof(s[0]), addressof(s[N - 1]), _r.l.data);
  }
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

// template <size_t N>
// HOSTDEV auto
// String::operator=(char const (&s)[N]) -> String &
//{
//   if (this->_capacity < static_cast<len_t>(N)) {
//     delete[] this->_data;
//     this->_capacity = static_cast<len_t>(bit_ceil(N));
//     this->_data = new char[bit_ceil(N)];
//   }
//   this->_size = static_cast<len_t>(N - 1);
//   memcpy(this->_data, s, N);
//   return *this;
// }
//
// template <size_t N>
//// NOLINTNEXTLINE(*-avoid-c-arrays)
// PURE HOSTDEV constexpr auto
// String::operator==(char const (&s)[N]) const noexcept -> bool
//{
//   if (this->_size != static_cast<len_t>(N - 1)) {
//     return false;
//   }
//   for (len_t i = 0; i < this->_size; ++i) {
//     if (this->_data[i] != s[i]) {
//       return false;
//     }
//   }
//   return true;
// }
//
// PURE HOSTDEV constexpr auto
// String::operator==(String const & s) const noexcept -> bool
//{
//   if (this->_size != s._size) {
//     return false;
//   }
//   for (len_t i = 0; i < this->_size; ++i) {
//     if (this->_data[i] != s._data[i]) {
//       return false;
//     }
//   }
//   return true;
// }
//
// PURE HOSTDEV constexpr auto
// String::operator<(String const & s) const noexcept -> bool
//{
//   len_t const min_size = this->_size < s._size ? this->_size : s._size;
//   for (len_t i = 0; i < min_size; ++i) {
//     if (this->_data[i] != s._data[i]) {
//       return this->_data[i] < s._data[i];
//     }
//   }
//   return this->_size < s._size;
// }
//
// PURE HOSTDEV constexpr auto
// String::operator<=(String const & s) const noexcept -> bool
//{
//   len_t const min_size = this->_size < s._size ? this->_size : s._size;
//   for (len_t i = 0; i < min_size; ++i) {
//     if (this->_data[i] != s._data[i]) {
//       return this->_data[i] < s._data[i];
//     }
//   }
//   return this->_size <= s._size;
// }
//
// PURE HOSTDEV constexpr auto
// String::operator>(String const & s) const noexcept -> bool
//{
//   len_t const min_size = this->_size < s._size ? this->_size : s._size;
//   for (len_t i = 0; i < min_size; ++i) {
//     if (this->_data[i] != s._data[i]) {
//       return this->_data[i] > s._data[i];
//     }
//   }
//   return this->_size > s._size;
// }
//
// PURE HOSTDEV constexpr auto
// String::operator>=(String const & s) const noexcept -> bool
//{
//   len_t const min_size = this->_size < s._size ? this->_size : s._size;
//   for (len_t i = 0; i < min_size; ++i) {
//     if (this->_data[i] != s._data[i]) {
//       return this->_data[i] > s._data[i];
//     }
//   }
//   return this->_size >= s._size;
// }
//
//// --------------------------------------------------------------------------
//// Methods
//// --------------------------------------------------------------------------
//
// PURE HOSTDEV constexpr auto
// String::contains(char const c) const noexcept -> bool
//{
//  for (len_t i = 0; i < this->_size; ++i) {
//    if (this->_data[i] == c) {
//      return true;
//    }
//  }
//  return false;
//}
//
// PURE constexpr auto
// String::starts_with(std::string const & s) const noexcept -> bool
//{
//
//  if (this->_size < static_cast<len_t>(s.size())) {
//    return false;
//  }
//  char const * const sdata = s.data();
//  for (len_t i = 0; i < static_cast<len_t>(s.size()); ++i) {
//    if (this->_data[i] != sdata[i]) {
//      return false;
//    }
//  }
//  return true;
//}
//
// PURE constexpr auto
// String::ends_with(std::string const & s) const noexcept -> bool
//{
//  auto const ssize = static_cast<len_t>(s.size());
//  len_t const vsize = this->_size;
//  if (vsize < ssize) {
//    return false;
//  }
//  char const * const sdata = s.data();
//  for (len_t i = 0; i < ssize; ++i) {
//    if (this->_data[vsize - 1 - i] != sdata[ssize - 1 - i]) {
//      return false;
//    }
//  }
//  return true;
//}

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
// NOLINTNEXTLINE
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

//// n includes null terminator
// HOSTDEV HIDDEN constexpr void
// String::initShort(uint64_t n) noexcept
//{
//   _r.s.is_long = false;
//   _r.s.size = n - 1;
// }
//
//// n includes null terminator
// HOSTDEV HIDDEN constexpr void
// String::initLong(uint64_t n) noexcept
//{
//   _r.l.is_long = 1;
//   _r.l.cap = n - 1;
//   _r.l.size = n - 1;
//   _r.l.data = static_cast<char *>(::operator new(static_cast<uint64_t>(n)));
// }

} // namespace um2

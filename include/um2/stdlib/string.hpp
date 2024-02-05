#pragma once

#include <um2/stdlib/algorithm.hpp> // copy
#include <um2/stdlib/math.hpp>      // min
#include <um2/stdlib/memory.hpp>    // addressof
#include <um2/stdlib/utility.hpp>   // move

#include <bit>     // std::endian::native, std::endian::big
#include <cstring> // memcpy, strcmp
#include <string>  // std::string

//==============================================================================
// STRING
//==============================================================================
// An std::string-like class, but without an allocator template parameter.
// Uses small string optimization.
//
// NOTE: ASSUMES LITTLE ENDIAN
// This should be true for all x86 and Apple processors and both NVIDIA and AMD GPUs.
//
// For developers: clang-tidy is a bit overzealous with its warnings in this file.

namespace um2
{

static_assert(std::endian::native == std::endian::little,
              "Only little endian is supported.");

class String
{

  // Heap-allocated string representation.
  // 24 bytes
  struct Long {
    uint64_t is_long : 1; // Single bit for representation flag.
    uint64_t cap : 63;    // Capacity of the string. (Does not include null.)
    uint64_t size;        // Size of the string. (Does not include null.)
    char * data;          // Pointer to the string data.
  };

  // Mask to convert to a 63-bit unsigned integer.
  static uint64_t constexpr long_cap_mask = 0x7FFFFFFFFFFFFFFF;

  // The maximum capacity of a short string.
  // 24 bytes - 1 byte = 23 bytes
  static uint64_t constexpr min_cap = sizeof(Long) - 1;

  // Stack-allocated string representation.
  struct Short {
    uint8_t is_long : 1; // Single bit for representation flag.
    uint8_t size : 7;    // 7 bits for the size of the string. (Does not include null.)
    char data[min_cap];  // Data of the string.
  };

  // Mask to convert to a 7-bit unsigned integer.
  static uint8_t constexpr short_size_mask = 0x7F;

  // Raw representation of the string.
  // For the purpose of copying and moving.
  struct Raw {
    uint64_t raw[3];
  };

  // Union of all representations.
  struct Rep {
    union {
      Long l;
      Short s;
      Raw r;
    };
  };

  Rep _r;

  //==============================================================================
  // Private methods
  //==============================================================================
  // NOLINTBEGIN(readability-identifier-naming) match std::string

  // Does a string of length n fit in a short string? n includes the null terminator.
  CONST HOSTDEV static constexpr auto
  fitsInShort(uint64_t n) noexcept -> bool;

  // Get the capacity of the long string. Does not include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongCap() const noexcept -> uint64_t;

  // Get a pointer to the long string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongPointer() noexcept -> char *;

  // Get a pointer to the long string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongPointer() const noexcept -> char const *;

  // Get the size of the long string. Does not include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongSize() const noexcept -> uint64_t;

  // Get a pointer to the string data regardless of representation.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPointer() noexcept -> char *;

  // Get a pointer to the string data regardless of representation.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPointer() const noexcept -> char const *;

  // Get the capacity of the short string. Does not include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr static auto
  getShortCap() noexcept -> uint64_t;

  // Get a pointer to the short string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortPointer() noexcept -> char *;

  // Get a pointer to the short string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortPointer() const noexcept -> char const *;

  // Get the size of the short string. Does not include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortSize() const noexcept -> uint8_t;

public:
  // The maximum capacity of a long string.
  static I constexpr npos = sizeMax();

  //==============================================================================
  // Constructors
  //==============================================================================

  HOSTDEV constexpr String() noexcept;

  HOSTDEV constexpr String(String const & s) noexcept;

  HOSTDEV constexpr String(String && s) noexcept;

  // NOLINTBEGIN(google-explicit-constructor) match std::string
  template <uint64_t N>
  HOSTDEV constexpr String(char const (&s)[N]) noexcept;
  // NOLINTEND(google-explicit-constructor)

  HOSTDEV constexpr explicit String(char const * s) noexcept;

  HOSTDEV constexpr String(char const * begin, char const * end) noexcept;

  HOSTDEV constexpr String(char const * s, I n) noexcept;

  template <std::integral T>
  explicit constexpr String(T x) noexcept;

  template <std::floating_point T>
  explicit constexpr String(T x) noexcept;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV constexpr ~String() noexcept
  {
    // Clang can't figure out that isLong() == true implies that data is initialized.
#ifndef __clang__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    if (isLong()) {
      ::operator delete(_r.l.data);
    }
#ifndef __clang__
#  pragma GCC diagnostic pop
#endif
  }

  //==============================================================================
  // Accessors
  //==============================================================================

  // Not including the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> I;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> char *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> char const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLong() const noexcept -> bool;

  // Not including the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> I;

  //==============================================================================
  // Operators
  //==============================================================================

  HOSTDEV constexpr auto
  operator=(String const & s) noexcept -> String &;

  HOSTDEV constexpr auto
  operator=(String && s) noexcept -> String &;

  constexpr auto
  operator=(std::string const & s) noexcept -> String &;

  constexpr auto
  operator=(std::string && s) noexcept -> String &;

  template <uint64_t N>
  HOSTDEV constexpr auto
  operator=(char const (&s)[N]) noexcept -> String &;

  PURE HOSTDEV constexpr auto
  operator==(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator!=(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator<(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator<=(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator>(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator>=(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator[](I i) noexcept -> char &;

  PURE HOSTDEV constexpr auto
  operator[](I i) const noexcept -> char const &;

  HOSTDEV constexpr auto
  operator+=(String const & s) noexcept -> String &;

  HOSTDEV constexpr auto
  operator+=(char c) noexcept -> String &;

  //==============================================================================
  // Methods
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  compare(String const & s) const noexcept -> int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  c_str() const noexcept -> char const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  ends_with(String const & s) const noexcept -> bool;

  template <uint64_t N>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  ends_with(char const (&s)[N]) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  find_last_of(char c) const noexcept -> I;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  starts_with(String const & s) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  substr(I pos, I len = npos) const -> String;

  // NOLINTEND(readability-identifier-naming)
}; // class String

//==============================================================================
// Non-member operators
//==============================================================================

HOSTDEV constexpr auto
operator+(String l, String const & r) noexcept -> String;

template <uint64_t N>
HOSTDEV constexpr auto
operator+(String l, char const (&r)[N]) noexcept -> String;

template <uint64_t N>
HOSTDEV constexpr auto
operator+(char const (&l)[N], String const & r) noexcept -> String;

//==============================================================================
// Non-member methods
//==============================================================================

template <typename T>
constexpr auto
toString(T const & t) noexcept -> String;

//==============================================================================
// MACROS
//==============================================================================
// To maintain constexpr-ness and readability, we define a few macros to avoid
// repeating ourselves
//
// gcc seems to have a bug that causes it to generate a call to memmove that
// is out of bounds when using std::copy with -O3. Therefore, we write out a basic
// copy loop instead.

// mimic allocateAndCopy(char const *, uint64_t)
#define LONG_ALLOCATE_AND_COPY(ptr, num_elem)                                            \
  char const * ss = (ptr);                                                               \
  uint64_t const nn = (num_elem);                                                        \
  ASSERT_ASSUME(nn > 0);                                                                 \
  _r.l.is_long = 1;                                                                      \
  _r.l.cap = (nn - 1) & long_cap_mask;                                                   \
  _r.l.size = nn - 1;                                                                    \
  _r.l.data = static_cast<char *>(::operator new(nn));                                   \
  auto first = ss;                                                                       \
  auto last = ss + nn;                                                                   \
  auto dest = _r.l.data;                                                                 \
  while (first != last) {                                                                \
    *dest = *first;                                                                      \
    ++first;                                                                             \
    ++dest;                                                                              \
  }

#define SHORT_COPY(ptr, num_elem)                                                        \
  char const * ss = (ptr);                                                               \
  uint64_t const nn = (num_elem);                                                        \
  ASSERT_ASSUME(nn > 0);                                                                 \
  _r.s.is_long = 0;                                                                      \
  _r.s.size = (nn - 1) & short_size_mask;                                                \
  auto first = ss;                                                                       \
  auto last = ss + nn;                                                                   \
  auto dest = addressof(_r.s.data[0]);                                                   \
  while (first != last) {                                                                \
    *dest = *first;                                                                      \
    ++first;                                                                             \
    ++dest;                                                                              \
  }

//==============================================================================
// Private methods
//==============================================================================

PURE HOSTDEV constexpr auto
String::getLongSize() const noexcept -> uint64_t
{
  return this->_r.l.size;
}

PURE HOSTDEV constexpr auto
String::getShortSize() const noexcept -> uint8_t
{
  return this->_r.s.size;
}

PURE HOSTDEV constexpr auto
String::getLongCap() const noexcept -> uint64_t
{
  return this->_r.l.cap;
}

PURE HOSTDEV constexpr auto
String::getShortCap() noexcept -> uint64_t
{
  return sizeof(Short::data) - 1;
}

PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-make-member-function-const) we offer const next
String::getLongPointer() noexcept -> char *
{
  return _r.l.data;
}

PURE HOSTDEV constexpr auto
String::getLongPointer() const noexcept -> char const *
{
  return _r.l.data;
}

PURE HOSTDEV constexpr auto
String::getShortPointer() noexcept -> char *
{
  return addressof(_r.s.data[0]);
}

PURE HOSTDEV constexpr auto
String::getShortPointer() const noexcept -> char const *
{
  return addressof(_r.s.data[0]);
}

PURE HOSTDEV constexpr auto
String::getPointer() noexcept -> char *
{
  return isLong() ? getLongPointer() : getShortPointer();
}

PURE HOSTDEV constexpr auto
String::getPointer() const noexcept -> char const *
{
  return isLong() ? getLongPointer() : getShortPointer();
}

// n includes null terminator
CONST HOSTDEV constexpr auto
String::fitsInShort(uint64_t n) noexcept -> bool
{
  return n <= min_cap;
}

template <typename T>
constexpr auto
toString(T const & t) noexcept -> String
{
  return String(t);
}

HOSTDEV constexpr auto
operator+(String l, String const & r) noexcept -> String
{
  l += r;
  return l;
}

template <uint64_t N>
HOSTDEV constexpr auto
operator+(String l, char const (&r)[N]) noexcept -> String
{
  l += String(r);
  return l;
}

template <uint64_t N>
HOSTDEV constexpr auto
operator+(char const (&l)[N], String const & r) noexcept -> String
{
  String tmp(l);
  tmp += r;
  return tmp;
}

//==============================================================================
// Constructors
//==============================================================================

HOSTDEV constexpr String::String() noexcept
    : _r()
{
  // zero-initialize (short string)
  _r.r.raw[0] = 0;
  _r.r.raw[1] = 0;
  _r.r.raw[2] = 0;
}

HOSTDEV constexpr String::String(String const & s) noexcept
{
  if (!s.isLong()) {
    // If this is a short string, it is trivially copyable
    _r.r = s._r.r;
  } else {
    LONG_ALLOCATE_AND_COPY(s._r.l.data, s._r.l.size + 1);
    ASSERT(_r.l.data[_r.l.size] == '\0');
  }
}

HOSTDEV constexpr String::String(String && s) noexcept
    : _r(um2::move(s._r))
{
  // If short string, we can copy trivially
  // If long string, we need to move the data.
  // Since the data is a pointer, we can just copy the pointer.
  // Therefore, either way, we can just copy the whole struct.
  s._r = Rep();
}

template <uint64_t N>
HOSTDEV constexpr String::String(char const (&s)[N]) noexcept
{
  // N includes the null terminator
  char const * p = addressof(s[0]);
  if constexpr (N <= min_cap) {
    SHORT_COPY(p, N);
    ASSERT(_r.s.data[N - 1] == '\0');
  } else {
    LONG_ALLOCATE_AND_COPY(p, N);
    ASSERT(_r.l.data[N - 1] == '\0');
  }
}

HOSTDEV constexpr String::String(char const * s) noexcept
{
  // Get the length of the string (not including null terminator)
  uint64_t n = 0;
  while (s[n] != '\0') {
    ++n;
  }
  ASSERT(n > 0);
  if (n + 1 <= min_cap) {
    SHORT_COPY(s, n + 1);
    ASSERT(_r.s.data[n] == '\0');
  } else {
    LONG_ALLOCATE_AND_COPY(s, n + 1);
    ASSERT(_r.l.data[n] == '\0');
  }
}

HOSTDEV constexpr String::String(char const * s, I const n) noexcept
{
  // Short string
  auto const cap = static_cast<uint64_t>(n);
  if (cap + 1 <= min_cap) {
    SHORT_COPY(s, cap + 1);
    _r.s.data[cap] = '\0';
  } else {
    LONG_ALLOCATE_AND_COPY(s, cap + 1);
    _r.l.data[cap] = '\0';
  }
}

HOSTDEV constexpr String::String(char const * begin, char const * end) noexcept
{
  // Call the constructor that takes a pointer and a length
  String tmp(begin, static_cast<I>(end - begin));
  *this = um2::move(tmp);
}

// std::to_string should not allocate here due to small string optimization, so
// we are fine keeping these for now. They should be replaced with something
// like snprintf in the future.
template <std::integral T>
constexpr String::String(T x) noexcept
{
  // A 64-bit integer can have at most 20 chars
  std::string const s = std::to_string(x);
  auto const cap = s.size();
  ASSERT_ASSUME(cap < min_cap);
  SHORT_COPY(s.data(), cap + 1);
  _r.s.data[cap] = '\0';
}

template <std::floating_point T>
constexpr String::String(T x) noexcept
{
  // A 64-bit floating point number can have at most 24 chars, but
  // many numbers will have fewer than 24 chars. So, we ASSERT that
  // this is the case.
  std::string const s = std::to_string(x);
  auto const cap = s.size();
  ASSERT_ASSUME(cap < min_cap);
  SHORT_COPY(s.data(), cap + 1);
  _r.s.data[cap] = '\0';
}

//==============================================================================
// Accessors
//==============================================================================

PURE HOSTDEV constexpr auto
String::isLong() const noexcept -> bool
{
  return this->_r.l.is_long;
}

PURE HOSTDEV constexpr auto
String::size() const noexcept -> I
{
  return isLong() ? static_cast<I>(getLongSize()) : static_cast<I>(getShortSize());
}

// Allocated bytes - 1 for null terminator
PURE HOSTDEV constexpr auto
String::capacity() const noexcept -> I
{
  return isLong() ? static_cast<I>(getLongCap()) : static_cast<I>(getShortCap());
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

//==============================================================================
// Operators
//==============================================================================

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
      LONG_ALLOCATE_AND_COPY(s._r.l.data, s._r.l.size + 1);
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
    _r = um2::move(s._r);
    // delete the data when it goes out of scope.
    s._r = Rep();
  }
  return *this;
}

// These std::string assignment operators are a bit inefficient, but the number of
// heap allocations is the same as if we had just copied the string, so it's not
// too bad.
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
HOSTDEV constexpr auto
String::operator=(char const (&s)[N]) noexcept -> String &
{
  if (isLong()) {
    ::operator delete(_r.l.data);
  }
  // Short string
  if constexpr (N <= min_cap) {
    SHORT_COPY(s, N);
    ASSERT(_r.s.data[N - 1] == '\0');
  } else {
    LONG_ALLOCATE_AND_COPY(s, N);
    ASSERT(_r.l.data[N - 1] == '\0');
  }
  return *this;
}

PURE HOSTDEV constexpr auto
String::operator==(String const & s) const noexcept -> bool
{
  I const l_size = size();
  I const r_size = s.size();
  if (l_size != r_size) {
    return false;
  }
  char const * l_data = data();
  char const * r_data = s.data();
  for (I i = 0; i < l_size; ++i) {
    // NOLINTNEXTLINE
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

PURE HOSTDEV constexpr auto
String::operator[](I i) noexcept -> char &
{
  return data()[i];
}

PURE HOSTDEV constexpr auto
String::operator[](I i) const noexcept -> char const &
{
  return data()[i];
}

HOSTDEV constexpr auto
String::operator+=(String const & s) noexcept -> String &
{
  auto const new_size = static_cast<uint64_t>(size() + s.size());
  if (fitsInShort(new_size + 1)) {
    // This must be a short string, so we can just copy the data.
    ASSERT(!isLong());
    ASSERT(!s.isLong());
    ASSERT(new_size < min_cap);
    memcpy(getPointer() + size(), s.data(), static_cast<uint64_t>(s.size() + 1));
    _r.s.size = new_size & short_size_mask;
  } else {
    // Otherwise, we need to allocate a new string and copy the data.
    char * tmp = static_cast<char *>(::operator new(new_size + 1));
    memcpy(tmp, data(), static_cast<uint64_t>(size()));
    memcpy(tmp + size(), s.data(), static_cast<uint64_t>(s.size() + 1));
    if (isLong()) {
      ::operator delete(_r.l.data);
    }
    _r.l.is_long = 1;
    _r.l.cap = (new_size + 1) & long_cap_mask;
    _r.l.size = new_size;
    _r.l.data = tmp;
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) Valgrind says this is fine
  return *this;
}

HOSTDEV constexpr auto
String::operator+=(char const c) noexcept -> String &
{
  // If this is a short string and the size of the new string is less than
  // the capacity of the short string, we can just append the new string.
  auto const new_size = static_cast<uint64_t>(size() + 1);
  if (fitsInShort(new_size + 1)) {
    ASSERT(!isLong());
    _r.s.data[size()] = c;
    _r.s.data[size() + 1] = '\0';
    _r.s.size += 1;
  } else {
    // Otherwise, we need to allocate a new string and copy the data.
    char * tmp = static_cast<char *>(::operator new(new_size + 1));
    memcpy(tmp, data(), static_cast<uint64_t>(size()));
    tmp[size()] = c;
    tmp[size() + 1] = '\0';
    if (isLong()) {
      ::operator delete(_r.l.data);
    }
    _r.l.is_long = 1;
    _r.l.cap = (new_size + 1) & long_cap_mask;
    _r.l.size = new_size;
    _r.l.data = tmp;
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) Valgrind says this is fine
  return *this;
}

//==============================================================================
// Methods
//==============================================================================

PURE HOSTDEV constexpr auto
String::compare(String const & s) const noexcept -> int
{
  I const l_size = size();
  I const r_size = s.size();
  I const min_size = um2::min(l_size, r_size);
  char const * l_data = data();
  char const * r_data = s.data();
  for (I i = 0; i < min_size; ++i) {
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
  for (I i = 0; i < s.size(); ++i) {
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
  I const l_size = size();
  I const r_size = s.size();
  if (l_size < r_size) {
    return false;
  }
  char const * l_data = data() + l_size - r_size;
  char const * r_data = s.data();
  for (I i = 0; i < r_size; ++i) {
    if (*l_data != *r_data) {
      return false;
    }
    ++l_data;
    ++r_data;
  }
  return true;
}

template <uint64_t N>
PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::string
String::ends_with(char const (&s)[N]) const noexcept -> bool
{
  return ends_with(String(s));
}

PURE HOSTDEV constexpr auto
String::substr(I pos, I len) const -> String
{
  ASSERT(pos <= size());
  if (len == npos || pos + len > size()) {
    len = size() - pos;
  }
  return String{data() + pos, len};
}

PURE HOSTDEV constexpr auto
String::find_last_of(char const c) const noexcept -> I
{
  for (I i = size(); i > 0; --i) {
    if (data()[i - 1] == c) {
      return i - 1;
    }
  }
  return npos;
}

} // namespace um2

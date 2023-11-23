#pragma once

#include <um2/stdlib/algorithm.hpp> // copy
#include <um2/stdlib/math.hpp>      // min
#include <um2/stdlib/memory.hpp>    // addressof
#include <um2/stdlib/utility.hpp>   // move

#include <bit>     // std::endian::native, std::endian::big
#include <cstring> // memcpy, strcmp
#include <string>  // std::string

// clang-tidy does poorly with conditional deletions and claims there are leaks.
// Valgrind shows no leaks.
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)

namespace um2
{

//==============================================================================
// STRING
//==============================================================================
// A std::string-like class, but without an allocator template parameter.
//
// NOTE: ASSUMES LITTLE ENDIAN
// This should be true for all x86 processors and NVIDIA GPUs.

static_assert(std::endian::native == std::endian::little,
              "Only little endian is supported.");

struct String {

private:
  // Heap-allocated string representation.
  // 24 bytes
  struct Long {
    uint64_t is_long : 1; // Single bit for representation flag.
    uint64_t cap : 63;    // Capacity of the string. (Does not include null.)
    uint64_t size;        // Size of the string.
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
    uint8_t size : 7;    // 7 bits for the size of the string.
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

public:
  // The maximum capacity of a long string.
  static Size constexpr npos = sizeMax();

  //==============================================================================
  // Constructors
  //==============================================================================

  HOSTDEV constexpr String() noexcept;

  HOSTDEV constexpr String(String const & s) noexcept;

  HOSTDEV constexpr String(String && s) noexcept;

  // NOLINTBEGIN(google-explicit-constructor); justification: match std::string
  template <uint64_t N>
  HOSTDEV constexpr String(char const (&s)[N]) noexcept;
  // NOLINTEND(google-explicit-constructor)

  HOSTDEV constexpr explicit String(char const * s) noexcept;

  HOSTDEV constexpr String(char const * s, Size n) noexcept;

  template <std::integral T>
  explicit constexpr String(T x) noexcept;

  template <std::floating_point T>
  explicit constexpr String(T x) noexcept;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV constexpr ~String() noexcept
  {
    if (isLong()) {
      ::operator delete(_r.l.data);
    }
  }

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> char *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> char const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLong() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Size;

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
  operator[](Size i) noexcept -> char &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> char const &;

  HOSTDEV constexpr auto
  operator+=(String const & s) noexcept -> String &;

  HOSTDEV constexpr auto
  operator+=(char c) noexcept -> String &;

  //==============================================================================
  // Methods
  //==============================================================================
  // NOLINTBEGIN(readability-identifier-naming); justification: match std::string

  PURE HOSTDEV [[nodiscard]] constexpr auto
  compare(String const & s) const noexcept -> int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  c_str() const noexcept -> char const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  ends_with(String const & s) const noexcept -> bool;

  template <uint64_t N>
  PURE HOSTDEV [[nodiscard]] auto
  ends_with(char const (&s)[N]) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  find_last_of(char c) const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  starts_with(String const & s) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  substr(Size pos, Size len = npos) const -> String;

  // NOLINTEND(readability-identifier-naming)

  //==============================================================================
  // HIDDEN
  //==============================================================================

  CONST HOSTDEV HIDDEN static constexpr auto
  fitsInShort(uint64_t n) noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongCap() const noexcept -> uint64_t;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongPointer() noexcept -> char *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongPointer() const noexcept -> char const *;

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongSize() const noexcept -> uint64_t;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getPointer() noexcept -> char *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getPointer() const noexcept -> char const *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr static auto
  getShortCap() noexcept -> uint64_t;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getShortPointer() noexcept -> char *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getShortPointer() const noexcept -> char const *;

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getShortSize() const noexcept -> uint8_t;

  HOSTDEV HIDDEN constexpr void
  initLong(uint64_t n) noexcept;

  HOSTDEV HIDDEN constexpr void
  initShort(uint64_t n) noexcept;

}; // struct String

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

// mimic allocateAndCopy(char const *, uint64_t)
#define LONG_ALLOCATE_AND_COPY(ptr, num_elem)                                            \
  char const * ss = (ptr);                                                               \
  uint64_t const nn = (num_elem);                                                        \
  ASSERT_ASSUME(nn > 0);                                                                 \
  _r.l.is_long = 1;                                                                      \
  _r.l.cap = (nn - 1) & long_cap_mask;                                                   \
  _r.l.size = nn - 1;                                                                    \
  _r.l.data = static_cast<char *>(::operator new(nn));                                   \
  copy(ss, ss + nn, _r.l.data);

#define SHORT_COPY(ptr, num_elem)                                                        \
  char const * ss = (ptr);                                                               \
  uint64_t const nn = (num_elem);                                                        \
  ASSERT_ASSUME(nn > 0);                                                                 \
  _r.s.is_long = 0;                                                                      \
  _r.s.size = (nn - 1) & short_size_mask;                                                \
  copy(ss, ss + nn, addressof(_r.s.data[0]));

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

HOSTDEV constexpr String::String(char const * s, Size const n) noexcept
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

// TODO(kcvaughn): write these without std::to_string
template <std::integral T>
constexpr String::String(T x) noexcept
{
  // A 64-bit integer can have at most 20 digits
  std::string const s = std::to_string(x);
  auto const cap = s.size();
  ASSERT_ASSUME(cap < min_cap);
  SHORT_COPY(s.data(), cap + 1);
  _r.s.data[cap] = '\0';
}

template <std::floating_point T>
constexpr String::String(T x) noexcept
{
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
String::size() const noexcept -> Size
{
  return isLong() ? static_cast<Size>(getLongSize()) : static_cast<Size>(getShortSize());
}

// Allocated bytes - 1 for null terminator
PURE HOSTDEV constexpr auto
String::capacity() const noexcept -> Size
{
  return isLong() ? static_cast<Size>(getLongCap()) : static_cast<Size>(getShortCap());
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
String::operator[](Size i) noexcept -> char &
{
  return data()[i];
}

PURE HOSTDEV constexpr auto
String::operator[](Size i) const noexcept -> char const &
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
  return *this;
}

//==============================================================================
// Methods
//==============================================================================

PURE HOSTDEV constexpr auto
String::compare(String const & s) const noexcept -> int
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
  for (Size i = 0; i < s.size(); ++i) {
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
  Size const l_size = size();
  Size const r_size = s.size();
  if (l_size < r_size) {
    return false;
  }
  char const * l_data = data() + l_size - r_size;
  char const * r_data = s.data();
  for (Size i = 0; i < r_size; ++i) {
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
// NOLINTNEXTLINE(readability-identifier-naming) justification: mimics std::string
String::ends_with(char const (&s)[N]) const noexcept -> bool
{
  return ends_with(String(s));
}

PURE HOSTDEV constexpr auto
String::substr(Size pos, Size len) const -> String
{
  ASSERT_ASSUME(pos <= size());
  if (len == npos || pos + len > size()) {
    len = size() - pos;
  }
  // It is important that we do not use a braced-init-list here
  // NOLINTNEXTLINE(modernize-return-braced-init-list) justified
  return String(data() + pos, len);
}

PURE HOSTDEV constexpr auto
String::find_last_of(char const c) const noexcept -> Size
{
  for (Size i = size(); i > 0; --i) {
    if (data()[i - 1] == c) {
      return i - 1;
    }
  }
  return npos;
}

//==============================================================================
// HIDDEN
//==============================================================================

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
// NOLINTNEXTLINE(readability-make-member-function-const) justification: can't be const
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

} // namespace um2
// NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)

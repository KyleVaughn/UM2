#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/algorithm/copy.hpp>
#include <um2/stdlib/memory/addressof.hpp>
#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/is_pointer_in_range.hpp>
#include <um2/stdlib/string_view.hpp>

//==============================================================================
// STRING
//==============================================================================

// For developers: clang-tidy is a bit overzealous with its warnings in this file.
// We check for memory leaks with Valgrind, so we are safe to ignore these warnings.
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete, clang-analyzer-cplusplus.NewDeleteLeaks)

namespace um2
{

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class String
{

  public:
  using Ptr = char *;
  using ConstPtr = char const *;

  private:
  // Heap-allocated string representation.
  // 24 bytes
  struct Long {
    Ptr data;             // Pointer to the string data.
    uint64_t size;        // Size of the string. (Does NOT include null.)
    uint64_t cap : 63;    // Capacity of the string. (Does include null.)
    uint64_t is_long : 1; // Single bit for representation flag.
  };

  // The maximum capacity of a short string.
  // 24 byte string - 1 byte for representation flag and size = 23 bytes
  // This includes the null terminator.
  static uint64_t constexpr min_cap = sizeof(Long) - 1;

  // Stack-allocated string representation.
  struct Short {
    char data[min_cap];  // Data of the string.
    uint8_t size : 7;    // 7 bits for the size of the string. (Does not include null.)
    uint8_t is_long : 1; // Single bit for representation flag.
  };

  // Mask to convert to a 63-bit unsigned integer.
  static uint64_t constexpr long_cap_mask = 0x7FFFFFFFFFFFFFFF;

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
      Short s;
      Long l;
      Raw r;
    };
  };

  Rep _r;

  //==============================================================================
  // Private functions
  //==============================================================================

  // Assign a short string to the string. Will not allocate memory.
  // n < min_cap
  HOSTDEV constexpr auto 
  assignShort(StringView sv) noexcept -> String &;

  // Assign a long string to the string. Will allocate memory if necessary.
  HOSTDEV constexpr auto
  assignLong(StringView sv) noexcept -> String &;

  // Does a string of length n fit in a short string? n does not include the null terminator.
  CONST HOSTDEV static constexpr auto
  fitsInShort(uint64_t n) noexcept -> bool;

  // Get the capacity of the long string. Includes the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongCap() const noexcept -> uint64_t;

  // Get a pointer to the long string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongPointer() noexcept -> Ptr;

  // Get a pointer to the long string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongPointer() const noexcept -> ConstPtr;

  // Get the size of the long string. Does NOT include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongSize() const noexcept -> uint64_t;

  // Get a pointer to the string data regardless of representation.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPointer() noexcept -> Ptr;

  // Get a pointer to the string data regardless of representation.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPointer() const noexcept -> ConstPtr;

  // Get a pointer to the short string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortPointer() noexcept -> Ptr;

  // Get a pointer to the short string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortPointer() const noexcept -> ConstPtr;

  // Get the size of the short string. Does NOT include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortSize() const noexcept -> uint64_t;

  // Initialize the string with a pointer to a string and its length.
  // Does not include the null terminator.
  HOSTDEV constexpr void
  init(ConstPtr s, uint64_t size) noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLong() const noexcept -> bool;

  HOSTDEV constexpr void
  setLongCap(uint64_t cap) noexcept;

  HOSTDEV constexpr void
  setLongPointer(Ptr p) noexcept;

  HOSTDEV constexpr void
  setLongSize(uint64_t size) noexcept;

  HOSTDEV constexpr void
  setShortSize(uint64_t size) noexcept;

public:
  // The maximum capacity of a long string.
  static Int constexpr npos = intMax();

  //==============================================================================
  // Constructors and assignment
  //==============================================================================

  HOSTDEV constexpr String() noexcept;

  HOSTDEV constexpr String(String const & s) noexcept;

  HOSTDEV constexpr String(String && s) noexcept;

  // NOLINTNEXTLINE(google-explicit-constructor) match std::string
  HOSTDEV constexpr String(char const * s) noexcept;

  HOSTDEV constexpr auto
  operator=(String const & s) noexcept -> String &;

  HOSTDEV constexpr auto
  operator=(String && s) noexcept -> String &;

  HOSTDEV constexpr auto
  assign(StringView sv) noexcept -> String &;

  HOSTDEV constexpr auto
  assign(char const * s, Int n) noexcept -> String &;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV inline constexpr ~String() noexcept;

  //==============================================================================
  // Element access
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> Ptr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> ConstPtr;

  //==============================================================================
  // Iterators
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() noexcept -> Ptr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> ConstPtr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cbegin() const noexcept -> ConstPtr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() noexcept -> Ptr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> ConstPtr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cend() const noexcept -> ConstPtr;

  //==============================================================================
  // Capacity
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  empty() const noexcept -> bool;

  // Not including the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Int;

  // The number of characters that can be held without reallocating storage.
  // Does not include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Int;


//  //==============================================================================
//  // Operators
//  //==============================================================================







//  operator>(String const & s) const noexcept -> bool;
//
//  PURE HOSTDEV constexpr auto
//  operator>=(String const & s) const noexcept -> bool;
//
//  PURE HOSTDEV constexpr auto
//  operator[](Int i) noexcept -> char &;
//
//  PURE HOSTDEV constexpr auto
//  operator[](Int i) const noexcept -> char const &;
//
//  HOSTDEV constexpr auto
//  operator+=(String const & s) noexcept -> String &;
//
//  HOSTDEV constexpr auto
//  operator+=(char c) noexcept -> String &;
//
//  //==============================================================================
//  // Methods
//  //==============================================================================
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  compare(String const & s) const noexcept -> int;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  c_str() const noexcept -> char const *;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  ends_with(String const & s) const noexcept -> bool;
//
//  template <uint64_t N>
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  ends_with(char const (&s)[N]) const noexcept -> bool;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  find_last_of(char c) const noexcept -> Int;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  starts_with(ConstPtr s) const noexcept -> bool;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  substr(Int pos, Int len = npos) const -> String;
//
}; // class String

//==============================================================================
// Private functions
//==============================================================================

HOSTDEV constexpr auto 
String::assignShort(StringView sv) noexcept -> String &
{
  uint64_t const n = sv.size();
  ASSERT(n < min_cap);
  Ptr p = nullptr;
  if (isLong()) {
    p = getLongPointer();
    setLongSize(n);
  } else {
    p = getShortPointer();
    setShortSize(n);
  }
  ASSERT(!um2::is_pointer_in_range(sv.begin(), sv.end(), p)); 
  um2::copy(sv.begin(), sv.end(), p);
  p[n] = '\0';
  return *this;
}

HOSTDEV constexpr auto
String::assignLong(StringView sv) noexcept -> String &
{
  uint64_t const n = sv.size();
  ASSERT(n >= min_cap);
  Ptr p = nullptr;
  if (static_cast<uint64_t>(capacity()) < n) {
    p = static_cast<Ptr>(::operator new(n + 1));
    // We know that the pointers don't alias, since we just allocated p.
    memcpy(p, sv.begin(), n);
    if (isLong()) {
      ::operator delete(getLongPointer());
    }
    setLongCap(n + 1);
    setLongPointer(p);
  } else {
    // We already have enough capacity, so we don't need to allocate.
    p = getLongPointer();
    ASSERT(!um2::is_pointer_in_range(sv.begin(), sv.end(), p));
    um2::copy(sv.begin(), sv.end(), p);
  }
  setLongSize(n);
  p[n] = '\0';
  return *this;
}

PURE HOSTDEV constexpr auto
String::getLongSize() const noexcept -> uint64_t
{
  return _r.l.size;
}

PURE HOSTDEV constexpr auto
String::getShortSize() const noexcept -> uint64_t
{
  return _r.s.size;
}

PURE HOSTDEV constexpr auto
String::getLongCap() const noexcept -> uint64_t
{
  return _r.l.cap;
}

PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-make-member-function-const) we offer const next
String::getLongPointer() noexcept -> Ptr
{
  return _r.l.data;
}

PURE HOSTDEV constexpr auto
String::getLongPointer() const noexcept -> ConstPtr
{
  return _r.l.data;
}

PURE HOSTDEV constexpr auto
String::getShortPointer() noexcept -> Ptr
{
  return um2::addressof(_r.s.data[0]);
}

PURE HOSTDEV constexpr auto
String::getShortPointer() const noexcept -> ConstPtr
{
  return um2::addressof(_r.s.data[0]);
}

PURE HOSTDEV constexpr auto
String::getPointer() noexcept -> Ptr
{
  return isLong() ? getLongPointer() : getShortPointer();
}

PURE HOSTDEV constexpr auto
String::getPointer() const noexcept -> ConstPtr
{
  return isLong() ? getLongPointer() : getShortPointer();
}

// n does not include the null terminator
CONST HOSTDEV constexpr auto
String::fitsInShort(uint64_t n) noexcept -> bool
{
  return n < min_cap;
}

HOSTDEV constexpr void
String::init(ConstPtr s, uint64_t size) noexcept
{
  ASSERT(s != nullptr);
  Ptr p = nullptr;
  if (fitsInShort(size)) {
    setShortSize(size);
    p = getShortPointer();
  } else {
    p = static_cast<Ptr>(::operator new(size + 1));
    setLongPointer(p);
    setLongCap(size + 1);
    setLongSize(size);
  }
  // We know the pointers don't alias, since the string isn't initialized yet.
  um2::copy(s, s + size, p);
  p[size] = '\0';
}

PURE HOSTDEV constexpr auto
String::isLong() const noexcept -> bool
{
  return _r.s.is_long;
}

HOSTDEV constexpr void
String::setLongCap(uint64_t cap) noexcept
{
  _r.l.cap = cap & long_cap_mask;
  _r.l.is_long = true;
}

HOSTDEV constexpr void
String::setLongPointer(Ptr p) noexcept
{
  _r.l.data = p;
}

HOSTDEV constexpr void
String::setLongSize(uint64_t size) noexcept
{
  _r.l.size = size;
}

HOSTDEV constexpr void
String::setShortSize(uint64_t size) noexcept
{
  ASSERT(size < min_cap);
  _r.s.size = size & short_size_mask;
  _r.s.is_long = false;
}

//==============================================================================
// Constructors and assignment
//==============================================================================

// For a union without a user-defined default constructor, value initialization is zero
// initialization
HOSTDEV constexpr String::String() noexcept
    : _r()
{
  ASSERT(_r.r.raw[0] == 0);
  ASSERT(_r.r.raw[1] == 0);
  ASSERT(_r.r.raw[2] == 0);
}

HOSTDEV constexpr String::String(String const & s) noexcept
{
  if (!s.isLong()) {
    // If this is a short string, it is trivially copyable
    _r = s._r;
  } else {
    init(s.getLongPointer(), s.getLongSize());
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

HOSTDEV constexpr String::String(char const * s) noexcept
{
  ASSERT(s != nullptr);
  // Get the length of the string (not including null terminator)
  auto const n = strlen(s);
  ASSERT(n > 0);
  init(s, n);
}

HOSTDEV constexpr auto
String::operator=(String const & s) noexcept -> String &
{
  if (this != um2::addressof(s)) {
    assign(s.data(), s.size());
  }
  return *this;
}

HOSTDEV constexpr auto
String::operator=(String && s) noexcept -> String &
{
  if (this != addressof(s)) {
    if (isLong()) {
      ::operator delete(getLongPointer());
    }
    _r = s._r;
    // We just copied the struct, so we are safe to overwrite s.
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    s._r = Rep();
  }
  return *this;
}

HOSTDEV constexpr auto
String::assign(StringView sv) noexcept -> String &
{
  return fitsInShort(sv.size()) ? assignShort(sv) : assignLong(sv); 
}

HOSTDEV constexpr auto
String::assign(char const * s, Int const n) noexcept -> String &
{
  ASSERT(s != nullptr);
  ASSERT(n > 0);
  StringView const sv(s, static_cast<uint64_t>(n));
  return assign(sv);
}

//////
//////HOSTDEV constexpr String::String(char const * begin, char const * end) noexcept
//////{
//////  // begin and end are pointers to the first and one past the last character
//////  // of the string, respectively.
//////  //
//////  // "test" -> begin = &t, end = &t + 4
//////  // Hence, end is not a valid memory location, nor necessarily the null
//////  // terminator.
//////  auto const n = static_cast<uint64_t>(end - begin);
//////  ASSERT(n > 0);
//////  if (n + 1 <= min_cap) {
//////    _r.s.is_long = 0;
//////    _r.s.size = n & short_size_mask;
//////    auto * dest = addressof(_r.s.data[0]);
//////    while (begin != end) {
//////      *dest = *begin;
//////      ++begin;
//////      ++dest;
//////    }
//////    *dest = '\0';
//////  } else {
//////    _r.l.is_long = 1;
//////    _r.l.cap = n & long_cap_mask;
//////    _r.l.size = n;
//////    _r.l.data = static_cast<char *>(::operator new(n));
//////    auto * dest = _r.l.data;
//////    while (begin != end) {
//////      *dest = *begin;
//////      ++begin;
//////      ++dest;
//////    }
//////    *dest = '\0';
//////  }
//////}
//////
//////// std::to_string should not allocate here due to small string optimization, so
//////// we are fine keeping these for now. They should be replaced with something
//////// like snprintf in the future.
//////template <std::integral T>
//////constexpr String::String(T x) noexcept
//////{
//////  // A 64-bit integer can have at most 20 chars
//////  std::string const s = std::to_string(x);
//////  auto const cap = s.size();
//////  ASSERT_ASSUME(cap < min_cap);
//////  SHORT_COPY(s.data(), cap + 1);
//////  _r.s.data[cap] = '\0';
//////}
//////
//////template <std::floating_point T>
//////constexpr String::String(T x) noexcept
//////{
//////  // A 64-bit floating point number can have at most 24 chars, but
//////  // many numbers will have fewer than 24 chars. So, we ASSERT that
//////  // this is the case.
//////  std::string const s = std::to_string(x);
//////  auto const cap = s.size();
//////  ASSERT_ASSUME(cap < min_cap);
//////  SHORT_COPY(s.data(), cap + 1);
//////  _r.s.data[cap] = '\0';
//////}
//////

//==============================================================================
// Destructor
//==============================================================================

HOSTDEV inline constexpr String::~String() noexcept
{
  if (isLong()) {
    ::operator delete(getLongPointer());
  }
}

//==============================================================================
// Element access
//==============================================================================

PURE HOSTDEV constexpr auto
String::data() noexcept -> Ptr
{
  return getPointer();
}

PURE HOSTDEV constexpr auto
String::data() const noexcept -> ConstPtr
{
  return getPointer();
}

//==============================================================================
// Iterators
//==============================================================================

PURE HOSTDEV constexpr auto
String::begin() noexcept -> Ptr
{
  return data();
}

PURE HOSTDEV constexpr auto
String::begin() const noexcept -> ConstPtr
{
  return data();
}

PURE HOSTDEV constexpr auto
String::cbegin() const noexcept -> ConstPtr
{
  return data();
}

PURE HOSTDEV constexpr auto
String::end() noexcept -> Ptr
{
  return data() + size();
}

PURE HOSTDEV constexpr auto
String::end() const noexcept -> ConstPtr
{
  return data() + size();
}

PURE HOSTDEV constexpr auto
String::cend() const noexcept -> ConstPtr
{
  return data() + size();
}

//==============================================================================
// Capacity
//==============================================================================

PURE HOSTDEV [[nodiscard]] constexpr auto
String::empty() const noexcept -> bool
{
  return size() == 0;
}

PURE HOSTDEV [[nodiscard]] constexpr auto
String::size() const noexcept -> Int
{
  return isLong() ? static_cast<Int>(getLongSize()) : static_cast<Int>(getShortSize());
}

// Allocated bytes - 1 for null terminator
PURE HOSTDEV constexpr auto
String::capacity() const noexcept -> Int
{
  return isLong() ? static_cast<Int>(getLongCap()) - 1 : static_cast<Int>(min_cap) - 1;
}













////==============================================================================
//// Operators
////==============================================================================
//






























//////// These std::string assignment operators are a bit inefficient, but the number of
//////// heap allocations is the same as if we had just copied the string, so it's not
//////// too bad.
//////constexpr auto
//////String::operator=(std::string const & s) noexcept -> String &
//////{
//////  String tmp(s.c_str());
//////  return *this = um2::move(tmp);
//////}
//////
//////constexpr auto
//////String::operator=(std::string && s) noexcept -> String &
//////{
//////  String tmp(s.c_str());
//////  return *this = um2::move(tmp);
//////}
//////
//////template <uint64_t N>
//////HOSTDEV constexpr auto
//////String::operator=(char const (&s)[N]) noexcept -> String &
//////{
//////  if (isLong()) {
//////    ::operator delete(_r.l.data);
//////  }
//////  // Short string
//////  if constexpr (N <= min_cap) {
//////    SHORT_COPY(s, N);
//////    ASSERT(_r.s.data[N - 1] == '\0');
//////  } else {
//////    LONG_ALLOCATE_AND_COPY(s, N);
//////    ASSERT(_r.l.data[N - 1] == '\0');
//////  }
//////  return *this;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::operator==(String const & s) const noexcept -> bool
//////{
//////  Int const l_size = size();
//////  Int const r_size = s.size();
//////  if (l_size != r_size) {
//////    return false;
//////  }
//////  char const * l_data = data();
//////  char const * r_data = s.data();
//////  for (Int i = 0; i < l_size; ++i) {
//////    // NOLINTNEXTLINE
//////    if (*l_data != *r_data) {
//////      return false;
//////    }
//////    ++l_data;
//////    ++r_data;
//////  }
//////  return true;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::operator!=(String const & s) const noexcept -> bool
//////{
//////  return !(*this == s);
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::operator<(String const & s) const noexcept -> bool
//////{
//////  return compare(s) < 0;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::operator<=(String const & s) const noexcept -> bool
//////{
//////  return compare(s) <= 0;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::operator>(String const & s) const noexcept -> bool
//////{
//////  return compare(s) > 0;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::operator>=(String const & s) const noexcept -> bool
//////{
//////  return compare(s) >= 0;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::operator[](Int i) noexcept -> char &
//////{
//////  return data()[i];
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::operator[](Int i) const noexcept -> char const &
//////{
//////  return data()[i];
//////}
//////
//////HOSTDEV constexpr auto
//////String::operator+=(String const & s) noexcept -> String &
//////{
//////  auto const new_size = static_cast<uint64_t>(size() + s.size());
//////  if (fitsInShort(new_size + 1)) {
//////    // This must be a short string, so we can just copy the data.
//////    ASSERT(!isLong());
//////    ASSERT(!s.isLong());
//////    ASSERT(new_size < min_cap);
//////    memcpy(getPointer() + size(), s.data(), static_cast<uint64_t>(s.size() + 1));
//////    _r.s.size = new_size & short_size_mask;
//////  } else {
//////    // Otherwise, we need to allocate a new string and copy the data.
//////    char * tmp = static_cast<char *>(::operator new(new_size + 1));
//////    memcpy(tmp, data(), static_cast<uint64_t>(size()));
//////    memcpy(tmp + size(), s.data(), static_cast<uint64_t>(s.size() + 1));
//////    if (isLong()) {
//////      ::operator delete(_r.l.data);
//////    }
//////    _r.l.is_long = 1;
//////    _r.l.cap = (new_size + 1) & long_cap_mask;
//////    _r.l.size = new_size;
//////    _r.l.data = tmp;
//////  }
//////  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) Valgrind says this is fine
//////  return *this;
//////}
//////
//////HOSTDEV constexpr auto
//////String::operator+=(char const c) noexcept -> String &
//////{
//////  // If this is a short string and the size of the new string is less than
//////  // the capacity of the short string, we can just append the new string.
//////  auto const new_size = static_cast<uint64_t>(size() + 1);
//////  if (fitsInShort(new_size + 1)) {
//////    ASSERT(!isLong());
//////    _r.s.data[size()] = c;
//////    _r.s.data[size() + 1] = '\0';
//////    _r.s.size += 1;
//////  } else {
//////    // Otherwise, we need to allocate a new string and copy the data.
//////    char * tmp = static_cast<char *>(::operator new(new_size + 1));
//////    memcpy(tmp, data(), static_cast<uint64_t>(size()));
//////    tmp[size()] = c;
//////    tmp[size() + 1] = '\0';
//////    if (isLong()) {
//////      ::operator delete(_r.l.data);
//////    }
//////    _r.l.is_long = 1;
//////    _r.l.cap = (new_size + 1) & long_cap_mask;
//////    _r.l.size = new_size;
//////    _r.l.data = tmp;
//////  }
//////  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) Valgrind says this is fine
//////  return *this;
//////}
//////
////==============================================================================
//// Member functions
////==============================================================================
//
//////PURE HOSTDEV constexpr auto
//////String::compare(String const & s) const noexcept -> int
//////{
//////  Int const l_size = size();
//////  Int const r_size = s.size();
//////  Int const min_size = um2::min(l_size, r_size);
//////  char const * l_data = data();
//////  char const * r_data = s.data();
//////  for (Int i = 0; i < min_size; ++i) {
//////    if (*l_data != *r_data) {
//////      return static_cast<int>(*l_data) - static_cast<int>(*r_data);
//////    }
//////    ++l_data;
//////    ++r_data;
//////  }
//////  return static_cast<int>(l_size) - static_cast<int>(r_size);
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::c_str() const noexcept -> char const *
//////{
//////  return data();
//////}
//////
////PURE HOSTDEV constexpr auto
////String::starts_with(ConstPtr s) const noexcept -> bool
////{
////  if (size() < s.size()) {
////    return false;
////  }
////  char const * l_data = data();
////  char const * r_data = s.data();
////  for (Int i = 0; i < s.size(); ++i) {
////    if (*l_data != *r_data) {
////      return false;
////    }
////    ++l_data;
////    ++r_data;
////  }
////  return true;
////}
//
//////PURE HOSTDEV constexpr auto
//////String::ends_with(String const & s) const noexcept -> bool
//////{
//////  Int const l_size = size();
//////  Int const r_size = s.size();
//////  if (l_size < r_size) {
//////    return false;
//////  }
//////  char const * l_data = data() + l_size - r_size;
//////  char const * r_data = s.data();
//////  for (Int i = 0; i < r_size; ++i) {
//////    if (*l_data != *r_data) {
//////      return false;
//////    }
//////    ++l_data;
//////    ++r_data;
//////  }
//////  return true;
//////}
//////
//////template <uint64_t N>
//////PURE HOSTDEV constexpr auto
//////// NOLINTNEXTLINE(readability-identifier-naming) match std::string
//////String::ends_with(char const (&s)[N]) const noexcept -> bool
//////{
//////  return ends_with(String(s));
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::substr(Int pos, Int len) const -> String
//////{
//////  ASSERT(pos <= size());
//////  if (len == npos || pos + len > size()) {
//////    len = size() - pos;
//////  }
//////  return String{data() + pos, len};
//////}
//////
//////PURE HOSTDEV constexpr auto
//////String::find_last_of(char const c) const noexcept -> Int
//////{
//////  for (Int i = size(); i > 0; --i) {
//////    if (data()[i - 1] == c) {
//////      return i - 1;
//////    }
//////  }
//////  return npos;
//////}

} // namespace um2

// NOLINTEND(clang-analyzer-cplusplus.NewDelete, clang-analyzer-cplusplus.NewDeleteLeaks)

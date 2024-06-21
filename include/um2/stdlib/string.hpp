#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/algorithm/copy.hpp>
#include <um2/stdlib/algorithm/max.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/memory/addressof.hpp>
#include <um2/stdlib/string_view.hpp>
#include <um2/stdlib/utility/is_pointer_in_range.hpp>
#include <um2/stdlib/utility/move.hpp>

#include <concepts>

//==============================================================================
// STRING
//==============================================================================

namespace um2
{

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
  HOSTDEV inline auto
  assignShort(StringView sv) noexcept -> String &;

  // Assign a long string to the string. Will allocate memory if necessary.
  HOSTDEV inline auto
  assignLong(StringView sv) noexcept -> String &;

  // Does a string of length n fit in a short string? n does not include the null
  // terminator.
  CONST HOSTDEV static inline auto
  fitsInShort(uint64_t n) noexcept -> bool;

  // Get the capacity of the long string. Includes the null terminator.
  PURE HOSTDEV [[nodiscard]] inline auto
  getLongCap() const noexcept -> uint64_t;

  // Get a pointer to the long string data.
  PURE HOSTDEV [[nodiscard]] inline auto
  getLongPointer() noexcept -> Ptr;

  // Get a pointer to the long string data.
  PURE HOSTDEV [[nodiscard]] inline auto
  getLongPointer() const noexcept -> ConstPtr;

  // Get the size of the long string. Does NOT include the null terminator.
  PURE HOSTDEV [[nodiscard]] inline auto
  getLongSize() const noexcept -> uint64_t;

  // Get a pointer to the string data regardless of representation.
  PURE HOSTDEV [[nodiscard]] inline auto
  getPointer() noexcept -> Ptr;

  // Get a pointer to the string data regardless of representation.
  PURE HOSTDEV [[nodiscard]] inline auto
  getPointer() const noexcept -> ConstPtr;

  // Get a pointer to the short string data.
  PURE HOSTDEV [[nodiscard]] inline auto
  getShortPointer() noexcept -> Ptr;

  // Get a pointer to the short string data.
  PURE HOSTDEV [[nodiscard]] inline auto
  getShortPointer() const noexcept -> ConstPtr;

  // Get the size of the short string. Does NOT include the null terminator.
  PURE HOSTDEV [[nodiscard]] inline auto
  getShortSize() const noexcept -> uint64_t;

  // Initialize the string with a pointer to a string and its length.
  // Does not include the null terminator.
  HOSTDEV inline void
  init(ConstPtr s, uint64_t size) noexcept;

  PURE HOSTDEV [[nodiscard]] inline auto
  isLong() const noexcept -> bool;

  HOSTDEV inline void
  setLongCap(uint64_t cap) noexcept;

  HOSTDEV inline void
  setLongPointer(Ptr p) noexcept;

  HOSTDEV inline void
  setLongSize(uint64_t size) noexcept;

  HOSTDEV inline void
  setShortSize(uint64_t size) noexcept;

  HOSTDEV inline void
  setSize(uint64_t size) noexcept;

  HOSTDEV inline void
  growByAndReplace(uint64_t old_cap, uint64_t delta_cap, uint64_t old_sz, uint64_t n_copy,
                   uint64_t n_del, uint64_t n_add, char const * s) noexcept;

public:
  // The maximum capacity of a long string.
  static constexpr Int npos = intMax();

  //==============================================================================
  // Constructors and assignment
  //==============================================================================

  HOSTDEV inline String() noexcept;

  HOSTDEV inline String(String const & s) noexcept;

  HOSTDEV inline String(String && s) noexcept;

  // NOLINTNEXTLINE(google-explicit-constructor) match std::string
  HOSTDEV inline String(char const * s) noexcept;

  HOSTDEV inline String(String const & s, Int pos, Int count = npos) noexcept;

  template <std::integral T>
  explicit String(T t) noexcept;

  template <std::floating_point T>
  explicit String(T t) noexcept;

  HOSTDEV inline auto
  operator=(String const & s) noexcept -> String &;

  HOSTDEV inline auto
  operator=(String && s) noexcept -> String &;

  HOSTDEV explicit inline String(StringView sv) noexcept;

  HOSTDEV inline
  // NOLINTNEXTLINE(google-explicit-constructor) match std::string
  operator StringView() const noexcept;

  HOSTDEV inline auto
  assign(StringView sv) noexcept -> String &;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV inline ~String() noexcept;

  //==============================================================================
  // Element access
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] inline auto
  front() noexcept -> char &;

  PURE HOSTDEV [[nodiscard]] inline auto
  front() const noexcept -> char const &;

  PURE HOSTDEV [[nodiscard]] inline auto
  back() noexcept -> char &;

  PURE HOSTDEV [[nodiscard]] inline auto
  back() const noexcept -> char const &;

  PURE HOSTDEV [[nodiscard]] inline auto
  data() noexcept -> Ptr;

  PURE HOSTDEV [[nodiscard]] inline auto
  data() const noexcept -> ConstPtr;

  //==============================================================================
  // Iterators
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] inline auto
  begin() noexcept -> Ptr;

  PURE HOSTDEV [[nodiscard]] inline auto
  begin() const noexcept -> ConstPtr;

  PURE HOSTDEV [[nodiscard]] inline auto
  cbegin() const noexcept -> ConstPtr;

  PURE HOSTDEV [[nodiscard]] inline auto
  end() noexcept -> Ptr;

  PURE HOSTDEV [[nodiscard]] inline auto
  end() const noexcept -> ConstPtr;

  PURE HOSTDEV [[nodiscard]] inline auto
  cend() const noexcept -> ConstPtr;

  //==============================================================================
  // Capacity
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] inline auto
  empty() const noexcept -> bool;

  // Not including the null terminator.
  PURE HOSTDEV [[nodiscard]] inline auto
  size() const noexcept -> Int;

  // The number of characters that can be held without reallocating storage.
  // Does not include the null terminator.
  PURE HOSTDEV [[nodiscard]] inline auto
  capacity() const noexcept -> Int;

  //==============================================================================
  // Modifiers
  //==============================================================================

  HOSTDEV inline auto
  append(char const * s, Int n) -> String &;

  HOSTDEV inline auto
  operator+=(char const * s) -> String &;

  HOSTDEV inline auto
  operator+=(String const & s) -> String &;

  //==============================================================================
  // Operations
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] inline auto
  compare(String const & s) const noexcept -> int;

  template <std::convertible_to<StringView> T>
  PURE HOSTDEV [[nodiscard]] inline auto
  compare(T const & t) const noexcept -> int;

  // NOLINTBEGIN(readability-identifier-naming) match std::string
  PURE HOSTDEV [[nodiscard]] inline auto
  ends_with(StringView sv) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] inline auto
  ends_with(char c) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] inline auto
  ends_with(char const * s) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] inline auto
  starts_with(StringView sv) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] inline auto
  starts_with(char c) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] inline auto
  starts_with(char const * s) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] inline auto
  find_last_of(char c, Int pos = npos) const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] inline auto
  substr(Int pos, Int count = npos) const noexcept -> String;

  // NOLINTEND(readability-identifier-naming) match std::string

}; // class String

//==============================================================================
// Free functions
//==============================================================================

PURE HOSTDEV [[nodiscard]] inline auto
operator==(String const & lhs, String const & rhs) -> bool
{
  return lhs.compare(rhs) == 0;
}

PURE HOSTDEV [[nodiscard]] inline auto
operator!=(String const & lhs, String const & rhs) -> bool
{
  return lhs.compare(rhs) != 0;
}

PURE HOSTDEV [[nodiscard]] inline auto
operator<(String const & lhs, String const & rhs) -> bool
{
  return lhs.compare(rhs) < 0;
}

PURE HOSTDEV [[nodiscard]] inline auto
operator<=(String const & lhs, String const & rhs) -> bool
{
  return lhs.compare(rhs) <= 0;
}

PURE HOSTDEV [[nodiscard]] inline auto
operator>(String const & lhs, String const & rhs) -> bool
{
  return lhs.compare(rhs) > 0;
}

PURE HOSTDEV [[nodiscard]] inline auto
operator>=(String const & lhs, String const & rhs) -> bool
{
  return lhs.compare(rhs) >= 0;
}

PURE HOSTDEV [[nodiscard]] inline auto
operator+(String const & lhs, String const & rhs) -> String
{
  String result(lhs);
  result += rhs;
  return result;
}

PURE HOSTDEV [[nodiscard]] inline auto
operator+(String const & lhs, char const * rhs) -> String
{
  String result(lhs);
  result += rhs;
  return result;
}

//==============================================================================
// Private functions
//==============================================================================

HOSTDEV inline auto
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

HOSTDEV inline auto
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

PURE HOSTDEV inline auto
String::getLongSize() const noexcept -> uint64_t
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
  return _r.l.size;
#pragma GCC diagnostic pop
}

PURE HOSTDEV inline auto
String::getShortSize() const noexcept -> uint64_t
{
  return _r.s.size;
}

PURE HOSTDEV inline auto
String::getLongCap() const noexcept -> uint64_t
{
  return _r.l.cap;
}

PURE HOSTDEV inline auto
// NOLINTNEXTLINE(readability-make-member-function-const) OK, we offer const next
String::getLongPointer() noexcept -> Ptr
{
  return _r.l.data;
}

PURE HOSTDEV inline auto
String::getLongPointer() const noexcept -> ConstPtr
{
  return _r.l.data;
}

PURE HOSTDEV inline auto
String::getShortPointer() noexcept -> Ptr
{
  return um2::addressof(_r.s.data[0]);
}

PURE HOSTDEV inline auto
String::getShortPointer() const noexcept -> ConstPtr
{
  return um2::addressof(_r.s.data[0]);
}

PURE HOSTDEV inline auto
String::getPointer() noexcept -> Ptr
{
  return isLong() ? getLongPointer() : getShortPointer();
}

PURE HOSTDEV inline auto
String::getPointer() const noexcept -> ConstPtr
{
  return isLong() ? getLongPointer() : getShortPointer();
}

// n does not include the null terminator
CONST HOSTDEV inline auto
String::fitsInShort(uint64_t n) noexcept -> bool
{
  return n < min_cap;
}

HOSTDEV inline void
String::init(ConstPtr s, uint64_t size) noexcept
{
  ASSERT(s != nullptr);
  Ptr p = nullptr;
  // GCC warns about wraparound here, but it should not be possible
  ASSERT_ASSUME(size < INT32_MAX);
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

PURE HOSTDEV inline auto
String::isLong() const noexcept -> bool
{
  return _r.s.is_long;
}

HOSTDEV inline void
String::setLongCap(uint64_t cap) noexcept
{
  _r.l.cap = cap & long_cap_mask;
  _r.l.is_long = true;
}

HOSTDEV inline void
String::setLongPointer(Ptr p) noexcept
{
  _r.l.data = p;
}

HOSTDEV inline void
String::setLongSize(uint64_t size) noexcept
{
  _r.l.size = size;
}

HOSTDEV inline void
String::setShortSize(uint64_t size) noexcept
{
  ASSERT(size < min_cap);
  _r.s.size = size & short_size_mask;
  _r.s.is_long = false;
}

HOSTDEV inline void
String::setSize(uint64_t size) noexcept
{
  if (isLong()) {
    setLongSize(size);
  } else {
    setShortSize(size);
  }
}

HOSTDEV inline void
String::growByAndReplace(uint64_t old_cap, uint64_t delta_cap, uint64_t old_sz,
                         uint64_t n_copy, uint64_t n_del, uint64_t n_add,
                         char const * s) noexcept
{
  // Haven't planned around massive strings yet. Assert that we don't overflow.
  ASSERT(delta_cap < static_cast<uint64_t>(intMax()));
  ASSERT(old_cap + delta_cap < static_cast<uint64_t>(intMax()));
  ASSERT(2 * old_cap < static_cast<uint64_t>(intMax()));
  Ptr old_p = getPointer();
  uint64_t const cap = um2::max(old_cap + delta_cap, 2 * old_cap);
  Ptr p = static_cast<Ptr>(::operator new(cap + 1)); // + 1 for null terminator
  if (n_copy != 0) {
    // to: p
    // from: old_p
    // n: n_copy
    um2::copy(old_p, old_p + n_copy, p);
  }
  if (n_add != 0) {
    // to: p + __n_copy
    // from: s
    // n: n_add
    um2::copy(s, s + n_add, p + n_copy);
  }
  uint64_t const sec_cp_sz = old_sz - n_del - n_copy;
  if (sec_cp_sz != 0) {
    // to: p + n_copy + n_add
    // from: old_p + n_copy + n_del
    // n: sec_cp_sz
    // old_p + n_copy + n_del + sec_cp_sz == old_p + old_sz
    um2::copy(old_p + n_copy + n_del, old_p + old_sz, p + n_copy + n_add);
  }
  // If the old string was not short, deallocate the memory.
  if (old_cap + 1 != min_cap) {
    ::operator delete(old_p);
  }
  setLongPointer(p);
  setLongCap(cap + 1); // + 1 for null terminator
  old_sz = n_copy + n_add + sec_cp_sz;
  setLongSize(old_sz);
  p[old_sz] = '\0';
}

//==============================================================================
// Constructors and assignment
//==============================================================================

// For a union without a user-defined default constructor, value initialization is zero
// initialization
HOSTDEV inline String::String() noexcept
    : _r()
{
  ASSERT(_r.r.raw[0] == 0);
  ASSERT(_r.r.raw[1] == 0);
  ASSERT(_r.r.raw[2] == 0);
}

HOSTDEV inline String::String(String const & s) noexcept
{
  if (!s.isLong()) {
    // If this is a short string, it is trivially copyable
    _r = s._r;
  } else {
    init(s.getLongPointer(), s.getLongSize());
  }
}

HOSTDEV inline String::String(String && s) noexcept
    : _r(um2::move(s._r))
{
  // If short string, we can copy trivially
  // If long string, we need to move the data.
  // Since the data is a pointer, we can just copy the pointer.
  // Therefore, either way, we can just copy the whole struct.
  s._r = Rep();
}

HOSTDEV inline String::String(char const * s) noexcept
{
  ASSERT(s != nullptr);
  // Get the length of the string (not including null terminator)
  auto const n = strlen(s);
  ASSERT(n > 0);
  init(s, n);
}

HOSTDEV inline String::String(String const & s, Int pos, Int count) noexcept
{
  Int const str_size = s.size();
  ASSERT(pos >= 0);
  ASSERT(pos <= str_size);
  Int const n = um2::min(count, str_size - pos);
  ASSERT(n >= 0);
  init(s.data() + pos, static_cast<uint64_t>(n));
}

// written is only read in debug mode
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
// NOLINTBEGIN(clang-analyzer-deadcode.DeadStores, clang-diagnostic-unused-variable) OK
template <>
inline String::String(int32_t t) noexcept
{
  char buf[32];
  int32_t const len = snprintf(nullptr, 0, "%d", t);
  int32_t const written = snprintf(buf, sizeof(buf), "%d", t);
  ASSERT(written == len);
  init(addressof(buf[0]), static_cast<uint64_t>(len));
}

template <>
inline String::String(int64_t t) noexcept
{
  char buf[32];
  int32_t const len = snprintf(nullptr, 0, "%ld", t);
  int32_t const written = snprintf(buf, sizeof(buf), "%ld", t);
  ASSERT(written == len);
  init(addressof(buf[0]), static_cast<uint64_t>(len));
}

template <>
inline String::String(float t) noexcept
{
  char buf[32];
  auto const d = static_cast<double>(t);
  int32_t const len = snprintf(nullptr, 0, "%f", d);
  int32_t const written = snprintf(buf, sizeof(buf), "%f", d);
  ASSERT(written == len);
  init(addressof(buf[0]), static_cast<uint64_t>(len));
}

template <>
inline String::String(double t) noexcept
{
  char buf[32];
  int32_t const len = snprintf(nullptr, 0, "%f", t);
  int32_t const written = snprintf(buf, sizeof(buf), "%f", t);
  ASSERT(written == len);
  init(addressof(buf[0]), static_cast<uint64_t>(len));
}

#pragma GCC diagnostic pop
// NOLINTEND(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)

HOSTDEV inline auto
String::operator=(String const & s) noexcept -> String &
{
  if (this != um2::addressof(s)) {
    assign(s);
  }
  return *this;
}

HOSTDEV inline auto
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

HOSTDEV inline String::String(StringView sv) noexcept { init(sv.data(), sv.size()); }

PURE HOSTDEV inline String::operator StringView() const noexcept
{
  return {data(), static_cast<uint64_t>(size())};
}

HOSTDEV inline auto
String::assign(StringView sv) noexcept -> String &
{
  return fitsInShort(sv.size()) ? assignShort(sv) : assignLong(sv);
}

//==============================================================================
// Destructor
//==============================================================================

HOSTDEV inline String::~String() noexcept
{
  if (isLong()) {
    ::operator delete(getLongPointer());
  }
}

//==============================================================================
// Element access
//==============================================================================

PURE HOSTDEV inline auto
String::front() noexcept -> char &
{
  return *begin();
}

PURE HOSTDEV inline auto
String::front() const noexcept -> char const &
{
  return *begin();
}

PURE HOSTDEV inline auto
String::back() noexcept -> char &
{
  return *(end() - 1);
}

PURE HOSTDEV inline auto
String::back() const noexcept -> char const &
{
  return *(end() - 1);
}

PURE HOSTDEV inline auto
String::data() noexcept -> Ptr
{
  return getPointer();
}

PURE HOSTDEV inline auto
String::data() const noexcept -> ConstPtr
{
  return getPointer();
}

//==============================================================================
// Iterators
//==============================================================================

PURE HOSTDEV inline auto
String::begin() noexcept -> Ptr
{
  return data();
}

PURE HOSTDEV inline auto
String::begin() const noexcept -> ConstPtr
{
  return data();
}

PURE HOSTDEV inline auto
String::cbegin() const noexcept -> ConstPtr
{
  return data();
}

PURE HOSTDEV inline auto
String::end() noexcept -> Ptr
{
  return data() + size();
}

PURE HOSTDEV inline auto
String::end() const noexcept -> ConstPtr
{
  return data() + size();
}

PURE HOSTDEV inline auto
String::cend() const noexcept -> ConstPtr
{
  return data() + size();
}

//==============================================================================
// Capacity
//==============================================================================

PURE HOSTDEV [[nodiscard]] inline auto
String::empty() const noexcept -> bool
{
  return size() == 0;
}

PURE HOSTDEV [[nodiscard]] inline auto
String::size() const noexcept -> Int
{
  return isLong() ? static_cast<Int>(getLongSize()) : static_cast<Int>(getShortSize());
}

// Allocated bytes - 1 for null terminator
PURE HOSTDEV inline auto
String::capacity() const noexcept -> Int
{
  return isLong() ? static_cast<Int>(getLongCap()) - 1 : static_cast<Int>(min_cap) - 1;
}

//==============================================================================
// Modifiers
//==============================================================================

HOSTDEV inline auto
String::append(char const * s, Int n) -> String &
{
  ASSERT(s != nullptr);
  ASSERT(n > 0);
  Int const cap = capacity(); // not including null terminator
  Int sz = size();
  Int const remaining = cap - sz;
  // If the remaining capacity is greater than or equal to n, there is no need to
  // reallocate
  if (remaining >= n) {
    Ptr p = getPointer();
    um2::copy(s, s + n, p + sz);
    sz += n;
    setSize(static_cast<uint64_t>(sz));
    p[sz] = '\0';
  } else {
    growByAndReplace(static_cast<uint64_t>(cap), static_cast<uint64_t>(n - remaining),
                     static_cast<uint64_t>(sz), static_cast<uint64_t>(sz), 0,
                     static_cast<uint64_t>(n), s);
  }
  return *this;
}

HOSTDEV inline auto
String::operator+=(char const * s) -> String &
{
  return append(s, static_cast<Int>(strlen(s)));
}

HOSTDEV inline auto
String::operator+=(String const & s) -> String &
{
  return append(s.data(), s.size());
}

//==============================================================================
// Operations
//==============================================================================

PURE HOSTDEV [[nodiscard]] inline auto
String::compare(String const & s) const noexcept -> int
{
  StringView const self_sv(data(), static_cast<uint64_t>(size()));
  StringView const sv(s.data(), static_cast<uint64_t>(s.size()));
  return self_sv.compare(sv);
}

template <std::convertible_to<StringView> T>
PURE HOSTDEV [[nodiscard]] inline auto
String::compare(T const & t) const noexcept -> int
{
  StringView const self_sv(data(), static_cast<uint64_t>(size()));
  StringView const sv(t);
  return self_sv.compare(sv);
}

PURE HOSTDEV [[nodiscard]] inline auto
String::ends_with(StringView sv) const noexcept -> bool
{
  StringView const self_sv(data(), static_cast<uint64_t>(size()));
  return self_sv.ends_with(sv);
}

PURE HOSTDEV [[nodiscard]] inline auto
String::ends_with(char c) const noexcept -> bool
{
  return !empty() && back() == c;
}

PURE HOSTDEV [[nodiscard]] inline auto
String::ends_with(char const * s) const noexcept -> bool
{
  return ends_with(StringView(s));
}

PURE HOSTDEV [[nodiscard]] inline auto
String::starts_with(StringView sv) const noexcept -> bool
{
  StringView const self_sv(data(), static_cast<uint64_t>(size()));
  return self_sv.starts_with(sv);
}

PURE HOSTDEV [[nodiscard]] inline auto
String::starts_with(char c) const noexcept -> bool
{
  return !empty() && front() == c;
}

PURE HOSTDEV [[nodiscard]] inline auto
String::starts_with(char const * s) const noexcept -> bool
{
  return starts_with(StringView(s));
}

PURE HOSTDEV [[nodiscard]] inline auto
String::find_last_of(char c, Int pos) const noexcept -> Int
{
  ASSERT(pos >= 0);
  StringView const self_sv(data(), static_cast<uint64_t>(size()));
  uint64_t const result = self_sv.find_last_of(c, static_cast<uint64_t>(pos));
  return result == StringView::npos ? npos : static_cast<Int>(result);
}

PURE HOSTDEV [[nodiscard]] inline auto
String::substr(Int pos, Int count) const noexcept -> String
{
  return {*this, pos, count};
}

} // namespace um2

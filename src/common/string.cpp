#include <um2/common/string.hpp>

namespace um2
{

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

UM2_HOSTDEV String::String(String const & s)
    : _size{s._size},
      _capacity{bit_ceil(s._size + 1)},
      _data{new char8_t[static_cast<ulen_t>(bit_ceil(s._size + 1))]}
{
  memcpy(this->_data, s._data, static_cast<size_t>(s._size) + 1);
}

// Turn off warning about member initializers since we need to delete _data before
// assigning a new value to it.
// NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
UM2_HOSTDEV String::String(String && s) noexcept
    : _size{s._size},
      _capacity{s._capacity},
      _data{s._data}
{
  s._size = 0;
  s._capacity = 0;
  s._data = nullptr;
}
// NOLINTEND(cppcoreguidelines-prefer-member-initializer)

String::String(std::string const & s)
    : _size{static_cast<len_t>(s.size())},
      _capacity{static_cast<len_t>(bit_ceil(s.size() + 1))},
      _data{new char8_t[bit_ceil(s.size() + 1)]}
{
  memcpy(this->_data, s.data(), s.size() + 1);
}

// --------------------------------------------------------------------------
// Operators
// --------------------------------------------------------------------------

UM2_HOSTDEV auto String::operator=(String const & s) -> String &
{
  if (this != &s) {
    len_t const sizep1 = s._size + 1;
    if (this->_capacity < sizep1) {
      delete[] this->_data;
      this->_capacity = bit_ceil(sizep1);
      this->_data = new char8_t[static_cast<ulen_t>(bit_ceil(sizep1))];
    }
    this->_size = s._size;
    memcpy(this->_data, s._data, static_cast<size_t>(sizep1));
  }
  return *this;
}

UM2_HOSTDEV auto String::operator=(String && s) noexcept -> String &
{
  if (this != &s) {
    delete[] this->_data;
    this->_data = s._data;
    this->_size = s._size;
    this->_capacity = s._capacity;
    s._size = 0;
    s._capacity = 0;
    s._data = nullptr;
  }
  return *this;
}

auto String::operator=(std::string const & s) -> String &
{
  size_t const sizep1 = s.size() + 1;
  if (this->_capacity < static_cast<len_t>(sizep1)) {
    delete[] this->_data;
    this->_capacity = static_cast<len_t>(bit_ceil(sizep1));
    this->_data = new char8_t[bit_ceil(sizep1)];
  }
  this->_size = static_cast<len_t>(s.size());
  memcpy(this->_data, s.data(), s.size() + 1);
  return *this;
}

UM2_PURE UM2_HOSTDEV auto String::operator==(String const & s) const noexcept -> bool
{
  return this->compare(s) == 0;
}

UM2_PURE auto String::operator==(std::string const & s) const noexcept -> bool
{
  return this->compare(reinterpret_cast<char8_t const *>(s.data())) == 0;
}

UM2_PURE UM2_HOSTDEV auto String::operator<(String const & s) const noexcept -> bool
{
  return this->compare(s) < 0;
}

UM2_PURE UM2_HOSTDEV auto String::operator<=(String const & s) const noexcept -> bool
{
  return this->compare(s) <= 0;
}

UM2_PURE UM2_HOSTDEV auto String::operator>(String const & s) const noexcept -> bool
{
  return this->compare(s) > 0;
}

UM2_PURE UM2_HOSTDEV auto String::operator>=(String const & s) const noexcept -> bool
{
  return this->compare(s) >= 0;
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

// Turn off warning about reinterpret_cast since we need to compare char8_t
// as unsigned char/char
#ifdef __CUDA_ARCH__
UM2_PURE UM2_DEVICE auto String::compare(char8_t const * const s) const noexcept -> int
{
  char8_t const * s1 = this->_data;
  char8_t const * s2 = s;
  while (*s1 && (*s1 == *s2)) {
    ++s1;
    ++s2;
  }
  return *reinterpret_cast<unsigned char const *>(s1) -
         *reinterpret_cast<unsigned char const *>(s2);
}
#else
UM2_PURE UM2_HOST auto String::compare(char8_t const * const s) const noexcept -> int
{
  // Reinterpret char8_t as char
  return strcmp(reinterpret_cast<char const *>(this->_data),
                reinterpret_cast<char const *>(s));
}
#endif

UM2_PURE UM2_HOSTDEV auto String::compare(String const & s) const noexcept -> int
{
  return this->compare(s._data);
}

UM2_PURE auto toString(String const & s) -> std::string
{
  auto const * const sdata = reinterpret_cast<char const *>(s.data());
  return {sdata, static_cast<size_t>(s.size())};
}

} // namespace um2

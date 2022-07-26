#pragma once

template<typename T>
struct vec2
{
  T x, y;

  constexpr vec2() = default;
  constexpr vec2(T _x, T _y) : x(_x), y(_y) {}

  template<typename U> 
  constexpr vec2<T>& operator=(vec2<U> const& v)
  {
    x = v.x;
    y = v.y;
    return *this;
  }
};

template<typename T>
constexpr vec2<T> operator+(vec2<T> const& v1, vec2<T> const& v2)
{
  return vec2<T>(v1.x + v2.x, v1.y + v2.y);
}

template<typename T>
constexpr bool operator==(vec2<T> const& v1, vec2<T> const& v2)
{
  return v1.x == v2.x && v1.y == v2.y;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const vec2<T>& v)
{
  os << "(" << v.x << ", " << v.y << ")";
  return os;
}

#pragma once

#include <cstddef>

namespace c10 {


#if defined(__APPLE__) && defined(__MACH__)
struct in_place_t {
  explicit in_place_t() = default;
};

template <std::size_t I>
struct in_place_index_t {
  explicit in_place_index_t() = default;
};

template <typename T>
struct in_place_type_t {
  explicit in_place_type_t() = default;
};

constexpr in_place_t in_place{};
#else
using std::in_place;
using std::in_place_index_t;
using std::in_place_t;
using std::in_place_type_t;
#endif

} // namespace c10

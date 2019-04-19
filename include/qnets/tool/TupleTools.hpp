#ifndef QNETS_TOOL_TUPLETOOLS_HPP
#define QNETS_TOOL_TUPLETOOLS_HPP

#include <functional>

namespace tupl
{
// Collection of helpers for dealing with variadic template parameter packs
// Everything in here is meant for usage at compile-time!
// With fold-expressions from C++17 some functions will become obsolete.

// Template Parameter List (to help with multiple parameter packs per template)
template <typename ...> struct list {};


// count pack and return count as desired integer type
template <typename SizeT, class ... Pack>
constexpr SizeT count() { return static_cast<SizeT>(sizeof...(Pack)); }


// --- Pack Sum

// sum of pack values
template <typename T, T ... ts>
constexpr T sum()
{
    T result = 0;
    for (auto &t : {ts...}) { result += t; }
    return result;
}

} // tupl

#endif

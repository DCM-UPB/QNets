#ifndef QNETS_TOOL_PACKTOOLS_HPP
#define QNETS_TOOL_PACKTOOLS_HPP

#include <functional>
#include <type_traits>

namespace pack
{
// Collection of helpers for dealing with variadic template parameter packs
// Everything in here is meant for usage at compile-time!
// With fold-expressions from C++17 some functions will become obsolete.

// Template Parameter List (to help with multiple parameter packs per template)
template <typename ...>
struct list {};

// count pack and return count as desired integer type
template <typename SizeT, class ... Pack>
constexpr SizeT count(Pack ... p) { return static_cast<SizeT>(sizeof...(Pack)); }

// --- Pack Sum

// sum of pack values, with start and end
template <typename SizeT, typename T, T ... ts>
constexpr T sum(SizeT begin_index = 0/*count from*/, SizeT end_index = count<SizeT>(ts...)/*to before this*/)
{
    //static_assert(end_index <= count<SizeT>(ts...), "[pack::sum] end_index > count(pack).");
    T result = 0;
    SizeT i = 0;
    for (auto &t : {ts...}) {
        if (i >= begin_index && i < end_index) {
            result += t;
        }
        ++i;
    }
    return result;
}

// --- Pack Product

// Product of pack values, with optional start and end
template <typename SizeT, typename T, T ... ts>
constexpr T prod(SizeT begin_index = 0/*count from*/, SizeT end_index = count<SizeT>(ts...)/*to before this*/)
{
    //static_assert(end_index <= count<SizeT>(ts...), "[pack::prod] end_index > count(pack).");
    T result = 1;
    SizeT i = 0;
    for (auto &t : {ts...}) {
        if (i >= begin_index && i < end_index) {
            result *= t;
        }
        ++i;
    }
    return result;
}

// --- Accumulate Function

// Accumulate function of pack values, with optional start and end
template <class FuncT, typename SizeT, typename T, T ... ts>
constexpr auto accumulate(FuncT func, SizeT begin_index = 0/*count from*/, SizeT end_index = count<SizeT>(ts...)/*to before this*/)
{
    //static_assert(end_index <= count<SizeT>(ts...), "[pack::accumulate] end_index > count(pack).");
    decltype(func) result = 0;
    SizeT i = 0;
    for (const auto &t : {ts...}) {
        if (i >= begin_index && i < end_index) {
            result += func(t);
        }
        ++i;
    }
    return result;
}
} // pack

#endif

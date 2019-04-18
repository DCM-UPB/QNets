#ifndef QNETS_TOOL_PACKTOOLS_HPP
#define QNETS_TOOL_PACKTOOLS_HPP

#include <functional>

namespace pack
{
// Collection of helpers for dealing with variadic template parameter packs
// Everything in here is meant for usage at compile-time!

// Template Parameter List (to allow multiple parameter packs per template)
template <typename ...>
struct list {};


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

// sum of pack values, with start and end
template <typename SizeT, typename T, T ... ts>
constexpr T sum(SizeT begin_index/*count from*/, SizeT end_index/*to before this*/)
{
    static_assert(end_index <= count<SizeT, ts...>(), "[pack::sum] end_index > count(pack).");
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

// product of pack values
template <typename T, T ... ts>
constexpr T prod()
{
    T result = 1;
    for (auto &t : {ts...}) { result *= t; }
    return result;
}

// product of pack values, with start and end
template <typename SizeT, typename T, T ... ts>
constexpr T prod(SizeT begin_index/*count from*/, SizeT end_index/*to before this*/)
{
    static_assert(end_index <= count<SizeT, ts...>(), "[pack::prod] end_index > count(pack).");
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

// accumulate function of pack values
template <typename ValueT, typename T, T ... ts>
constexpr ValueT accumulate(std::function<ValueT(const T&)> func)
{
    ValueT result = 0;
    for (auto &t : {ts...}) { result += func(t); }
    return result;
}

// accumulate function of pack values, with start and end
template <typename SizeT, typename ValueT, typename T, T ... ts>
constexpr ValueT accumulate(SizeT begin_index/*count from*/, SizeT end_index/*to before this*/, std::function<ValueT(const T&)> func)
{
    static_assert(end_index <= count<SizeT, ts...>(), "[pack::accumulate] end_index > count(pack).");
    ValueT result = 0;
    SizeT i = 0;
    for (auto &t : {ts...}) {
        if (i >= begin_index && i < end_index) {
            result += func(t);
        }
        ++i;
    }
    return result;
}

} // pack

#endif

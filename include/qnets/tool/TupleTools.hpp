#ifndef QNETS_TOOL_TUPLETOOLS_HPP
#define QNETS_TOOL_TUPLETOOLS_HPP

#include <utility>
#include <tuple>

namespace tupl
{
// Collection of helpers for dealing with tuples

// count tuple and return count as desired integer type
template <typename SizeT, class TupleT>
constexpr SizeT count() { return static_cast<SizeT>(std::tuple_size<TupleT>::value); }

// --- apply function to all

template <class TupleT, class FuncT, std::size_t ... Is>
constexpr auto apply_impl(TupleT &t, FuncT f, std::index_sequence<Is...>)
{
    return f(std::get<Is>(t)...);
}

template <class TupleT, class FuncT>
constexpr auto apply(TupleT &t, FuncT f)
{
    return apply_impl(t, f, std::make_index_sequence<std::tuple_size<TupleT>::value>{});
}

// --- create container with unary function return values

template <class ContT, class TupleT, class FuncT, std::size_t ... Is>
constexpr auto make_fcont_impl(TupleT &t, FuncT f, std::index_sequence<Is...>)
{
    return ContT{f(std::get<Is>(t))...};
}

template <class ContT, class TupleT, class FuncT>
constexpr auto make_fcont(TupleT &t, FuncT f)
{
    return make_fcont_impl<ContT>(t, f, std::make_index_sequence<std::tuple_size<TupleT>::value>{});
}
} // tupl

#endif

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

namespace detail
{
template <class TupleT, class FuncT, size_t ... Is>
constexpr auto apply_impl(TupleT &t, FuncT f, std::index_sequence<Is...>)
{
    return f(std::get<Is>(t)...);
}
} // detail

template <class TupleT, class FuncT>
constexpr auto apply(TupleT &t, FuncT f)
{
    return detail::apply_impl(t, f, std::make_index_sequence<std::tuple_size<TupleT>::value>{});
}

// --- accumulate unary function

namespace detail // recursive accumulation helper
{
template <class TupleT, class FuncT>
constexpr auto accumulate_impl(TupleT &/*t*/, FuncT /*f*/, std::index_sequence<>)
{
    return 0; // terminate recursion
}

template <class TupleT, class FuncT, size_t I, size_t ... Is>
constexpr auto accumulate_impl(TupleT &t, FuncT f, std::index_sequence<I, Is...>)
{
    return f(std::get<I>(t)) + accumulate_impl(t, f, std::index_sequence<Is...>{});
}
} // detail

template <class TupleT, class FuncT>
constexpr auto accumulate(TupleT &t, FuncT f)
{
    return detail::accumulate_impl(t, f, std::make_index_sequence<std::tuple_size<TupleT>::value>{});
}

// --- create container filled with unary function return values

namespace detail
{
template <class ContT, class TupleT, class FuncT, size_t ... Is>
constexpr auto make_fcont_impl(TupleT &t, FuncT f, std::index_sequence<Is...>)
{
    return ContT{f(std::get<Is>(t))...};
}
} // detail

template <class ContT, class TupleT, class FuncT>
constexpr auto make_fcont(TupleT &t, FuncT f)
{
    return detail::make_fcont_impl<ContT>(t, f, std::make_index_sequence<std::tuple_size<TupleT>::value>{});
}
} // tupl

#endif

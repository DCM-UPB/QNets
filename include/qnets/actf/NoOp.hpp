#ifndef QNETS_ACTF_NOOP_HPP
#define QNETS_ACTF_NOOP_HPP

#include <cmath>
#include <algorithm>

namespace actf
{
class NoOp // Identity/No-Op Activation Function (Array ACTF)
{
public:
    template <typename ValueT>
    constexpr void f(ValueT begin[], const ValueT * end) {/*no change*/}

    template <typename ValueT>
    constexpr void fd1(ValueT begin[], const ValueT * end, ValueT d1[])
    {
        std::fill(d1, d1 + (end - begin), 1.);
    }

    template <typename ValueT>
    constexpr void fd12(ValueT begin[], const ValueT * end, ValueT d1[], ValueT d2[])
    {
        std::fill(d1, d1 + (end - begin), 1.);
        std::fill(d2, d2 + (end - begin), 0.);
    }
};
} // actf

#endif

#ifndef QNETS_ACTF_EXP_HPP
#define QNETS_ACTF_EXP_HPP

#include <cmath>

namespace actf
{
class Exp // Exponential Activation Function (Array ACTF)
{
public:
    template <typename ValueT>
    constexpr void f(ValueT begin[], const ValueT * end)
    {
        for (; begin < end; ++begin) {
            *begin = exp(*begin);
        }
    }

    template <typename ValueT>
    constexpr void fd1(ValueT begin[], const ValueT * end, ValueT d1[])
    {
        for (; begin < end; ++begin, ++d1) {
            *begin = exp(*begin);
            *d1 = *begin;
        }
    }

    template <typename ValueT>
    constexpr void fd12(ValueT begin[], const ValueT * end, ValueT d1[], ValueT d2[])
    {
        for (; begin < end; ++begin, ++d1, ++d2) {
            *begin = exp(*begin);
            *d1 = *begin;
            *d2 = *begin;
        }
    }
};
} // actf

#endif

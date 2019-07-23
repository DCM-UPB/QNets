#ifndef QNETS_ACTF_SRLU_HPP
#define QNETS_ACTF_SRLU_HPP

#include <cmath>

namespace actf
{
class SRLU // Smooth Rectified Linear Unit (also known as Softplus)
{
public:
    template <typename ValueT>
    constexpr void f(ValueT begin[], const ValueT * end)
    {
        for (; begin < end; ++begin) {
            *begin = log1p(exp(*begin)); // log(1+e^x)
        }
    }

    template <typename ValueT>
    constexpr void fd1(ValueT begin[], const ValueT * end, ValueT d1[])
    {
        for (; begin < end; ++begin, ++d1) {
            *d1 = 1./(1. + exp(-(*begin))); // 1 / (1+e^-x)
            *begin = log1p(exp(*begin)); // log(1+e^x)
        }
    }

    template <typename ValueT>
    constexpr void fd12(ValueT begin[], const ValueT * end, ValueT d1[], ValueT d2[])
    {
        for (; begin < end; ++begin, ++d1, ++d2) {
            const double etimesmx = exp(-(*begin));
            const double etimesmx1 = etimesmx + 1.;
            *begin = log1p(exp(*begin)); // log(1+e^x)
            *d1 = 1./etimesmx1; // 1 / (1+e^-x)
            *d2 = etimesmx / (etimesmx1*etimesmx1); // e^-x/(1+e^-x)^2
        }
    }
};
} // actf

#endif

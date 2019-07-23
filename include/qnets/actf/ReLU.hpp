#ifndef QNETS_ACTF_RELU_HPP
#define QNETS_ACTF_RELU_HPP

#include <cmath>

namespace actf
{
class ReLU // Rectified Linear Activation Function (Array ACTF)
{
public:
    template <typename ValueT>
    constexpr void f(ValueT begin[], const ValueT * end)
    {
        for (; begin < end; ++begin) {
            if (*begin < 0.) {
                *begin = 0.;
            }
        }
    }

    template <typename ValueT>
    constexpr void fd1(ValueT begin[], const ValueT * end, ValueT d1[])
    {
        for (; begin < end; ++begin, ++d1) {
            if (*begin > 0.) {
                *d1 = 1.;
            }
            else {
                *begin = 0.;
                *d1 = 0.;
            }
        }
    }

    template <typename ValueT>
    constexpr void fd12(ValueT begin[], const ValueT * end, ValueT d1[], ValueT d2[])
    {
        for (; begin < end; ++begin, ++d1, ++d2) {
            if (*begin > 0.) {
                *d1 = 1.;
            }
            else {
                *begin = 0.;
                *d1 = 0.;
            }
            *d2 = 0.;
        }
    }
};
} // actf

#endif

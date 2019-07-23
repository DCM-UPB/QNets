#ifndef QNETS_ACTF_SINE_HPP
#define QNETS_ACTF_SINE_HPP

#include <cmath>

namespace actf
{
class Sine // Sine Activation Function
{
public:
    template <typename ValueT>
    constexpr void f(ValueT begin[], const ValueT * end)
    {
        for (; begin < end; ++begin) {
            *begin = sin(*begin);
        }
    }

    template <typename ValueT>
    constexpr void fd1(ValueT begin[], const ValueT * end, ValueT d1[])
    {
        for (; begin < end; ++begin, ++d1) {
            *d1 = cos(*begin);
            *begin = sin(*begin);
        }
    }

    template <typename ValueT>
    constexpr void fd12(ValueT begin[], const ValueT * end, ValueT d1[], ValueT d2[])
    {
        for (; begin < end; ++begin, ++d1, ++d2) {
            *d1 = cos(*begin);
            *d2 = -sin(*begin);
            *begin = sin(*begin);
        }
    }
};
} // actf

#endif

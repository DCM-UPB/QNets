#ifndef QNETS_ACTF_SIGMOID_HPP
#define QNETS_ACTF_SIGMOID_HPP

#include <cmath>

namespace actf
{
class Sigmoid // Sigmoid Activation Function (Array ACTF)
{
public:
    template <typename ValueT>
    constexpr void f(ValueT begin[], const ValueT * end)
    {
        for (; begin < end; ++begin) {
            *begin = 1./(1. + exp(-(*begin)));
        }
    }

    template <typename ValueT>
    constexpr void fd1(ValueT begin[], const ValueT * end, ValueT d1[])
    {
        for (; begin < end; ++begin, ++d1) {
            *begin = 1./(1. + exp(-(*begin))); // f
            *d1 = *begin*(1. - *begin); // fd1
        }
    }

    template <typename ValueT>
    constexpr void fd12(ValueT begin[], const ValueT * end, ValueT d1[], ValueT d2[])
    {
        for (; begin < end; ++begin, ++d1, ++d2) {
            *begin = 1./(1. + exp(-(*begin))); // f
            *d1 = *begin*(1. - *begin); // fd1
            *d2 = *d1*(1. - 2.*(*begin)); // fd2
        }
    }
};
} // actf

#endif

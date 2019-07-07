#ifndef QNETS_ACTF_TANSIG_HPP
#define QNETS_ACTF_TANSIG_HPP

#include <cmath>

namespace actf
{
class TanSig // TanSigmoid Activation Function (Array ACTF)
{
public:
    template <typename ValueT>
    constexpr void f(ValueT begin[], const ValueT * end)
    {
        for (; begin < end; ++begin) {
            *begin = 2.0/(1.0 + exp(-2.0*(*begin))) - 1.0;
        }
    }

    template <typename ValueT>
    constexpr void fd1(ValueT begin[], const ValueT * end, ValueT d1[])
    {
        for (; begin < end; ++begin, ++d1) {
            const double expf = exp(-2.0*(*begin));
            const double quot = 2.0/(1.0 + expf);
            *begin = quot - 1.; // f
            *d1 = expf * quot * quot; // d1 = 4 * exp(-2*in) / (1 + exp(-2*in))^2
        }
    }

    template <typename ValueT>
    constexpr void fd12(ValueT begin[], const ValueT * end, ValueT d1[], ValueT d2[])
    {
        for (; begin < end; ++begin, ++d1, ++d2) {
            const double expf = exp(-2.0*(*begin));
            const double quot = 2.0/(1.0 + expf);
            const double prod = expf*quot;
            *begin = quot - 1.; // f
            *d1 = expf * quot * quot; // d1
            *d2 = 2.0*prod*quot*(prod - 1.0);  // d2 = 8 * exp(-2*in) / (1 + exp(-2*in))^2 * (2 * exp(-2*in) / (1 + exp(-2*in)) - 1)
        }
    }
};
} // actf

#endif

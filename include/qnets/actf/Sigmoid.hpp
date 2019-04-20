#ifndef QNETS_ACTF_SIGMOID_HPP
#define QNETS_ACTF_SIGMOID_HPP

#include <cmath>

namespace actf
{
class Sigmoid // Test Generic Array ACTF
{
public:

    template <class InputIt, class OutputIt>
    constexpr void f(InputIt cur, const InputIt end, OutputIt out)
    {
        for (; cur < end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur)));
        }
    }

    template <class InputIt, class OutputIt>
    constexpr void fd1(InputIt cur, const InputIt end, OutputIt outf, OutputIt outfd1)
    {
        for (; cur < end; ++cur, ++outf, ++outfd1) {
            *outf = 1./(1. + exp(-(*cur))); // f
            *outfd1 = *outf*(1. - *outf); // fd1
        }
    }

    template <class InputIt, class OutputIt>
    constexpr void fd12(InputIt cur, const InputIt end, OutputIt outf, OutputIt outfd1, OutputIt outfd2)
    {
        for (; cur < end; ++cur, ++outf, ++outfd1, ++outfd2) {
            *outf = 1./(1. + exp(-(*cur))); // f
            *outfd1 = *outf*(1. - *outf); // fd1
            *outfd2 = *outfd1*(1. - 2.*(*outf)); // fd2
        }
    }
};
} // actf

#endif

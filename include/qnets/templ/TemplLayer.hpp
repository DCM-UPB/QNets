#ifndef QNETS_TEMPL_TEMPLLAYER_HPP
#define QNETS_TEMPL_TEMPLLAYER_HPP

#include "qnets/templ/DerivConfig.hpp"

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace templ
{
// --- TemplNet Layers

// Layer Config
//
// To pass non-input layer configurations as variadic parameter pack
template <int N_OUT, class ACTFType>
struct LayerConfig
{
    static constexpr int noutput = N_OUT;
    using ACTF_Type = ACTFType;

    static constexpr int size() { return noutput; }
};


// The actual Layer class
//
template <typename ValueT, int NET_NINPUT, int NET_NOUTPUT, int NBETA_NEXT, int N_IN, int N_OUT, class ACTFType, DerivConfig DCONF>
class TemplLayer: public LayerConfig<N_OUT, ACTFType>
{
public:
    // N_IN dependent sizes
    static constexpr int ninput = N_IN;
    static constexpr int nbeta = (N_IN + 1)*N_OUT;
    static constexpr int net_nin = NET_NINPUT;
    static constexpr int net_nout = NET_NOUTPUT;

    // Sizes which also depend on DCONF
    static constexpr StaticDFlags<DCONF> dconf{};
    static constexpr int nd1 = dconf.d1 ? NET_NINPUT*N_OUT : 0; // number of input derivative values
    static constexpr int nd1_prev = dconf.d1 ? NET_NINPUT*N_IN : 0; // number of deriv values from previous layer

    static constexpr int nd2 = dconf.d2 ? nd1 : 0;
    static constexpr int nd2_prev = dconf.d2 ? nd1_prev : 0;

    static_assert(NBETA_NEXT%(1 + N_OUT) == 0, ""); // -> BUG!
    static constexpr int nout_next = NBETA_NEXT/(1 + N_OUT);
    static constexpr int nad1 = dconf.d1 || dconf.vd1 ? N_OUT : 0;
    static constexpr int nad1_next = dconf.d1 || dconf.vd1 ? nout_next : 0;
    static constexpr int nad2 = dconf.d2 ? N_OUT : 0;
    static constexpr int nad2_next = dconf.d2 ? nout_next : 0;

    static constexpr int nvd1 = (dconf.d1 || dconf.vd1) ? NET_NOUTPUT*N_OUT : 0; // number of backprop grad values
    static constexpr int nvd1_next = (dconf.d1 || dconf.vd1)
                                     ? NET_NOUTPUT*nout_next
                                     : 0; // number of backprop grad values from previous layer

    static constexpr int nvd2 = dconf.d2
                                ? NET_NOUTPUT*N_OUT
                                : 0; // number of approximate diagonal second order backprop grad values
    static constexpr int nvd2_next = dconf.d2
                                     ? NET_NOUTPUT*nout_next
                                     : 0;

    static constexpr int nbd2 = dconf.d2 ? NET_NOUTPUT*N_OUT*N_OUT : 0; // number of second order backprop grad values
    static constexpr int nbd2_next = dconf.d2
                                     ? NET_NOUTPUT*nout_next*nout_next
                                     : 0;

private: // arrays
    std::array<ValueT, N_OUT> _out{};
    // the deriv arrays could be quite large, so we need to heap allocate
    const std::unique_ptr<std::array<ValueT, nd1>> _d1_ptr{std::make_unique<std::array<ValueT, nd1>>()};
    const std::unique_ptr<std::array<ValueT, nd2>> _d2_ptr{std::make_unique<std::array<ValueT, nd2>>()};
    const std::unique_ptr<std::array<ValueT, nvd1>> _vd1_ptr{std::make_unique<std::array<ValueT, nvd1>>()};
    const std::unique_ptr<std::array<ValueT, nvd2>> _vd2_ptr{std::make_unique<std::array<ValueT, nvd2>>()};
    const std::unique_ptr<std::array<ValueT, nbd2>> _bd2_ptr{std::make_unique<std::array<ValueT, nbd2>>()};

    std::array<ValueT, nad1> _ad1{}; // activation function d1
    std::array<ValueT, nad2> _ad2{}; // activation function d2

public: // public member variables
    ACTFType actf{}; // the activation function
    std::array<ValueT, nbeta> beta{}; // the weights

    // public const output references
    constexpr const std::array<ValueT, N_OUT> &out() const { return _out; }
    constexpr const std::array<ValueT, nd1> &d1() const { return *_d1_ptr; }
    constexpr const std::array<ValueT, nd2> &d2() const { return *_d2_ptr; }
    constexpr const std::array<ValueT, nvd1> &vd1() const { return *_vd1_ptr; }
    constexpr const std::array<ValueT, nvd2> &vd2() const { return *_vd2_ptr; }
    constexpr const std::array<ValueT, nbd2> &bd2() const { return *_bd2_ptr; }

    constexpr const std::array<ValueT, nad1> &ad1() const { return _ad1; }
    constexpr const std::array<ValueT, nad2> &ad2() const { return _ad2; };

private:
    constexpr void _computeFeed(const ValueT input[])
    {
        int beta_i0 = 1; // increments through the indices of the first non-offset beta per unit
        for (int i = 0; i < N_OUT; ++i, beta_i0 += N_IN + 1) {
            _out[i] = std::inner_product(input, input + ninput, beta.begin() + beta_i0, beta[beta_i0 - 1]/*bias weight*/); // found to be faster than loop
        }
    }

    constexpr void _computeActivation(bool flag_ad1, bool flag_ad2 /*is overriding*/)
    {
        if (flag_ad2) {
            actf.fd12(_out.begin(), _out.end(), _out.begin(), _ad1.begin(), _ad2.begin());
        }
        else if (flag_ad1) {
            actf.fd1(_out.begin(), _out.end(), _out.begin(), _ad1.begin());
        }
        else {
            actf.f(_out.begin(), _out.end(), _out.begin());
        }
    }

    constexpr void _computeOutput(const ValueT input[], DynamicDFlags dflags)
    {
        this->_computeFeed(input);
        this->_computeActivation(dflags.d1() || dflags.vd1(), dflags.d2());
    }

    constexpr void _computeD1_Layer(const ValueT in_d1[])
    {
        auto &D1 = *_d1_ptr;
        D1.fill(0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1);
            const int d_i0 = i*NET_NINPUT;
            for (int j = 0; j < N_IN; ++j) {
                for (int k = 0; k < NET_NINPUT; ++k) {
                    D1[d_i0 + k] += beta[beta_i0 + j]*in_d1[j*NET_NINPUT + k];
                }
            }
            for (int l = d_i0; l < d_i0 + NET_NINPUT; ++l) {
                D1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeD1_Input() // i.e. in_d1[i][i] = 1., else 0
    {
        auto &D1 = *_d1_ptr;
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(NET_NINPUT + 1);
            for (int j = 0; j < NET_NINPUT; ++j) {
                D1[i*NET_NINPUT + j] = _ad1[i]*beta[beta_i0 + j];
            }
        }
    }

    constexpr void _computeD12_Layer(const ValueT in_d1[], const ValueT in_d2[])
    {
        auto &D1 = *_d1_ptr;
        auto &D2 = *_d2_ptr;
        D1.fill(0.);
        D2.fill(0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1);
            const int d_i0 = i*NET_NINPUT;
            for (int j = 0; j < N_IN; ++j) {
                for (int k = 0; k < NET_NINPUT; ++k) {
                    D1[d_i0 + k] += beta[beta_i0 + j]*in_d1[j*NET_NINPUT + k];
                    D2[d_i0 + k] += beta[beta_i0 + j]*in_d2[j*NET_NINPUT + k];
                }
            }
            for (int l = d_i0; l < d_i0 + NET_NINPUT; ++l) {
                D2[l] = _ad1[i]*D2[l] + _ad2[i]*D1[l]*D1[l];
                D1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeD12_Input()
    {
        auto &D1 = *_d1_ptr;
        auto &D2 = *_d2_ptr;
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(NET_NINPUT + 1);
            for (int j = 0; j < NET_NINPUT; ++j) {
                D1[i*NET_NINPUT + j] = _ad1[i]*beta[beta_i0 + j];
                D2[i*NET_NINPUT + j] = _ad2[i]*beta[beta_i0 + j]*beta[beta_i0 + j];
            }
        }
    }


    constexpr void _forwardInput(const ValueT input[], DynamicDFlags dflags)
    {
        // statically secure this call (i.e. using it on non-input layer will not compile)
        static_assert(N_IN == NET_NINPUT, "[TemplLayer::ForwardInput] N_IN != NET_NINPUT");

        dflags = dflags.AND(dconf); // AND static and dynamic conf
        this->_computeOutput(input, dflags);

        // fill diagonal d1,d2
        if (dflags.d2()) {
            this->_computeD12_Input();
        }
        else if (dflags.d1()) {
            this->_computeD1_Input();
        }
    }


    constexpr void _forwardLayer(const ValueT input[], const ValueT in_d1[], const ValueT in_d2[], DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        this->_computeOutput(input, dflags);

        // input derivs
        if (dflags.d2()) {
            this->_computeD12_Layer(in_d1, in_d2);
        }
        else if (dflags.d1()) {
            this->_computeD1_Layer(in_d1);
        }
    }


    constexpr void _backwardOutput(DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        static_assert(N_OUT == NET_NOUTPUT, "[TemplLayer::BackwardOutput] nout_next != NET_NOUTPUT");
        auto &VD1 = *_vd1_ptr;
        auto &VD2 = *_vd2_ptr;
        //auto &BD2 = *_bd2_ptr;

        VD1.fill(0.);
        VD2.fill(0.);
        //BD2.fill(0.);
        if (!dflags.d1() && !dflags.vd1()) { return; }
        for (int i = 0; i < NET_NOUTPUT; ++i) {
            VD1[i*NET_NOUTPUT + i] = 1.;//_ad1[i];
        }

        if (!dflags.d2()) { return; }
        for (int i = 0; i < NET_NOUTPUT; ++i) {
            VD2[i*NET_NOUTPUT + i] = 1.;//_ad2[i];
        }
        /*for (int i = 0; i < NET_NOUTPUT; ++i) {
            BD2[i*NET_NOUTPUT*NET_NOUTPUT + i*NET_NOUTPUT + i] = _ad2[i];
        }*/
    }

    constexpr void _backwardLayerD1(const ValueT vd1_next[], const ValueT ad1_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        auto &VD1 = *_vd1_ptr;
        VD1.fill(0.);
        if (!dflags.d1() && !dflags.vd1()) { return; }

        for (int i = 0; i < NET_NOUTPUT; ++i) {
            for (int j = 0; j < nout_next; ++j) {
                const ValueT jfac = ad1_next[j]*vd1_next[i*nout_next + j];
                const int beta_i0 = 1 + j*(N_OUT + 1);
                for (int k = 0; k < N_OUT; ++k) {
                    VD1[i*N_OUT + k] += beta_next[beta_i0 + k]*jfac;
                }
            }
        }
    }

    constexpr void _backwardLayerD2(const ValueT vd1_next[], const ValueT vd2_next[]/*, const ValueT bd2_next[],*/, const ValueT ad1_next[], const ValueT ad2_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        auto &VD1 = *_vd1_ptr;
        auto &VD2 = *_vd2_ptr;
        //auto &BD2 = *_bd2_ptr;
        VD1.fill(0.);
        VD2.fill(0.);
        //BD2.fill(0.);
        if (!dflags.d1() && !dflags.vd1()) { return; }

        for (int i = 0; i < NET_NOUTPUT; ++i) {
            for (int j = 0; j < nout_next; ++j) {
                const ValueT jfac1 = ad1_next[j]*vd1_next[i*nout_next + j];
                const ValueT jfac2 = dflags.d2() ? ad1_next[j]*ad1_next[j]*vd2_next[i*nout_next + j] + ad2_next[j]*vd1_next[i*nout_next + j] : 0;
                const int beta_i0 = 1 + j*(N_OUT + 1);
                for (int k = 0; k < N_OUT; ++k) {
                    const ValueT bjk = beta_next[beta_i0 + k];
                    VD1[i*N_OUT + k] += bjk*jfac1;
                    if (dflags.d2()) {
                        VD2[i*N_OUT + k] += bjk*bjk*jfac2;
                    }
                }
            }
            /*if (dflags.d2()) {
                const int bd_i0 = i*N_OUT*N_OUT;
                for (int j = 0; j < nout_next; ++j) {
                    for (int k = 0; k < N_OUT; ++k) {
                        const int bd_i0_k = bd_i0 + k*N_OUT;
                        const ValueT bjk = beta_next[1 + j*(N_OUT + 1) + k];
                        for (int l = 0; l < N_OUT; ++l) {
                            BD2[bd_i0_k + l] += bjk*bjk*vd2_next[i*nout_next + j];

                        }
                    }
            }*/
            /*
            if (dflags.d2()) {
                for (int k = 0; k < N_OUT; ++k) {
                    VD2[d_i0 + k] = _ad1[k]*_ad1[k]*VD2[d_i0 + k] + _ad2[k]*VD1[d_i0 + k];
                }
            }
            for (int k = 0; k < N_OUT; ++k) {
                VD1[d_i0 + k] *= _ad1[k];
            }*/
        }
    }

    constexpr void _layerGrad(const ValueT input[], ValueT vd1_block[], const int iout, DynamicDFlags dflags) const
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        if (!dflags.vd1()) { return; }
        const auto VD1_iout = (*_vd1_ptr).begin() + iout*N_OUT;

        for (int j = 0; j < N_OUT; ++j) {
            const ValueT jfac = _ad1[j]*VD1_iout[j];
            *vd1_block++ = jfac; // bias weight gradient
            for (int k = 0; k < N_IN; ++k, ++vd1_block) {
                *vd1_block = input[k]*jfac;
            }
        }
    }

    constexpr void _layerGrad2(const ValueT input[], ValueT vd2_block[], const int iout, DynamicDFlags dflags) const
    {
        const auto VD1_iout = (*_vd1_ptr).begin() + iout*N_OUT;
        const auto VD2_iout = (*_vd2_ptr).begin() + iout*N_OUT;

        for (int j = 0; j < N_OUT; ++j) {
            const ValueT jfac = _ad1[j]*_ad1[j]*VD2_iout[j] + _ad2[j]*VD1_iout[j];
            *vd2_block++ = jfac; // bias weight gradient
            for (int k = 0; k < N_IN; ++k, ++vd2_block) {
                *vd2_block = input[k]*input[k]*jfac;
            }
        }
    }

    constexpr void _inputGrad(ValueT d1_out[], ValueT d2_out[], DynamicDFlags dflags) const
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        if (!dflags.d1()) { return; }
        std::fill(d1_out, d1_out + NET_NOUTPUT*N_IN, 0.);
        if (dflags.d2()) {
            std::fill(d2_out, d2_out + NET_NOUTPUT*N_IN, 0.);
        }
        const auto &VD1 = *_vd1_ptr;
        const auto &VD2 = *_vd2_ptr;

        for (int i = 0; i < NET_NOUTPUT; ++i) {
            for (int j = 0; j < N_OUT; ++j) {
                const ValueT jfac1 = _ad1[j]*VD1[i*N_OUT + j];
                const ValueT jfac2 = dflags.d2() ? _ad1[j]*_ad1[j]*VD2[i*N_OUT + j] + _ad2[j]*VD1[i*N_OUT + j] : 0;
                for (int k = 0; k < N_IN; ++k) {
                    const ValueT bjk = beta[1 + j*(N_IN + 1) + k];
                    d1_out[i*N_IN + k] += bjk*jfac1;
                    if (dflags.d2()) {
                        d2_out[i*N_IN + k] += bjk*bjk*jfac2;
                    }
                }
            }
            /*if (dflags.d2()) {
                for (int j = 0; j < N_OUT; ++j) {
                    for (int k = 0; k < N_IN; ++k) {
                        const ValueT bjk = beta[1 + j*(N_IN + 1) + k];
                        //d2_out[i*N_IN + k] += bjk*bjk*(VD2[i*N_OUT + j] - VD1[i*N_OUT + j] / _ad1[j] * _ad2[j])/_ad1[j];
                        const ValueT vd1_recon = VD1[i*N_OUT + j]/_ad1[j];
                        const ValueT vd2_recon = (VD2[i*N_OUT + j] - VD1[i*N_OUT + j]/_ad1[j]*_ad2[j])/_ad1[j]/_ad1[j];
                        //d2_out[i*N_IN + k] += bjk * _ad1[j] * vd2_recon + vd1_sum * bjk * bjk * _ad2[j];
                        //d2_out[i*N_IN + k] += bjk * VD2[i*N_OUT + j] + VD1[i*N_OUT + j] * bjk * bjk;
                        d2_out[i*N_IN + k] += vd2_recon*bjk*bjk*_ad1[j]*_ad1[j] + vd1_recon*_ad2[j]*bjk*bjk;
                    }
                }
            }*/
        }
    }

public: // public propagate methods
    // We support 2 different ways to provide arrays for propagate calls:
    // Array: Bounds statically checked due to type system
    // Pointer: No bounds checking


    // --- Propagation of input data (not layer)

    constexpr void ForwardInput(const std::array<ValueT, NET_NINPUT> &input, DynamicDFlags dflags)
    {
        _forwardInput(input.begin(), dflags);
    }

    constexpr void ForwardInput(const ValueT input[], DynamicDFlags dflags)
    {
        _forwardInput(input, dflags);
    }


    // --- Forward Propagation of layer data

    constexpr void ForwardLayer(const std::array<ValueT, N_IN> &input, const std::array<ValueT, nd1_prev> &in_d1, const std::array<ValueT, nd2_prev> &in_d2, DynamicDFlags dflags)
    {
        _forwardLayer(input.begin(), in_d1.begin(), in_d2.begin(), dflags);
    }

    constexpr void ForwardLayer(const ValueT input[], const ValueT in_d1[], const ValueT in_d2[], DynamicDFlags dflags)
    {
        _forwardLayer(input, in_d1, in_d2, dflags);
    }


    // --- Backward Propagation for an output layer

    constexpr void BackwardOutput(DynamicDFlags dflags)
    {
        _backwardOutput(dflags);
    }


    // --- Backward Propagation for a hidden layer

    constexpr void BackwardLayer(const std::array<ValueT, nvd1_next> &vd1_next, const std::array<ValueT, nvd2_next> &vd2_next,
            const std::array<ValueT, nad1_next> &ad1_next, const std::array<ValueT, nad2_next> &ad2_next, const std::array<ValueT, NBETA_NEXT> &beta_next, DynamicDFlags dflags)
    {
        _backwardLayerD2(vd1_next.begin(), vd2_next.begin(), ad1_next.begin(), ad2_next.begin(), beta_next.begin(), dflags);
    }

    /*constexpr void BackwardLayerD2(const ValueT vd1_next[], const ValueT vd2_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        _backwardLayer(vd1_next, vd2_next, beta_next, dflags);
    }*/


    // --- Calculate weight gradient block of output unit iout with respect to this layers' weights

    constexpr void storeLayerGradients1(const std::array<ValueT, N_IN> &input, std::array<ValueT, nbeta> &vd1_block, const int iout, DynamicDFlags dflags) const
    {
        _layerGrad(input.begin(), vd1_block.begin(), iout, dflags);
    }

    constexpr void storeLayerGradients2(const std::array<ValueT, N_IN> &input, std::array<ValueT, nbeta> &vd2_block, const int iout, DynamicDFlags dflags) const
    {
        _layerGrad2(input.begin(), vd2_block.begin(), iout, dflags);
    }
    constexpr void storeLayerGradients(const std::array<ValueT, N_IN> &input, std::array<ValueT, nbeta> &vd1_block, std::array<ValueT, nbeta> &vd2_block, const int iout, DynamicDFlags dflags) const
    {
        storeLayerGradients1(input, vd1_block, iout, dflags);
        storeLayerGradients2(input, vd2_block, iout, dflags);
    }

    constexpr void storeLayerGradients(const ValueT input[], ValueT vd1_block[], const int iout, DynamicDFlags dflags) const
    {
        _layerGrad(input, vd1_block, iout, dflags);
    }

    // --- Calculate input gradient block of output unit iout with respect to this layers inputs

    constexpr void storeInputGradients(std::array<ValueT, NET_NOUTPUT*N_IN> &d1_out, std::array<ValueT, NET_NOUTPUT*N_IN> &d2_out, DynamicDFlags dflags) const
    {
        _inputGrad(d1_out.begin(), d2_out.begin(), dflags);
    }

    constexpr void storeInputGradients(ValueT d1_out[], ValueT d2_out[], DynamicDFlags dflags) const
    {
        _inputGrad(d1_out, d2_out, dflags);
    }
};
} // templ

#endif

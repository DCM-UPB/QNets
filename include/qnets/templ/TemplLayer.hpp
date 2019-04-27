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

    static constexpr int nd2 = dconf.d2
                               ? NET_NINPUT*N_OUT
                               : 0; // number of forward-accumulated first/second order input derivative values
    static constexpr int nd2_prev = dconf.d2 ? NET_NINPUT*N_IN : 0; // the same number of previous layer

    static_assert(NBETA_NEXT%(1 + N_OUT) == 0, ""); // -> BUG!
    static constexpr int nout_next = NBETA_NEXT/(1 + N_OUT);
    static constexpr int nad1 = dconf.needsAny() ? N_OUT : 0;
    static constexpr int nad2 = (dconf.d2 || dconf.vd2) ? N_OUT : 0;

    static constexpr int nbd1 = dconf.needsBD1() ? NET_NOUTPUT*N_OUT : 0; // number of stored backprop values
    static constexpr int nbd1_next = dconf.needsBD1() ? NET_NOUTPUT*nout_next : 0; // number from previous layer

    static constexpr int nbd2 = dconf.needsBD2()
                                ? NET_NOUTPUT*N_OUT
                                : 0; // number of diagonal second order backprop values
    static constexpr int nbd2_next = dconf.needsBD2() ? NET_NOUTPUT*nout_next : 0;

private: // arrays
    std::array<ValueT, N_OUT> _out{};
    // the deriv arrays could be quite large, so we heap allocate them
    const std::unique_ptr<std::array<ValueT, nd2>> _d1_ptr{std::make_unique<std::array<ValueT, nd2>>()};
    const std::unique_ptr<std::array<ValueT, nd2>> _d2_ptr{std::make_unique<std::array<ValueT, nd2>>()};
    const std::unique_ptr<std::array<ValueT, nbd1>> _bd1_ptr{std::make_unique<std::array<ValueT, nbd1>>()}; // intermediate values for vd1, but NOT stored as dnet/du
    const std::unique_ptr<std::array<ValueT, nbd2>> _bd2_ptr{std::make_unique<std::array<ValueT, nbd2>>()}; // intermediate values for diag-vd2, but NOT stored as dnet^2/du^2
    std::array<ValueT, nad1> _ad1{}; // activation function d1
    std::array<ValueT, nad2> _ad2{}; // activation function d2

public: // public member variables
    ACTFType actf{}; // the activation function
    std::array<ValueT, nbeta> beta{}; // the weights (NOTE: If the network should contain millions of weights, stacksize must be increased for the program)

    // public const output references
    constexpr const std::array<ValueT, N_OUT> &out() const { return _out; }
    constexpr const std::array<ValueT, nd2> &d1() const { return *_d1_ptr; }
    constexpr const std::array<ValueT, nd2> &d2() const { return *_d2_ptr; }
    constexpr const std::array<ValueT, nbd1> &bd1() const { return *_bd1_ptr; }
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
        this->_computeActivation(dflags.needsAny(), dflags.d2() || dflags.vd2());
    }

    // forward-accumulate second order input derivatives from a layer
    constexpr void _computeD2_Layer(const ValueT * in_d1, const ValueT * in_d2)
    {
        auto &D1 = *_d1_ptr;
        auto &D2 = *_d2_ptr;
        D1.fill(0.);
        D2.fill(0.);
        for (int i = 0; i < N_OUT; ++i) {
            for (int j = 0; j < N_IN; ++j) {
                const ValueT bij = beta[1 + i*(N_IN + 1) + j];
                for (int k = 0; k < NET_NINPUT; ++k) {
                    D1[i*NET_NINPUT + k] += bij*in_d1[j*NET_NINPUT + k];
                    D2[i*NET_NINPUT + k] += bij*in_d2[j*NET_NINPUT + k];
                }
            }
            for (int l = i*NET_NINPUT; l < (i + 1)*NET_NINPUT; ++l) {
                D2[l] = _ad1[i]*D2[l] + _ad2[i]*D1[l]*D1[l];
                D1[l] *= _ad1[i];
            }
        }
    }

    // forward-accumulate second order deriv when the inputs correspond (besides shift/scale) to the true network inputs
    constexpr void _computeD2_Input()
    {
        static_assert(N_IN == NET_NINPUT, "");
        auto &D1 = *_d1_ptr;
        auto &D2 = *_d2_ptr;
        for (int i = 0; i < N_OUT; ++i) {
            for (int j = 0; j < N_IN; ++j) {
                const ValueT bij = beta[1 + i*(N_IN + 1) + j];
                D1[i*N_IN + j] = _ad1[i]*bij;
                D2[i*N_IN + j] = _ad2[i]*bij*bij;
            }
        }
    }

    // start forward pass from true network inputs
    constexpr void _forwardInput(const ValueT input[], DynamicDFlags dflags)
    {
        // statically secure this call (i.e. using it on non-input layer will not compile)
        static_assert(N_IN == NET_NINPUT, "[TemplLayer::ForwardInput] N_IN != NET_NINPUT");

        dflags = dflags.AND(dconf); // AND static and dynamic conf
        this->_computeOutput(input, dflags);

        // fill diagonal d1,d2
        if (dflags.d2()) {
            this->_computeD2_Input();
        }
    }

    // continue forward pass from previous layer
    constexpr void _forwardLayer(const ValueT input[], const ValueT in_d1[], const ValueT in_d2[], DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        this->_computeOutput(input, dflags);

        // input derivs
        if (dflags.d2()) {
            this->_computeD2_Layer(in_d1, in_d2);
        }
    }

    // initialize backprop on output layer
    constexpr void _backwardOutput(DynamicDFlags dflags)
    {
        static_assert(N_OUT == NET_NOUTPUT, "[TemplLayer::setOutputVD] N_OUT != NET_NOUTPUT");
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        auto &BD1 = *_bd1_ptr;
        auto &BD2 = *_bd2_ptr;
        BD1.fill(0.);
        BD2.fill(0.);

        // set the diagonal elements
        if (!dflags.needsBD1()) { return; }
        for (int i = 0; i < NET_NOUTPUT; ++i) {
            BD1[i*NET_NOUTPUT + i] = _ad1[i];
        }
        if (!dflags.needsBD2()) { return; }
        for (int i = 0; i < NET_NOUTPUT; ++i) {
            BD2[i*NET_NOUTPUT + i] = _ad2[i];
        }
    }

    // continue backprop coming from a layer (first order version)
    constexpr void _backwardLayerBD1(const ValueT bd1_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        auto &BD1 = *_bd1_ptr;

        for (int i = 0; i < NET_NOUTPUT; ++i) {
            for (int j = 0; j < nout_next; ++j) {
                const int beta_i0 = 1 + j*(N_OUT + 1);
                for (int k = 0; k < N_OUT; ++k) {
                    BD1[i*N_OUT + k] += beta_next[beta_i0 + k]*bd1_next[i*nout_next + j];
                }
            }
            for (int k = 0; k < N_OUT; ++k) {
                BD1[i*N_OUT + k] *= _ad1[k];
            }
        }
    }

    // continue backprop coming from a layer (first + second order version)
    constexpr void _backwardLayerBD12(const ValueT bd1_next[], const ValueT bd2_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        auto &BD1 = *_bd1_ptr;
        auto &BD2 = *_bd2_ptr;

        for (int i = 0; i < NET_NOUTPUT; ++i) {
            const int d_i0 = i*N_OUT;
            for (int j = 0; j < nout_next; ++j) {
                for (int k = 0; k < N_OUT; ++k) {
                    const ValueT bjk = beta_next[1 + j*(N_OUT + 1) + k];
                    BD1[d_i0 + k] += bjk*bd1_next[i*nout_next + j];
                    BD2[d_i0 + k] += bjk*bjk*bd2_next[i*nout_next + j];
                }
            }
            for (int k = 0; k < N_OUT; ++k) {
                BD2[d_i0 + k] = _ad1[k]*_ad1[k]*BD2[d_i0 + k] + _ad2[k]*BD1[d_i0 + k];
                BD1[d_i0 + k] *= _ad1[k];
            }
        }
    }

    // continue backprop coming from a layer
    constexpr void _backwardLayer(const ValueT bd1_next[], const ValueT bd2_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        (*_bd1_ptr).fill(0.);
        (*_bd2_ptr).fill(0.);
        if (!dflags.needsBD1()) { return; }
        if (dflags.needsBD2()) {
            _backwardLayerBD12(bd1_next, bd2_next, beta_next, dflags);
        }
        else {
            _backwardLayerBD1(bd1_next, beta_next, dflags);
        }
    }

    constexpr void _layerGrad(const ValueT input[], ValueT vd1_block[], const int iout, DynamicDFlags dflags) const
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        if (!dflags.vd1()) { return; }
        const auto BD1_iout = (*_bd1_ptr).begin() + iout*N_OUT;

        for (int j = 0; j < N_OUT; ++j) {
            *vd1_block++ = BD1_iout[j]; // bias weight gradient
            for (int k = 0; k < N_IN; ++k, ++vd1_block) {
                *vd1_block = input[k]*BD1_iout[j];
            }
        }
    }

    constexpr void _layerGrad2(const ValueT input[], ValueT vd2_block[], const int iout, DynamicDFlags dflags) const
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        if (!dflags.vd2()) { return; }
        const auto BD2_iout = (*_bd2_ptr).begin() + iout*N_OUT;

        for (int j = 0; j < N_OUT; ++j) {
            *vd2_block++ = BD2_iout[j]; // bias weight gradient
            for (int k = 0; k < N_IN; ++k, ++vd2_block) {
                *vd2_block = input[k]*input[k]*BD2_iout[j];
            }
        }
    }

    constexpr void _inputGrad(ValueT d1_out[], DynamicDFlags dflags) const
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        if (!dflags.d1()) { return; }
        std::fill(d1_out, d1_out + NET_NOUTPUT*N_IN, 0.);
        const auto &BD1 = *_bd1_ptr;

        for (int i = 0; i < NET_NOUTPUT; ++i) {
            for (int j = 0; j < N_OUT; ++j) {
                const ValueT bdj = BD1[i*N_OUT + j];
                for (int k = 0; k < N_IN; ++k) {
                    const ValueT bjk = beta[1 + j*(N_IN + 1) + k];
                    d1_out[i*N_IN + k] += bjk*bdj;
                }
            }
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

    constexpr void ForwardLayer(const std::array<ValueT, N_IN> &input, const std::array<ValueT, nd2_prev> &in_d1, const std::array<ValueT, nd2_prev> &in_d2, DynamicDFlags dflags)
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

    constexpr void BackwardLayer(const std::array<ValueT, nbd1_next> &bd1_next, const std::array<ValueT, nbd2_next> &bd2_next, const std::array<ValueT, NBETA_NEXT> &beta_next, DynamicDFlags dflags)
    {
        _backwardLayer(bd1_next.begin(), bd2_next.begin(), beta_next.begin(), dflags);
    }

    constexpr void BackwardLayer(const ValueT bd1_next[], const ValueT bd2_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        _backwardLayer(bd1_next, bd2_next, beta_next, dflags);
    }


    // --- Calculate weight gradient block of output unit iout with respect to this layers' weights

    constexpr void storeLayerVD1(const std::array<ValueT, N_IN> &input, std::array<ValueT, nbeta> &vd1_block, int iout, DynamicDFlags dflags) const
    {
        _layerGrad(input.begin(), vd1_block.begin(), iout, dflags);
    }

    constexpr void storeLayerVD2(const std::array<ValueT, N_IN> &input, std::array<ValueT, nbeta> &vd2_block, int iout, DynamicDFlags dflags) const
    {
        _layerGrad2(input.begin(), vd2_block.begin(), iout, dflags);
    }

    constexpr void storeLayerVD1(const ValueT input[], ValueT vd1_block[], int iout, DynamicDFlags dflags) const
    {
        _layerGrad(input, vd1_block, iout, dflags);
    }

    constexpr void storeLayerVD2(const ValueT input[], ValueT vd2_block[], int iout, DynamicDFlags dflags) const
    {
        _layerGrad2(input, vd2_block, iout, dflags);
    }


    // --- Calculate input gradient block of output units with respect to this layers inputs

    constexpr void storeInputD1(std::array<ValueT, NET_NOUTPUT*N_IN> &d1_out, DynamicDFlags dflags) const
    {
        _inputGrad(d1_out.begin(), dflags);
    }

    constexpr void storeInputD1(ValueT d1_out[], DynamicDFlags dflags) const
    {
        _inputGrad(d1_out, dflags);
    }
};
} // templ

#endif

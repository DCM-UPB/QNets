#ifndef QNETS_TEMPL_TEMPLLAYER_HPP
#define QNETS_TEMPL_TEMPLLAYER_HPP

#include "qnets/templ/DerivConfig.hpp"

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
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
template <typename ValueT, int NET_NINPUT, int IBETA_BEGIN, int N_IN, int N_OUT, class ACTFType, DerivConfig DCONF>
class TemplLayer: public LayerConfig<N_OUT, ACTFType>
{
public:
    // N_IN/IBETA dependent sizes
    static constexpr int ninput = N_IN;
    static constexpr int nbeta = (N_IN + 1)*N_OUT;
    static constexpr int ibeta_begin = IBETA_BEGIN;
    static constexpr int ibeta_end = IBETA_BEGIN + nbeta;

    // Sizes which also depend on DCONF
    static constexpr StaticDFlags<DCONF> dconf{};
    static constexpr int nd1 = dconf.d1 ? NET_NINPUT*N_OUT : 0; // number of input derivative values
    static constexpr int nd1_feed = dconf.d1 ? NET_NINPUT*N_IN : 0; // number of deriv values from previous layer
    static constexpr int nd2 = dconf.d2 ? nd1 : 0;
    static constexpr int nd2_feed = dconf.d2 ? nd1_feed : 0;
    static constexpr int nvd1 = dconf.vd1 ? ibeta_end*N_OUT : 0; // number of variational derivative values
    static constexpr int nvd1_feed = dconf.vd1 ? IBETA_BEGIN*N_IN : 0; // number of deriv values from previous layer

private: // arrays
    std::array<ValueT, N_OUT> _out{};
    std::array<ValueT, nd1> _d1{};
    std::array<ValueT, nd2> _d2{};

    // the vderiv array can be quite large, so we need to heap allocate
    const std::unique_ptr<std::array<ValueT, nvd1>> _vd1_ptr{std::make_unique<std::array<ValueT, nvd1>>()};

    std::array<ValueT, dconf.d1 || dconf.vd1 ? N_OUT : 0> _ad1{}; // activation function d1
    std::array<ValueT, dconf.d2 ? N_OUT : 0> _ad2{}; // activation function d2

public: // public member variables
    ACTFType actf{}; // the activation function
    std::array<ValueT, nbeta> beta{}; // the weights

    // public const output references
    constexpr const std::array<ValueT, N_OUT> &out() const { return _out; }
    constexpr const std::array<ValueT, nd1> &d1() const { return _d1; }
    constexpr const std::array<ValueT, nd2> &d2() const { return _d2; }
    constexpr const std::array<ValueT, nvd1> &vd1() const { return *_vd1_ptr; }

private:
    constexpr void _computeFeed(const ValueT input[])
    {
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1); // increments through the indices of the first non-offset beta per unit
            _out[i] = beta[beta_i0 - 1]; // bias weight
            _out[i] += std::inner_product(input, input + ninput, beta.begin() + beta_i0, 0.); // found to be much faster than loop
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
        this->_computeActivation(dflags.d1(), dflags.d2());
    }

    constexpr void _computeD1_Layer(const ValueT in_d1[])
    {
        std::fill(_d1.begin(), _d1.end(), 0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1);
            const int d_i0 = i*NET_NINPUT;
            for (int j = 0; j < N_IN; ++j) {
                for (int k = 0; k < NET_NINPUT; ++k) {
                    _d1[d_i0 + k] += beta[beta_i0 + j]*in_d1[j*NET_NINPUT + k];
                }
            }
            for (int l = d_i0; l < d_i0 + NET_NINPUT; ++l) {
                _d1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeD1_Input() // i.e. in_d1[i][i] = 1., else 0
    {
        std::fill(_d1.begin(), _d1.end(), 0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(NET_NINPUT + 1);
            for (int j = 0; j < NET_NINPUT; ++j) {
                _d1[i*NET_NINPUT + j] = _ad1[i]*beta[beta_i0 + j];
            }
        }
    }

    constexpr void _computeD12_Layer(const ValueT in_d1[], const ValueT in_d2[])
    {
        std::fill(_d1.begin(), _d1.end(), 0.);
        std::fill(_d2.begin(), _d2.end(), 0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1);
            const int d_i0 = i*NET_NINPUT;
            for (int j = 0; j < N_IN; ++j) {
                for (int k = 0; k < NET_NINPUT; ++k) {
                    _d1[d_i0 + k] += beta[beta_i0 + j]*in_d1[j*NET_NINPUT + k];
                    _d2[d_i0 + k] += beta[beta_i0 + j]*in_d2[j*NET_NINPUT + k];
                }
            }
            for (int l = d_i0; l < d_i0 + NET_NINPUT; ++l) {
                _d2[l] = _ad1[i]*_d2[l] + _ad2[i]*_d1[l]*_d1[l];
                _d1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeD12_Input()
    {
        std::fill(_d1.begin(), _d1.end(), 0.);
        std::fill(_d2.begin(), _d2.end(), 0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(NET_NINPUT + 1);
            for (int j = 0; j < NET_NINPUT; ++j) {
                _d1[i*NET_NINPUT + j] = _ad1[i]*beta[beta_i0 + j];
                _d2[i*NET_NINPUT + j] = _ad2[i]*beta[beta_i0 + j]*beta[beta_i0 + j];
            }
        }
    }

    constexpr void _computeVD1_Layer(const ValueT input[], const ValueT in_vd1[])
    {
        ValueT * const VD = (*_vd1_ptr).begin();
        std::fill(VD, VD + nvd1, 0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1);
            const int d_i0 = i*ibeta_end;
            // add old elements
            for (int j = 0; j < N_IN; ++j) {
                for (int k = 0; k < IBETA_BEGIN; ++k) {
                    VD[d_i0 + k] += beta[beta_i0 + j]*in_vd1[j*IBETA_BEGIN + k];
                }
            }
            // add new elements
            const int beta_inew = IBETA_BEGIN + beta_i0; // the first new non-offset beta index
            VD[d_i0 + beta_inew - 1] = 1.; // the bias weight derivative
            std::copy(input, input + N_IN, VD + d_i0 + beta_inew);
            for (int l = d_i0; l < d_i0 + ibeta_end; ++l) {
                VD[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeVD1_Input(const ValueT input[])
    {
        ValueT * const VD = (*_vd1_ptr).begin();
        std::fill(VD, VD + nvd1, 0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(NET_NINPUT + 1);
            const int d_i0 = i*ibeta_end;
            VD[d_i0 + beta_i0 - 1] = 1.; // the bias weight derivative
            std::copy(input, input + NET_NINPUT, VD + d_i0 + beta_i0);
            for (int l = d_i0; l < d_i0 + ibeta_end; ++l) {
                VD[l] *= _ad1[i];
            }
        }
    }

    constexpr void _propagateInput(const ValueT input[], DynamicDFlags dflags)
    {
        // statically secure this call (i.e. using it on non-input layer will not compile)
        static_assert(N_IN == NET_NINPUT, "[TemplLayer::PropagateInput] N_IN != NET_NINPUT");
        static_assert(IBETA_BEGIN == 0, "[TemplLayer::PropagateInput] IBETA_BEGIN != 0");

        dflags = dflags.AND(dconf); // AND static and dynamic conf
        this->_computeOutput(input, dflags);

        // fill diagonal d1,d2
        if (dflags.d2()) {
            this->_computeD12_Input();
        }
        else if (dflags.d1()) {
            this->_computeD1_Input();
        }

        // input vderiv
        if (dflags.vd1()) {
            this->_computeVD1_Input(input);
        }
    }


    constexpr void _propagateLayer(const ValueT input[], const ValueT in_d1[], const ValueT in_d2[], const ValueT in_vd1[], DynamicDFlags dflags)
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

        // vderiv
        if (dflags.vd1()) {
            this->_computeVD1_Layer(input, in_vd1);
        }
    }

public: // public propagate methods
    // We support 3 different ways to provide arrays for propagate calls:
    // Array: Bounds statically checked due to type system
    // Vector: Runtime bounds checking (once per call)
    // C-Style (Pointer): No bounds checking


    // --- Propagation of input data (not layer)

    constexpr void PropagateInput(const std::array<ValueT, NET_NINPUT> &input, DynamicDFlags dflags)
    {
        _propagateInput(input.begin(), dflags);
    }

    constexpr void PropagateInput(const std::vector<ValueT> &input, DynamicDFlags dflags)
    {
        assert(input.size() == N_IN);
        _propagateInput(input.begin(), dflags);
    }

    constexpr void PropagateInput(const ValueT input[], DynamicDFlags dflags)
    {
        _propagateInput(input, dflags);
    }


    // --- Propagation of layer data

    constexpr void PropagateLayer(const std::array<ValueT, N_IN> &input, const std::array<ValueT, nd1_feed> &in_d1, const std::array<ValueT, nd2_feed> &in_d2, const std::array<ValueT, nvd1_feed> &in_vd1, DynamicDFlags dflags)
    {
        _propagateLayer(input.begin(), in_d1.begin(), in_d2.begin(), in_vd1.begin(), dflags);
    }

    constexpr void PropagateLayer(const std::vector<ValueT> &input, const std::vector<ValueT> &in_d1, const std::vector<ValueT> &in_d2, const std::vector<ValueT> &in_vd1, DynamicDFlags dflags)
    {
        assert(input.size() == N_IN);
        assert(in_d1.size() == nd1_feed);
        assert(in_d2.size() == nd2_feed);
        assert(in_vd1.size() == nvd1_feed);
        _propagateLayer(input.begin(), in_d1.begin(), in_d2.begin(), in_vd1.begin(), dflags);
    }

    constexpr void PropagateLayer(const ValueT input[], const ValueT in_d1[], const ValueT in_d2[], const ValueT in_vd1[], DynamicDFlags dflags)
    {
        _propagateLayer(input, in_d1, in_d2, in_vd1, dflags);
    }
};

// --- Helper function to propagate through a tuple of layers

namespace detail
{ // Recursive FFProp over tuple
template <class TupleT>
constexpr void ffprop_layers_impl(TupleT &/*layers*/, DynamicDFlags /*dflags*/, std::index_sequence<>) {}

template <class TupleT, size_t I, size_t ... Is>
constexpr void ffprop_layers_impl(TupleT &layers, DynamicDFlags dflags, std::index_sequence<I, Is...>)
{
    const auto &prev_layer = std::get<I>(layers);
    std::get<I + 1>(layers).PropagateLayer(prev_layer.out(), prev_layer.d1(), prev_layer.d2(), prev_layer.vd1(), dflags);
    ffprop_layers_impl<TupleT>(layers, dflags, std::index_sequence<Is...>{});
}
} // detail

// The public function
template <class ArrayT, class TupleT>
constexpr void propagateLayers(const ArrayT &input, TupleT &layers, DynamicDFlags dflags)
{
    std::get<0>(layers).PropagateInput(input, dflags);
    detail::ffprop_layers_impl<TupleT>(layers, dflags, std::make_index_sequence<std::tuple_size<TupleT>::value - 1>{});
}
} // templ

#endif

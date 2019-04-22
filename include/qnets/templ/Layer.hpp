#ifndef QNETS_TEMPL_LAYER_HPP
#define QNETS_TEMPL_LAYER_HPP

#include "qnets/templ/DerivConfig.hpp"

#include <type_traits>

namespace templ
{
// --- TemplNet Layers

// Layer Config
//
// To pass non-input layer configurations as variadic parameter pack
template <typename SizeT, SizeT N_IN, SizeT N_OUT, class ACTFType>
struct LayerConfig
{
    static constexpr SizeT ninput = N_IN;
    static constexpr SizeT noutput = N_OUT;
    static constexpr SizeT nbeta = (N_IN + 1)*N_OUT;
    static constexpr SizeT nlink = N_IN*N_OUT;
    using ACTF_Type = ACTFType;

    static constexpr SizeT size() { return noutput; }
};


// The actual Layer class
//
template <typename SizeT, typename ValueT, SizeT NET_NINPUT, SizeT N_IN, SizeT N_OUT, class ACTFType, DerivConfig DCONF>
class Layer: public LayerConfig<SizeT, N_IN, N_OUT, ACTFType>
{
public: // sizes
    static constexpr StaticDFlags<DCONF> dconf{};
    static constexpr SizeT nd1 = dconf.d1 ? NET_NINPUT*N_OUT : 0; // number of input derivative values
    static constexpr SizeT nd1_feed = dconf.d1 ? NET_NINPUT*N_IN : 0; // number of deriv values from previous layer
    static constexpr SizeT nd2 = dconf.d2 ? nd1 : 0;
    static constexpr SizeT nd2_feed = dconf.d2 ? nd1_feed : 0;

private: // arrays
    std::array<ValueT, N_OUT> _out;
    std::array<ValueT, nd1> _d1;
    std::array<ValueT, nd2> _d2;
    std::array<ValueT, dconf.d1 ? N_OUT : 0> _ad1; // activation function d1
    std::array<ValueT, dconf.d2 ? N_OUT : 0> _ad2; // activation function d2

public: // public member variables
    ACTFType actf{}; // the activation function
    std::array<ValueT, Layer::nbeta> beta{}; // the weights

    // public const output references
    constexpr const std::array<ValueT, N_OUT> &out() const { return _out; }
    constexpr const std::array<ValueT, nd1> &d1() const { return _d1; }
    constexpr const std::array<ValueT, nd2> &d2() const { return _d2; }
    //std::array<ValueT, LayerConf::nvp> vd1;

private: // private methods
    constexpr void _computeFeed(const std::array<ValueT, N_IN> &input)
    {
        for (SizeT i = 0; i < N_OUT; ++i) {
            const SizeT beta_i0 = 1 + i*(N_IN + 1); // increments through the indices of the first non-offset beta per unit
            _out[i] = beta[beta_i0 - 1]; // bias weight
            _out[i] += std::inner_product(input.begin(), input.end(), beta.begin()+beta_i0, 0.);
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

    constexpr void _computeOutput(const std::array<ValueT, N_IN> &input, DynamicDFlags dflags)
    {
        this->_computeFeed(input);
        this->_computeActivation(dflags.d1(), dflags.d2());
    }

    constexpr void _computeD1_Layer(const std::array<ValueT, nd1_feed> &in_d1)
    {
        std::fill(_d1.begin(), _d1.end(), 0.);
        for (SizeT i = 0; i < N_OUT; ++i) {
            const SizeT beta_i0 = 1 + i*(N_IN + 1);
            const SizeT d_i0 = i*NET_NINPUT;
            for (SizeT j = 0; j < N_IN; ++j) {
                for (SizeT k = 0; k < NET_NINPUT; ++k) {
                    _d1[d_i0 + k] += beta[beta_i0 + j]*in_d1[j*NET_NINPUT + k];
                }
            }
            for (SizeT l = d_i0; l < d_i0+NET_NINPUT; ++l) {
                _d1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeD1_Input() // i.e. in_d1[i][i] = 1., else 0
    {
        std::fill(_d1.begin(), _d1.end(), 0.);
        for (SizeT i = 0; i < N_OUT; ++i) {
            const SizeT beta_i0 = 1 + i*(N_IN + 1);
            for (SizeT j = 0; j < N_IN; ++j) {
                _d1[i*NET_NINPUT + j] = _ad1[i] * beta[beta_i0 + j];
            }
        }
    }

    constexpr void _computeD12_Layer(const std::array<ValueT, nd1_feed> &in_d1, const std::array<ValueT, nd2_feed> &in_d2)
    {
        std::fill(_d1.begin(), _d1.end(), 0.);
        std::fill(_d2.begin(), _d2.end(), 0.);
        for (SizeT i = 0; i < N_OUT; ++i) {
            const SizeT beta_i0 = 1 + i*(N_IN + 1);
            const SizeT d_i0 = i*NET_NINPUT;
            for (SizeT j = 0; j < N_IN; ++j) {
                for (SizeT k = 0; k < NET_NINPUT; ++k) {
                    _d1[d_i0 + k] += beta[beta_i0 + j]*in_d1[j*NET_NINPUT + k];
                    _d2[d_i0 + k] += beta[beta_i0 + j]*in_d2[j*NET_NINPUT + k];
                }
            }
            for (SizeT l = d_i0; l < d_i0+NET_NINPUT; ++l) {
                _d2[l] = _ad1[i]*_d2[l] + _ad2[i]*_d1[l]*_d1[l];
                _d1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeD12_Input()
    {
        std::fill(_d1.begin(), _d1.end(), 0.);
        std::fill(_d2.begin(), _d2.end(), 0.);
        for (SizeT i = 0; i < N_OUT; ++i) {
            const SizeT beta_i0 = 1 + i*(N_IN + 1);
            for (SizeT j = 0; j < N_IN; ++j) {
                _d1[i*NET_NINPUT + j] = _ad1[i] * beta[beta_i0 + j];
                _d2[i*NET_NINPUT + j] = _ad2[i] * beta[beta_i0 + j] * beta[beta_i0 + j];
            }
        }
    }

public: // public methods
    constexpr void PropagateInput(const std::array<ValueT, N_IN> &input, DynamicDFlags dflags) // propagation of input data (not layer)
    {
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

    constexpr void PropagateLayer(const std::array<ValueT, N_IN> &input, const std::array<ValueT, nd1_feed> &in_d1, const std::array<ValueT, nd2_feed> &in_d2, DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        this->_computeOutput(input, dflags);
        if (dflags.d2()) {
            this->_computeD12_Layer(in_d1, in_d2);
        }
        else if (dflags.d1()) {
            this->_computeD1_Layer(in_d1);
        }
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
    std::get<I + 1>(layers).PropagateLayer(prev_layer.out(), prev_layer.d1(), prev_layer.d2(), dflags);
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

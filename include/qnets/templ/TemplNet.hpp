#ifndef FFNN_NET_TEMPLNET_HPP
#define FFNN_NET_TEMPLNET_HPP

#include "qnets/tool/PackTools.hpp"
#include "qnets/tool/TupleTools.hpp"
#include "qnets/templ/TemplLayer.hpp"
#include "qnets/templ/LayerPackTools.hpp"
#include "qnets/templ/DerivConfig.hpp"

#include <array>
#include <tuple>
#include <algorithm>
#include <utility>
#include <type_traits>

namespace templ
{

// detail namespace with helpers
namespace detail
{
// NOTE: For technical reasons, we need to use a TemplNetShape class for static creation
//       of arrays like nbeta_shape (which depends on TemplLayer, not LayerConf),
//       making use of integer_sequence. Might become obsolete with C++17.
template <class LTuplType, class ISeq>
struct TemplNetShape;

template <class LTuplType, int ... Is>
struct TemplNetShape<LTuplType, std::integer_sequence<int, Is...>>
{
private:
    using LayerTuple = LTuplType;

public:
    // Some static arrays to make access easier (needs to be defined again below the class, until C++17)
    static constexpr std::array<int, sizeof...(Is)> nunits{std::tuple_element<Is, LayerTuple>::type::size()...};
    static constexpr std::array<int, sizeof...(Is)> nbetas{std::tuple_element<Is, LayerTuple>::type::nbeta...};
};
template <class LTuplType, int ... Is>
constexpr std::array<int, sizeof...(Is)> TemplNetShape<LTuplType, std::integer_sequence<int, Is...>>::nunits;
template <class LTuplType, int ... Is>
constexpr std::array<int, sizeof...(Is)> TemplNetShape<LTuplType, std::integer_sequence<int, Is...>>::nbetas;


// --- subroutines to propagate (i.e. fwd+back) input through a tuple of layers

// Recursive ForwardProp over tuple
template <class TupleT>
constexpr void fwdprop_layers_impl(TupleT &/*layers*/, DynamicDFlags /*dflags*/, std::index_sequence<>) {}

template <class TupleT, size_t I, size_t ... Is>
constexpr void fwdprop_layers_impl(TupleT &layers, DynamicDFlags dflags, std::index_sequence<I, Is...>)
{
    const auto &prev_layer = std::get<I>(layers);
    std::get<I + 1>(layers).ForwardLayer(prev_layer.out(), prev_layer.d1(), prev_layer.d2(), dflags);
    fwdprop_layers_impl<TupleT>(layers, dflags, std::index_sequence<Is...>{});
}

// Recursive BackProp over tuple
template <class TupleT>
constexpr void backprop_layers_impl(TupleT &/*layers*/, DynamicDFlags /*dflags*/, std::index_sequence<>) {}

template <class TupleT, size_t I, size_t ... Is>
constexpr void backprop_layers_impl(TupleT &layers, DynamicDFlags dflags, std::index_sequence<I, Is...>)
{
    constexpr size_t idx = sizeof...(Is);
    const auto &next_layer = std::get<idx + 1>(layers);
    std::get<idx>(layers).BackwardLayer(next_layer.vd1(), next_layer.beta, dflags);
    backprop_layers_impl<TupleT>(layers, dflags, std::index_sequence<Is...>{});
}

// calculate final weight gradients of a backpropped layer
template <int ibeta_begin, int nbeta_net, class LayerT, class ArrayT1, class ArrayT2>
constexpr void calc_grad_layer(const LayerT &layer, const ArrayT1 &input, ArrayT2 &vd1, DynamicDFlags dflags)
{
    if (!dflags.vd1()) {
        std::fill(vd1.begin(), vd1.end(), 0.);
        return;
    }
    for (int i = 0; i < layer.net_nout; ++i) {
        layer.storeLayerGradients(input.begin(), vd1.begin() + i*nbeta_net + ibeta_begin, i, dflags);
    }
}

// accumulate network gradients into vd1
template <int ibeta_begin, int nbeta_net, class TupleT, class ArrayT1, class ArrayT2>
constexpr void grad_layers_impl(const TupleT &/*layers*/, const ArrayT1 &/*input*/, ArrayT2 &/*vd1*/, DynamicDFlags /*dflags*/, std::index_sequence<>) {}

template <int ibeta_begin, int nbeta_net, class TupleT, class ArrayT1, class ArrayT2, size_t I, size_t ... Is>
constexpr void grad_layers_impl(const TupleT &layers, const ArrayT1 &input, ArrayT2 &vd1, DynamicDFlags dflags, std::index_sequence<I, Is...>)
{
    using layerT = std::tuple_element_t<I, TupleT>;
    const auto &this_layer = std::get<I>(layers);

    calc_grad_layer<ibeta_begin, nbeta_net, decltype(this_layer)/*clang fix*/>(this_layer, input, vd1, dflags);
    grad_layers_impl<ibeta_begin + layerT::nbeta, nbeta_net, TupleT>(layers, this_layer.out(), vd1, dflags, std::index_sequence<Is...>{});
}
} // detail



// --- The fully templated TemplNet FFNN

template <typename ValueT, DerivConfig DCONF, int N_IN, class ... LayerConfs>
class TemplNet
{
public:
    // --- Static Setup

    // LayerTuple type / Shape
    using LayerTuple = typename lpack::LayerPackTuple<ValueT, DCONF, N_IN, LayerConfs...>::type;
    using Shape = detail::TemplNetShape<LayerTuple, std::make_integer_sequence<int, sizeof...(LayerConfs)>>;

    // some basic static sizes
    static constexpr int nlayer = tupl::count<int, LayerTuple>();
    static constexpr int ninput = std::tuple_element<0, LayerTuple>::type::ninput;
    static constexpr int noutput = std::tuple_element<nlayer - 1, LayerTuple>::type::noutput;
    static constexpr int nbeta = lpack::countBetas<N_IN, LayerConfs...>();
    static constexpr int nunit = lpack::countUnits<LayerConfs...>();

    // static derivative config
    static constexpr StaticDFlags<DCONF> dconf{};

    // Static Output Deriv Array Sizes (depend on DCONF)
    static constexpr int nd1 = std::tuple_element<nlayer - 1, LayerTuple>::type::nd1;
    static constexpr int nd2 = std::tuple_element<nlayer - 1, LayerTuple>::type::nd2;
    static constexpr int nvd1 = dconf.vd1 ? noutput*nbeta : 0;


    // Basic assertions
    static_assert(nlayer == static_cast<int>(sizeof...(LayerConfs)), ""); // -> BUG!
    static_assert(ninput == N_IN, ""); // -> BUG!
    static_assert(noutput == Shape::nunits[nlayer - 1], ""); // -> BUG!
    static_assert(nlayer > 1, "[TemplNet] nlayer <= 1");
    static_assert(lpack::hasNoEmptyLayer<(ninput > 0), LayerConfs...>(), "[TemplNet] LayerConf pack contains empty Layer.");


    // --- Non-statics

private:
    // The layer tuple
    LayerTuple _layers{};

    // arrays of array.begin() pointers, for run-time indexing
    const std::array<const ValueT *, nlayer> _out_begins;
    const std::array<ValueT *, nlayer> _beta_begins;

    // vd1 array
    std::array<ValueT, nvd1> _vd1{};

public:
    // dynamic (opt-out) derivative config (default to DCONF or explicit set in ctor)
    DynamicDFlags dflags{DCONF};

    // input array
    std::array<ValueT, ninput> input{};

public:
    explicit constexpr TemplNet(DynamicDFlags init_dflags = DynamicDFlags{DCONF}):
            _out_begins(tupl::make_fcont<std::array<const ValueT *, nlayer>>(_layers, [](const auto &layer) { return &layer.out().front(); })),
            _beta_begins(tupl::make_fcont<std::array<ValueT *, nlayer>>(_layers, [](auto &layer) { return &layer.beta.front(); })),
            dflags(init_dflags) {}

    // --- Get information about the NN structure

    static constexpr int getNLayer() { return nlayer; }
    static constexpr int getNInput() { return ninput; }
    static constexpr int getNOutput() { return noutput; }
    static constexpr int getNUnit() { return nunit; }
    static constexpr int getNUnit(int i) { return Shape::nunits[i]; }
    static constexpr const auto &getUnitShape() { return Shape::nunits; }

    // Read access to LayerTuple / individual layers
    constexpr const LayerTuple &getLayers() const { return _layers; }
    template <int I>
    constexpr const auto &getLayer() const { return std::get<I>(_layers); }

    // --- const get Value Arrays/Elements
    constexpr const auto &getInput() const { return input; } // alternative const read of public input array
    constexpr const auto &getOutput() const { return std::get<nlayer - 1>(_layers).out(); } // get values of output layer
    constexpr ValueT getOutput(int i) const { return this->getOutput()[i]; }
    constexpr const auto &getD1() const { return std::get<nlayer - 1>(_layers).d1(); } // get derivative of output with respect to input
    constexpr ValueT getD1(int i, int j) const { return this->getD1()[i*ninput + j]; }
    constexpr const auto &getD2() const { return std::get<nlayer - 1>(_layers).d2(); }
    constexpr ValueT getD2(int i, int j) const { return this->getD2()[i*ninput + j]; }
    constexpr const auto &getVD1() const { return _vd1; }
    constexpr ValueT getVD1(int i, int j) const { return _vd1[i*nbeta + j]; }

    // --- check derivative setup
    static constexpr bool allowsD1() { return dconf.d1; }
    static constexpr bool allowsD2() { return dconf.d2; }
    static constexpr bool allowsVD1() { return dconf.vd1; }

    constexpr bool hasD1() const { return dconf.d1 && dflags.d1(); }
    constexpr bool hasD2() const { return dconf.d2 && dflags.d2(); }
    constexpr bool hasVD1() const { return dconf.vd1 && dflags.vd1(); }

    // --- Access Network Weights (Betas)
    static constexpr int getNBeta() { return nbeta; }
    static constexpr int getNBeta(int i) { return Shape::nbetas[i]; }
    static constexpr const auto &getBetaShape() { return Shape::nbetas; }

    constexpr ValueT getBeta(int i) const // get beta by index
    {
        int idx = 0;
        while (i >= Shape::nbetas[idx]) {
            i -= Shape::nbetas[idx];
            ++idx;
        }
        return *(_beta_begins[idx] + i);
    }

    template <class IterT>
    constexpr void getBetas(IterT begin, const IterT end) const // get betas into range
    {
        int idx = 0;
        while (begin < end) {
            const auto blocksize = (end - begin > Shape::nbetas[idx]) ? Shape::nbetas[idx] : end - begin;
            std::copy(_beta_begins[idx], _beta_begins[idx] + blocksize, begin);
            begin += blocksize;
            ++idx;
        }
    }
    // get betas into array
    constexpr void getBetas(std::array<ValueT, nbeta> &b_arr) const { return getBetas(b_arr.begin(), b_arr.end()); }

    constexpr void setBeta(int i, ValueT beta)
    {
        int idx = 0;
        while (i >= Shape::nbetas[idx]) {
            i -= Shape::nbetas[idx];
            ++idx;
        }
        *(_beta_begins[idx] + i) = beta;
    }

    template <class IterT>
    constexpr void setBetas(IterT begin, const IterT end)
    {
        int idx = 0;
        while (begin < end) {
            const auto blocksize = (end - begin > Shape::nbetas[idx]) ? Shape::nbetas[idx] : end - begin;
            std::copy(begin, begin + blocksize, _beta_begins[idx]);
            begin += blocksize;
            ++idx;
        }
    }
    // set betas from array
    constexpr void setBetas(const std::array<ValueT, nbeta> &b_arr) { setBetas(b_arr.begin(), b_arr.end()); }
    /*
    void randomizeBetas(); // has to be changed maybe if we add beta that are not "normal" weights*/

    // --- Manage the variational parameters (which may contain a subset of beta and/or non-beta parameters),
    //     which exist only after that they are assigned to actual parameters in the network (e.g. betas)
    //int getNVariationalParameters() { return _nvp; }
    /*ValueT getVariationalParameter(const int &ivp);
    void getVariationalParameter(ValueT * vp);
    void setVariationalParameter(const int &ivp, const ValueT &vp);
    void setVariationalParameter(const ValueT * vp);
    */

    // shortcut for (connecting and) adding substrates
    //void enableDerivatives(bool flag_d1, bool flag_d2, bool flag_vd1);


    // Set initial parameters
    constexpr void setInput(int i, ValueT val) { input[i] = val; }
    template <class IterT>
    constexpr void setInput(IterT begin, const IterT end) { std::copy(begin, end, input.begin()); }
    constexpr void setInput(const std::array<ValueT, ninput> &in_arr) { input = in_arr; }


    // --- Propagation

    constexpr void FFPropagate()
    {
        using namespace detail;

        // fdwprop
        std::get<0>(_layers).ForwardInput(input, dflags);
        fwdprop_layers_impl(_layers, dflags, std::make_index_sequence<nlayer - 1>{});

        // backprop
        std::get<nlayer - 1>(_layers).BackwardOutput(dflags);
        backprop_layers_impl(_layers, dflags, std::make_index_sequence<nlayer - 1>{});

        // store backprop grads into vd1
        grad_layers_impl<0, nbeta>(_layers, input, _vd1, dflags, std::make_index_sequence<nlayer>{});
    }

    // Shortcut for computation: set input and get all values and derivatives with one calculations.
    // If some derivatives are not supported (substrate missing) the values will be leaved unchanged.
    //void evaluate(const ValueT * in, ValueT * out, ValueT * d1 = nullptr, ValueT * d2 = nullptr, ValueT * vd1 = nullptr);

    // --- Store FFNN on file
    //void storeOnFile(const char * filename, bool store_betas = true);
};
} // templ

#endif

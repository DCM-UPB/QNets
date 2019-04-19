#ifndef FFNN_NET_TEMPLNET_HPP
#define FFNN_NET_TEMPLNET_HPP

#include "qnets/tool/PackHelpers.hpp"

#include <array>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>

namespace templ
{

enum class ArrayACTF { Log };

class LogACTF // Test Array ACTF
{
public:

    template <class InputIt, class OutputIt>
    constexpr void f(InputIt cur, InputIt end, OutputIt out)
    {
        for (; cur < end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur)));
        }
    }

    template <class InputIt, class OutputIt>
    constexpr void fd1(InputIt cur, InputIt end, OutputIt out)
    {
        for (; cur < end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur))); // f
            *out = *out*(1. - *out); // fd1
        }
    }

    template <class InputIt, class OutputIt>
    constexpr void fd2(InputIt cur, InputIt end, OutputIt out)
    {
        for (; cur < end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur))); // f
            *out = *out*(1. - *out)*(1. - 2.*(*out)); // fd2
        }
    }

    template <class InputIt, class OutputIt>
    constexpr void fad(InputIt cur, InputIt end, OutputIt outf, OutputIt outfd1, OutputIt outfd2)
    {
        for (; cur < end; ++cur, ++outf, ++outfd1, ++outfd2) {
            *outf = 1./(1. + exp(-(*cur))); // f
            *outfd1 = *outf*(1. - *outf); // fd1
            *outfd2 = *outfd1*(1. - 2.*(*outf)); // fd2
        }
    }
};


// --- Configuration structs

// To configure "allocation" of derivative-related std::arrays
// You may still opt OUT of the derivative COMPUTATION dynamically
// at run time, but the pre-reserved memory will always be kept.
template <bool RESERVE_D1 = false, bool RESERVE_D2 = false, bool RESERVE_VD1 = false>
struct DerivConfig
{
    static constexpr bool d1 = RESERVE_D1;
    static constexpr bool d2 = RESERVE_D2;
    static constexpr bool vd1 = RESERVE_VD1;
};

// to pass non-input layers as variadic parameter pack
template <typename SizeT, SizeT N_IN, SizeT N_OUT, class ACTF>
struct LayerConfig
{
    static constexpr SizeT ninput = N_IN;
    static constexpr SizeT noutput = N_OUT;
    static constexpr SizeT nbeta = (N_IN + 1)*N_OUT;
    static constexpr SizeT nlink = N_IN*N_OUT;
    using ACTF_Type = ACTF;

    static constexpr SizeT size() { return noutput; }
};

template <class ValueT, class LayerConf, class DerivConf>
struct Layer: LayerConf
{
    static constexpr typename LayerConf::ACTF_Type actf{};
    std::array<ValueT, LayerConf::noutput> out;
    std::array<ValueT, LayerConf::nlink> d1;
    std::array<ValueT, LayerConf::nlink> d2;
    //std::array<ValueT, LayerConf::nvp> vd1;
    std::array<ValueT, LayerConf::nbeta> beta;
};


// --- Layer pack helpers (will be mostly obsolete with C++17)

// count total number of units across Layer pack (recursive)
template <typename SizeT>
constexpr SizeT countUnits() { return 0; } // terminate recursion
template <typename SizeT, class Layer, class ... LAYERS>
constexpr SizeT countUnits()
{
    return Layer::size() + countUnits<SizeT, LAYERS...>(); // recurse until all layer's units counted
}

// count total number of weights across Layer pack (recursive)
template <typename SizeT>
constexpr SizeT countBetas(SizeT/*prev_nunits*/) { return 0; } // terminate recursion
template <typename SizeT, class Layer, class ... LAYERS>
constexpr SizeT countBetas(SizeT prev_nunits)
{
    return Layer::size()*(prev_nunits + 1/*offset*/) + countBetas<SizeT, LAYERS...>(Layer::size()); // recurse
}

// check for empty layers across pack (use it "passing" first argument with initial boolean (usually true)
template <bool ret>
constexpr bool hasNoEmptyLayer() { return ret; } // terminate recursion
template <bool ret, class Layer, class ... LAYERS>
constexpr bool hasNoEmptyLayer()
{
    return hasNoEmptyLayer<(ret && Layer::size() > 0), LAYERS...>();
}


// --- The fully templated TemplNet FFNN

template <typename SizeT, typename ValueT, class DerivConf, class ... LayerConfs>
class TemplNet
{
private:
    // --- Static Setup

    // Layer tuple
    using LayerTuple = std::tuple<Layer<ValueT, LayerConfs, DerivConf>...>;
    LayerTuple _layers;

    // store some basics
    static constexpr SizeT _nlayer = pack::count<SizeT, LayerConfs...>();
    static constexpr SizeT _ninput = std::tuple_element<0, LayerTuple>::type::ninput;
    static constexpr SizeT _noutput = std::tuple_element<_nlayer - 1, LayerTuple>::type::noutput;

    // Some homogenous arrays to make access easier
    static constexpr std::array<SizeT, sizeof...(LayerConfs)> _nunit_shape{LayerConfs::size()...};
    static constexpr std::array<SizeT, sizeof...(LayerConfs)> _nbeta_shape{LayerConfs::nbeta...};
    //const std::array<ValueT *, sizeof...(LayerConfs)> _beta_begins;

    // Basic assertions
    static_assert(_nlayer > 1, "[TemplNet] nlayer <= 1");
    static_assert(hasNoEmptyLayer<(_ninput > 0), LayerConfs...>(), "[TemplNet] Layer pack contains empty Layer.");

public:
    // input array
    std::array<ValueT, _ninput> input;

    // convenience reference to output array
    const std::array<ValueT, _noutput> &output;

    bool flag_d1, flag_d2, flag_vd1;  // flags that indicate if derivatives should currently be calculated or not

public:
    constexpr TemplNet():
            _layers({Layer<ValueT, LayerConfs, DerivConf>()...}), input{}, output(std::get<_nlayer - 1>(_layers).out),
            flag_d1(DerivConf::d1), flag_d2(DerivConf::d2), flag_vd1(DerivConf::vd1) {}


    // --- Get information about the NN structure
    static constexpr SizeT getNLayer() { return _nlayer; }
    static constexpr SizeT getNInput() { return _ninput; }
    static constexpr SizeT getNOutput() { return _noutput; }
    static constexpr SizeT getNUnit() { return countUnits<SizeT, LayerConfs...>(); }
    static constexpr SizeT getNUnit(SizeT i) { return _nunit_shape[i]; }
    static constexpr const auto &getShape() { return _nunit_shape; }

    // const access to LayerTuple
    // allows to read information on per-layer basis
    constexpr const LayerTuple &getLayers() const { return _layers; }
    template <SizeT I>
    constexpr const auto &getLayer() const { return std::get<I>(_layers); }

    // --- const get Value Arrays
    constexpr const auto &getInput() const { return input; } // alternative const read of public input array
    constexpr const auto &getOutput() const { return std::get<_nlayer - 1>(_layers).out; } // get values of output layer
    constexpr const auto &getFirstDerivative() const { return std::get<_nlayer - 1>(_layers).d1; } // get derivative of output with respect to input
    constexpr const auto &getSecondDerivative() const { return std::get<_nlayer - 1>(_layers).d2; }

    // --- check derivative setup
    static constexpr bool allowsFirstDerivative() { return DerivConf::d1; }
    static constexpr bool allowsSecondDerivative() { return DerivConf::d2; }
    static constexpr bool allowsVariationalFirstDerivative() { return DerivConf::vd1; }

    constexpr bool hasFirstDerivative() const { return flag_d1; }
    constexpr bool hasSecondDerivative() const { return flag_d2; }
    constexpr bool hasVariationalFirstDerivative() const { return flag_vd1; }

    // --- Access Network Weights (Betas)
    static constexpr SizeT getNBeta() { return countBetas<SizeT, LayerConfs...>(getNInput()); }
    static constexpr SizeT getNBeta(SizeT i) { return _nbeta_shape[i]; }
    static constexpr const auto &getBetaShape() { return _nbeta_shape; }
    static constexpr SizeT getNLink() { return getNBeta() - getNUnit(); } // substract offsets
    static constexpr SizeT getNLink(SizeT i) { return _nbeta_shape[i] - _nunit_shape[i]; }

    constexpr ValueT getBeta(SizeT i) const
    {
        SizeT offset = 0;
        SizeT idx = 0;
        while (i < offset + getNUnit(idx)) {
            offset += getNUnit(idx);
            ++idx;
        }
        return 0;
        //return *(_beta_begins[idx] + (i - offset));
    }
    /*ValueT getBeta(const SizeT &ib);
    void getBeta(ValueT * beta);
    void setBeta(const SizeT &ib, const ValueT &beta);
    void setBeta(const ValueT * beta);
    void randomizeBetas(); // has to be changed maybe if we add beta that are not "normal" weights*/

    // --- Manage the variational parameters (which may contain a subset of beta and/or non-beta parameters),
    //     which exist only after that they are assigned to actual parameters in the network (e.g. betas)
    //SizeT getNVariationalParameters() { return _nvp; }
    /*ValueT getVariationalParameter(const SizeT &ivp);
    void getVariationalParameter(ValueT * vp);
    void setVariationalParameter(const SizeT &ivp, const ValueT &vp);
    void setVariationalParameter(const ValueT * vp);*/

    // shortcut for (connecting and) adding substrates
    //void enableDerivatives(bool flag_d1, bool flag_d2, bool flag_vd1);


    // Set initial parameters
    /*void setInput(const ValueT * in);
    void setInput(const SizeT &i, const ValueT &in);*/

    // --- Computation
    //void FFPropagate();

    // Shortcut for computation: set input and get all values and derivatives with one calculations.
    // If some derivatives are not supported (substrate missing) the values will be leaved unchanged.
    //void evaluate(const ValueT * in, ValueT * out, ValueT * d1 = nullptr, ValueT * d2 = nullptr, ValueT * vd1 = nullptr);


    // --- Get outputs
    /*
    void getOutput(ValueT * out);
    ValueT getOutput(const SizeT &i);

    void getFirstDerivative(ValueT ** d1);
    void getFirstDerivative(const SizeT &iu, ValueT * d1);  // iu is the unit index
    ValueT getFirstDerivative(const SizeT &iu, const SizeT &i1d); // i is the index of the output elemnet (i.e. unit=1, offset unit is meaningless), i1d the index of the input element

    void getSecondDerivative(ValueT ** d2);
    void getSecondDerivative(const SizeT &i, ValueT * d2);  // i is the output index
    ValueT getSecondDerivative(const SizeT &i, const SizeT &i2d); // i is the index of the output element, i2d the index of the input element

    void getVariationalFirstDerivative(ValueT ** vd1);
    void getVariationalFirstDerivative(const SizeT &i, ValueT * vd1);  // i is the output index
    ValueT getVariationalFirstDerivative(const SizeT &i, const SizeT &iv1d);  // i is the index of the output element, iv1d the index of the beta element
    */

    // --- Store FFNN on file
    //void storeOnFile(const char * filename, bool store_betas = true);
};

template <typename SizeT, typename ValueT, class DERIV_SETUP, class ... LAYERS>
constexpr std::array<SizeT, sizeof...(LAYERS)> TemplNet<SizeT, ValueT, DERIV_SETUP, LAYERS...>::_nunit_shape;
template <typename SizeT, typename ValueT, class DERIV_SETUP, class ... LAYERS>
constexpr std::array<SizeT, sizeof...(LAYERS)> TemplNet<SizeT, ValueT, DERIV_SETUP, LAYERS...>::_nbeta_shape;
} // templ

#endif

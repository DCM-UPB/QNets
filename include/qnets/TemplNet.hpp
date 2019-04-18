#ifndef FFNN_NET_TEMPLNET_HPP
#define FFNN_NET_TEMPLNET_HPP

#include "qnets/tool/PackHelpers.hpp"

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>

namespace templ
{

template <class IteratorT = double *>
class LogACTF // Test Array ACTF
{
public:

    static void f(IteratorT cur, IteratorT end, IteratorT out)
    {
        for (; cur < end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur)));
        }
    }

    static void fd1(IteratorT cur, IteratorT end, IteratorT out)
    {
        for (; cur < end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur))); // f
            *out = *out*(1. - *out); // fd1
        }
    }

    static void fd2(IteratorT cur, IteratorT end, IteratorT out)
    {
        for (; cur < end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur))); // f
            *out = *out*(1. - *out)*(1. - 2.*(*out)); // fd2
        }
    }

    static void fad(IteratorT cur, IteratorT end, IteratorT outf, IteratorT outfd1, IteratorT outfd2)
    {
        for (; cur < end; ++cur, ++outf, ++outfd1, ++outfd2) {
            *outf = 1./(1. + exp(-(*cur))); // f
            *outfd1 = *outf*(1. - *outf); // fd1
            *outfd2 = *outfd1*(1. - 2.*(*outf)); // fd2
        }
    }
};

// --- Configuration "structs" (for user)

// To configure "allocation" of derivative-related std::arrays
// You may still opt OUT of the derivative COMPUTATION dynamically
// at run time, but the pre-reserved memory will always be kept.
template <bool RESERVE_D1 = false, bool RESERVE_D2 = false, bool RESERVE_VD1 = false>
struct DerivSetup
{
    static constexpr bool d1 = RESERVE_D1;
    static constexpr bool d2 = RESERVE_D2;
    static constexpr bool vd1 = RESERVE_VD1;
};

// to pass non-input layers as variadic parameter pack
template <typename SizeT, SizeT NU, class ACTF>
struct Layer
{
    static constexpr SizeT size = NU;
    using actf = ACTF;
};


// --- Helpers

// count total number of units across Layer pack (recursive)
template <typename SizeT>
constexpr SizeT countUnits() { return 0; } // terminate recursion
template <typename SizeT, class Layer, class ... LAYERS>
constexpr SizeT countUnits()
{
    return Layer::size + countUnits<SizeT, LAYERS...>(); // recurse until all layer's units counted
}

// count total number of weights across Layer pack (recursive)
template <typename SizeT>
constexpr SizeT countNVP(SizeT/*prev_nunits*/) { return 0; } // terminate recursion
template <typename SizeT, class Layer, class ... LAYERS>
constexpr SizeT countNVP(SizeT prev_nunits)
{
    return Layer::size*(prev_nunits + 1/*offset*/) + countNVP<SizeT, LAYERS...>(Layer::size); // recurse
}

// check for empty layers across pack (use it "passing" first argument as true)
template <bool ret> constexpr bool hasNoEmptyLayer() { return ret; } // terminate recursion
template <bool ret, class Layer, class ... LAYERS>
constexpr bool hasNoEmptyLayer()
{
    return hasNoEmptyLayer<(ret && Layer::size > 0), LAYERS...>();
}

// make array of nunits
template<typename SizeT, class ... LAYERS>
constexpr auto makeNUnitsArray(SizeT input_nunits)
{
    return std::array<SizeT, 1 + pack::count<SizeT, LAYERS...>()> { input_nunits, LAYERS::size... };
}

// --- The fully templated TemplNet FFNN

template <typename SizeT, typename ValueT, class DERIV_SETUP, SizeT NU_IN, class ... LAYERS>
class TemplNet
{
public:
    // constexpr sizes
    static constexpr SizeT nlayer = 1 /*IN*/ + pack::count<SizeT, LAYERS...>();

    // check for mistakes (statically)
    static_assert(nlayer > 2, "[TemplNet] nlayer <= 2");
    static_assert(hasNoEmptyLayer<(NU_IN > 0), LAYERS...>(), "[TemplNet] Layer pack contains empty Layer.");

    static constexpr SizeT nunit_tot = NU_IN + countUnits<SizeT, LAYERS...>();
    static constexpr SizeT nvp_tot = countNVP<SizeT, LAYERS...>(NU_IN);
    static constexpr SizeT nlink_tot = nvp_tot - nunit_tot + NU_IN; // substract offsets

    // nunit array
    static constexpr std::array<SizeT, nlayer> nunits = makeNUnitsArray<SizeT, LAYERS...>(NU_IN);

    // input array
    std::array<ValueT, NU_IN> input{};

    // output array (const ref)
    const std::array<ValueT, nunits.back()> &output;

private:
    std::array<ValueT, nunits.back()> _output{};

    bool _flag_d1{}, _flag_d2{}, _flag_vd1{};  // flags that indicate if derivatives should currently be calculated or not

public:
    constexpr TemplNet(): output(_output), _flag_d1(DERIV_SETUP::d1), _flag_d2(DERIV_SETUP::d2), _flag_vd1(DERIV_SETUP::vd1) {}
    constexpr TemplNet(bool flag_d1, bool flag_d2, bool flag_vd1): TemplNet()
    {
        _flag_d1 = flag_d1;
        _flag_d2 = flag_d2;
        _flag_vd1 = flag_vd1;
    }

    // --- Get information about the NN structure
    /*
    SizeT getNLayers() { return _L.size(); }
    SizeT getNFedLayers() { return _L_fed.size(); }
    SizeT getNNeuralLayers() { return _L_nn.size(); }
    SizeT getNHiddenLayers() { return _L_nn.size() - 1; }

    SizeT getNInput() { return _L_in->getNInputUnits(); }
    SizeT getNOutput() { return _L_out->getNOutputNNUnits(); }
    SizeT getLayerSize(const SizeT &li) { return _L[li]->getNUnits(); }
    
    bool hasFirstDerivative() { return _flag_d1; }
    bool hasSecondDerivative() { return _flag_d2; }
    bool hasVariationalFirstDerivative() { return _flag_vd1; }
    */

    // --- Manage the betas, which exist only after that the FFNN has been connected
    //SizeT getNBeta();
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

    // --- Toggle the calculations of derivatives
    void enableFirstDerivatives() { _flag_d1 = true; }  // coordinates first derivatives
    void disableFirstDerivatives() { _flag_d1 = false; }
    void enableSecondDerivatives() { _flag_d2 = true; }  // coordinates second derivatives
    void disableSecondDerivatives() { _flag_d2 = false; }
    void enableVariationalFirstDerivatives() { _flag_vd1 = true; }  // variational first derivatives
    void disableVariationalFirstDerivatives() { _flag_vd1 = false; }

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
} // templ

#endif

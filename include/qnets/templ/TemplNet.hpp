#ifndef FFNN_NET_TEMPLNET_HPP
#define FFNN_NET_TEMPLNET_HPP

#include "qnets/tool/PackTools.hpp"
#include "qnets/tool/TupleTools.hpp"
#include "qnets/templ/Layer.hpp"
#include "qnets/templ/LayerPackTools.hpp"
#include "qnets/templ/DerivConfig.hpp"

#include <array>
#include <tuple>
#include <algorithm>
#include <numeric>

namespace templ
{

// --- The fully templated TemplNet FFNN

template <typename SizeT, typename ValueT, DerivConfig DCONF, class ... LayerConfs>
class TemplNet
{
private:
    // --- Static Setup

    using LayerConfTuple = std::tuple<LayerConfs...>; // used to compute some statics

    // store some basics (these stay private so they may be freely removed/replaced by constexpr method)
    static constexpr SizeT _nlayer = tupl::count<SizeT, LayerConfTuple>();
    static constexpr SizeT _ninput = std::tuple_element<0, LayerConfTuple>::type::ninput;
    static constexpr SizeT _noutput = std::tuple_element<_nlayer - 1, LayerConfTuple>::type::noutput;
    static constexpr SizeT _nbeta = lpack::countBetas<SizeT, LayerConfs...>(_ninput);
    static constexpr SizeT _nunit = lpack::countUnits<SizeT, LayerConfs...>();

    // Some static arrays to make access easier (needs to be defined again below the class, until C++17)
    // Therefore we keep the sizeof...() as size instead of using nlayer static variable
    static constexpr std::array<SizeT, sizeof...(LayerConfs)> _nunit_shape{LayerConfs::size()...};
    static constexpr std::array<SizeT, sizeof...(LayerConfs)> _nbeta_shape{LayerConfs::nbeta...};


    // --- Layer tuple

    using LayerTuple = std::tuple<Layer<SizeT, ValueT, _ninput, LayerConfs::ninput, LayerConfs::noutput, typename LayerConfs::ACTF_Type, DCONF>...>;
    LayerTuple _layers;

    // Static Output Deriv Array Sizes
    static constexpr SizeT _nd1_out = std::tuple_element<_nlayer - 1, LayerTuple>::type::nd1;
    static constexpr SizeT _nd2_out = std::tuple_element<_nlayer - 1, LayerTuple>::type::nd2;

    // Basic assertions
    static_assert(_nlayer > 1, "[TemplNet] nlayer <= 1");
    static_assert(lpack::hasNoEmptyLayer<(_ninput > 0), LayerConfs...>(), "[TemplNet] Layer pack contains empty Layer.");


    // --- Non-static

    // arrays of array.begin() pointers, for run-time public indexing
    const std::array<const ValueT *, _nlayer> _out_begins;
    const std::array<ValueT *, _nlayer> _beta_begins;

public:
    // static/dynamic derivative config
    static constexpr StaticDFlags<DCONF> dconf{}; // static derivative config
    DynamicDFlags dflags{DCONF}; // dynamic (opt-out) derivative config

    // input array
    std::array<ValueT, _ninput> input{};

    // convenient public const-references to output arrays
    const std::array<ValueT, _noutput> &output;
    const std::array<ValueT, _nd1_out> &out_d1;
    const std::array<ValueT, _nd2_out> &out_d2;

public:
    constexpr TemplNet():
            _layers{Layer<SizeT, ValueT, _ninput, LayerConfs::ninput, LayerConfs::noutput, typename LayerConfs::ACTF_Type, DCONF>{}...},
            _out_begins(tupl::make_fcont<std::array<const ValueT *, _nlayer>>(_layers, [](const auto &layer) { return layer.out.cbegin(); })),
            _beta_begins(tupl::make_fcont<std::array<ValueT *, _nlayer>>(_layers, [](auto &layer) { return layer.beta.begin(); })),
            output(std::get<_nlayer - 1>(_layers).out), out_d1{std::get<_nlayer - 1>(_layers).d1}, out_d2{std::get<_nlayer - 1>(_layers).d2} {}


    // --- Get information about the NN structure
    static constexpr SizeT getNLayer() { return _nlayer; }
    static constexpr SizeT getNInput() { return _ninput; }
    static constexpr SizeT getNOutput() { return _noutput; }
    static constexpr SizeT getNUnit() { return _nunit; }
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
    //constexpr const auto &getFirstDerivative() const { return out_d1; } // get derivative of output with respect to input
    //constexpr const auto &getSecondDerivative() const { return out_d2; }

    // --- check derivative setup
    static constexpr bool allowsFirstDerivative() { return dconf.d1; }
    static constexpr bool allowsSecondDerivative() { return dconf.d2; }
    static constexpr bool allowsVariationalFirstDerivative() { return dconf.vd1; }

    constexpr bool hasFirstDerivative() const { return dconf.d1 && dflags.d1(); }
    constexpr bool hasSecondDerivative() const { return dconf.d2 && dflags.d2(); }
    constexpr bool hasVariationalFirstDerivative() const { return dconf.vd1 && dflags.vd1(); }

    // --- Access Network Weights (Betas)
    static constexpr SizeT getNBeta() { return _nbeta; }
    static constexpr SizeT getNBeta(SizeT i) { return _nbeta_shape[i]; }
    static constexpr const auto &getBetaShape() { return _nbeta_shape; }
    static constexpr SizeT getNLink() { return _nbeta - _nunit; } // substract offsets
    static constexpr SizeT getNLink(SizeT i) { return _nbeta_shape[i] - _nunit_shape[i]; }

    constexpr ValueT getBeta(SizeT i) const
    {
        SizeT idx = 0;
        while (i >= _nbeta_shape[idx]) {
            i -= _nbeta_shape[idx];
            ++idx;
        }
        return *(_beta_begins[idx] + i);
    }

    template <class IterT>
    constexpr void getBetas(IterT begin, IterT end) const
    {
        SizeT idx = 0;
        while (begin < end) {
            std::copy(_beta_begins[idx], _beta_begins[idx] + _nbeta_shape[idx], begin);
            begin += _nbeta_shape[idx];
            ++idx;
        }
    }
    template <class IterT>
    constexpr void getBetas(IterT begin) const { return getBetas(begin, begin + getNBeta()); }

    void setBeta(SizeT i, ValueT beta)
    {
        SizeT idx = 0;
        while (i >= _nbeta_shape[idx]) {
            i -= _nbeta_shape[idx];
            ++idx;
        }
        *(_beta_begins[idx] + i) = beta;
    }

    template <class IterT>
    constexpr void setBeta(IterT begin, IterT end)
    {
        SizeT idx = 0;
        while (begin < end) {
            std::copy(begin, begin + _nbeta_shape[idx], _beta_begins[idx]);
            begin += _nbeta_shape[idx];
            ++idx;
        }
    }
    /*
    ValueT getBeta(const SizeT &ib);
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
    void setVariationalParameter(const ValueT * vp);
    */

    // shortcut for (connecting and) adding substrates
    //void enableDerivatives(bool flag_d1, bool flag_d2, bool flag_vd1);


    // Set initial parameters
    void setInput(SizeT i, ValueT val) { input[i] = val; }
    template <class IterT>
    void setInput(IterT begin) { std::copy(begin, begin+_ninput, input.begin()); }

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

template <typename SizeT, typename ValueT, DerivConfig DCONF, class ... LayerConfs>
constexpr std::array<SizeT, sizeof...(LayerConfs)> TemplNet<SizeT, ValueT, DCONF, LayerConfs...>::_nunit_shape;
template <typename SizeT, typename ValueT, DerivConfig DCONF, class ... LayerConfs>
constexpr std::array<SizeT, sizeof...(LayerConfs)> TemplNet<SizeT, ValueT, DCONF, LayerConfs...>::_nbeta_shape;
} // templ

#endif

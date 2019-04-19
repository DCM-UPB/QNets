#ifndef FFNN_NET_TEMPLNET_HPP
#define FFNN_NET_TEMPLNET_HPP

#include "qnets/tool/PackTools.hpp"
#include "qnets/tool/TupleTools.hpp"
#include "qnets/templ/Layer.hpp"
#include "qnets/templ/LayerPackTools.hpp"

#include <array>
#include <tuple>
#include <algorithm>
#include <numeric>

namespace templ
{

// --- The fully templated TemplNet FFNN

template <typename SizeT, typename ValueT, class DerivConf, class ... LayerConfs>
class TemplNet
{
private:
    // --- Static Setup

    // Layer tuples

    using LayerConfTuple = std::tuple<LayerConfs...>; // this one we can make static
    static constexpr LayerConfTuple _layerConfs{};

    using LayerTuple = std::tuple<Layer<ValueT, LayerConfs, DerivConf>...>;
    LayerTuple _layers;

    // store some basics
    static constexpr SizeT _nlayer = tupl::count<SizeT, LayerTuple>();
    static constexpr SizeT _ninput = std::tuple_element<0, LayerTuple>::type::ninput;
    static constexpr SizeT _noutput = std::tuple_element<_nlayer - 1, LayerTuple>::type::noutput;

    // Some static arrays to make access easier (needs to be defined again below the class, until C++17)
    // Therefore we keep the sizeof...() as size instead of using nlayer static variable
    static constexpr std::array<SizeT, sizeof...(LayerConfs)> _nunit_shape{LayerConfs::size()...};
    static constexpr std::array<SizeT, sizeof...(LayerConfs)> _nbeta_shape{LayerConfs::nbeta...};
    //static constexpr std::array<SizeT, sizeof...(LayerConfs)> _nbeta_offsets{tupl::make_fcont<std::array<SizeT, _nlayer>>(_layerConfs, [](const auto &layer) mutable { return layer.size();})};

    // arrays of array.begin() pointers, for run-time public indexing
    const std::array<ValueT *, _nlayer> _out_begins;
    const std::array<ValueT *, _nlayer> _beta_begins;

    // Basic assertions
    static_assert(_nlayer > 1, "[TemplNet] nlayer <= 1");
    static_assert(lpack::hasNoEmptyLayer<(_ninput > 0), LayerConfs...>(), "[TemplNet] Layer pack contains empty Layer.");

public:
    // input array
    std::array<ValueT, _ninput> input;

    // convenience reference to output array
    const std::array<ValueT, _noutput> &output;

    bool flag_d1, flag_d2, flag_vd1;  // flags that indicate if derivatives should currently be calculated or not

public:
    constexpr TemplNet():
            _layers{Layer<ValueT, LayerConfs, DerivConf>()...},
            _out_begins{tupl::make_fcont<std::array<ValueT *, _nlayer>>(_layers, [](auto &layer) { return layer.out.begin(); })},
            _beta_begins{tupl::make_fcont<std::array<ValueT *, _nlayer>>(_layers, [](auto &layer) { return layer.beta.begin(); })},
            input{}, output(std::get<_nlayer - 1>(_layers).out),
            flag_d1(DerivConf::d1), flag_d2(DerivConf::d2), flag_vd1(DerivConf::vd1) {}


    // --- Get information about the NN structure
    static constexpr SizeT getNLayer() { return _nlayer; }
    static constexpr SizeT getNInput() { return _ninput; }
    static constexpr SizeT getNOutput() { return _noutput; }
    static constexpr SizeT getNUnit() { return lpack::countUnits<SizeT, LayerConfs...>(); }
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
    static constexpr SizeT getNBeta() { return lpack::countBetas<SizeT, LayerConfs...>(getNInput()); }
    static constexpr SizeT getNBeta(SizeT i) { return _nbeta_shape[i]; }
    static constexpr const auto &getBetaShape() { return _nbeta_shape; }
    static constexpr SizeT getNLink() { return getNBeta() - getNUnit(); } // substract offsets
    static constexpr SizeT getNLink(SizeT i) { return _nbeta_shape[i] - _nunit_shape[i]; }

    /*constexpr ValueT getBeta(SizeT i) const
    {
        SizeT idx = 0;
        while (i < _nbeta_offsets[idx]) {
            ++idx;
        }
        return *(_beta_begins[idx] + (i - _nbeta_offsets[idx]));
    }*/
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
